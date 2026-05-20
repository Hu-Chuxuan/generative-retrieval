#!/usr/bin/env python3
"""
Gemma-4-E4B-it ablation: five variants controlled by --variant flag.

  advanced-full         Advanced prompt  +  full pool (100 products with random fillers)
  advanced-esci         Advanced prompt  +  ESCI candidates only (no fillers)
  baseline-esci         Baseline prompt  +  ESCI candidates only (no fillers)
  advanced-full-fewshot Advanced prompt + 3 few-shot examples + full pool
  advanced-esci-fewshot Advanced prompt + 3 few-shot examples + ESCI candidates only

Usage:
  python eval_gemma_e4b_ablation.py --variant advanced-full         --output results/gemma_e4b_adv_full.csv
  python eval_gemma_e4b_ablation.py --variant advanced-esci         --output results/gemma_e4b_adv_esci.csv
  python eval_gemma_e4b_ablation.py --variant baseline-esci         --output results/gemma_e4b_base_esci.csv
  python eval_gemma_e4b_ablation.py --variant advanced-full-fewshot --output results/gemma_e4b_adv_full_fewshot.csv
  python eval_gemma_e4b_ablation.py --variant advanced-esci-fewshot --output results/gemma_e4b_adv_esci_fewshot.csv
"""

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATA_DIR      = Path(__file__).parent.parent / "shopping_queries_dataset"
MODEL         = "google/gemma-4-E4B-it"
MAX_MODEL_LEN = 32768
POOL_SIZE     = 100          # only used for advanced-full
_MAX_FIELD_CHARS = 500

_FIELDS = ["product_id", "product_title", "product_brand", "product_color",
           "product_bullet_point", "product_description"]

# ── prompts ──────────────────────────────────────────────────────────────────

ADVANCED_SYSTEM = (
    "You are an Amazon product search expert. A product is relevant ONLY if it "
    "satisfies EVERY requirement in the query. Treat all attributes as hard filters: "
    "wrong color, wrong product type, or violating a 'without X' / 'no X' constraint "
    "makes a product irrelevant regardless of other similarities."
)

ADVANCED_USER = """\
From the list of {pool_size} products below, identify the {n_select} that EXACTLY \
match ALL requirements of the query. Each attribute is a hard filter.

Query: {query}

Matching rules:
- "without X" / "no X": product must NOT have or be X
- Color, size, material: must match exactly, not approximately
- Product type: must be the exact type requested, not a substitute

Products:
{product_list}

Output ONLY a comma-separated list of exactly {n_select} product numbers, \
with no explanation."""

BASELINE_SYSTEM = "You are a product search expert. Follow instructions exactly."

BASELINE_USER = """\
Given the customer search query below, identify the {n_select} most relevant products \
from the list of {pool_size} products.

Query: {query}

Products:
{product_list}

Output ONLY a comma-separated list of exactly {n_select} product numbers, \
with no explanation."""

# ── few-shot examples ─────────────────────────────────────────────────────────
# Each tuple: (user_message, assistant_answer)
# Three examples targeting the key failure modes:
#   1. "without X" negative constraint (syringe without needle)
#   2. Exact color/spec match (black hair dye, not brown)
#   3. Product type specificity (eye cream ≠ hair lightener)

_FS_TEMPLATE = """\
From the list of {n} products below, identify the {k} that EXACTLY \
match ALL requirements of the query. Each attribute is a hard filter.

Query: {query}

Matching rules:
- "without X" / "no X": product must NOT have or be X
- Color, size, material: must match exactly, not approximately
- Product type: must be the exact type requested, not a substitute

Products:
{products}

Output ONLY a comma-separated list of exactly {k} product numbers, \
with no explanation."""

FEW_SHOT_EXAMPLES = [
    # Example 1: negative constraint — product 1 has a needle, product 3 is wrong type
    (
        _FS_TEMPLATE.format(
            n=3, k=1, query="syringe without needle",
            products="\n\n".join([
                "1. BD 10mL Luer-Lok Tip Syringe with Needle\n"
                "   Brand: BD\n   Color: clear\n"
                "   Bullet points: Sterile single-use syringe with attached needle.\n"
                "   Description: ",
                "2. NORM-JECT 10mL All-Plastic Syringe without Needle\n"
                "   Brand: NORM-JECT\n   Color: clear\n"
                "   Bullet points: Luer slip tip, no needle included. Ideal for dispensing liquids.\n"
                "   Description: ",
                "3. 3M Nexcare Hypoallergenic Medical Tape\n"
                "   Brand: 3M\n   Color: white\n"
                "   Bullet points: Gentle tape for sensitive skin. Not a syringe.\n"
                "   Description: ",
            ]),
        ),
        "2",
    ),
    # Example 2: exact color — products 1 is brown (wrong color), 2 & 3 are black (correct)
    (
        _FS_TEMPLATE.format(
            n=3, k=2, query="#1 black permanent hair dye",
            products="\n\n".join([
                "1. Garnier Nutrisse Cream, 20 Golden Brown\n"
                "   Brand: Garnier\n   Color: 20 Golden Brown\n"
                "   Bullet points: Permanent color, 100% gray coverage, nourishing formula.\n"
                "   Description: ",
                "2. L'Oreal Paris Excellence Creme, 1 Black\n"
                "   Brand: L'Oreal Paris\n   Color: 1 Black\n"
                "   Bullet points: Permanent hair color, triple protection, 100% gray coverage.\n"
                "   Description: ",
                "3. Revlon Colorsilk Permanent Hair Color, 10 Black\n"
                "   Brand: Revlon\n   Color: 10 Black\n"
                "   Bullet points: Ammonia-free permanent formula, 100% gray coverage.\n"
                "   Description: ",
            ]),
        ),
        "2, 3",
    ),
    # Example 3: product type — product 1 is a hair lightener, product 3 is a supplement
    (
        _FS_TEMPLATE.format(
            n=3, k=1, query="caffeine eye cream",
            products="\n\n".join([
                "1. Tints of Nature 3-in-1 Hair Lightener Kit\n"
                "   Brand: Tints of Nature\n   Color: white\n"
                "   Bullet points: Ammonia-free lightener. Caffeine-infused for shine. Lightens hair up to 7 shades.\n"
                "   Description: ",
                "2. No7 Protect & Perfect Intense Advanced Eye Cream\n"
                "   Brand: No7\n   Color: \n"
                "   Bullet points: Contains caffeine to reduce puffiness and dark circles. Eye area cream.\n"
                "   Description: ",
                "3. Optimum Nutrition Essential Caffeine Tablets, 200mg\n"
                "   Brand: Optimum Nutrition\n   Color: \n"
                "   Bullet points: 200mg caffeine per tablet. Dietary supplement, not a topical cream.\n"
                "   Description: ",
            ]),
        ),
        "2",
    ),
]

# estimated token overhead of all three few-shot turns (~900 tokens × 3 chars/token)
_FEW_SHOT_CHAR_OVERHEAD = 2700

VARIANTS = {
    "advanced-full":         (ADVANCED_SYSTEM, ADVANCED_USER, True,  False),
    "advanced-esci":         (ADVANCED_SYSTEM, ADVANCED_USER, False, False),
    "baseline-esci":         (BASELINE_SYSTEM, BASELINE_USER, False, False),
    "advanced-full-fewshot": (ADVANCED_SYSTEM, ADVANCED_USER, True,  True),
    "advanced-esci-fewshot": (ADVANCED_SYSTEM, ADVANCED_USER, False, True),
}

# ── data loading ──────────────────────────────────────────────────────────────

def load_data(locale: str, split: str, num_queries: int | None):
    examples = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_examples.parquet")
    products = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_products.parquet")
    df = pd.merge(examples, products, how="left", on=["product_locale", "product_id"])
    df = df[
        (df["large_version"] == 1)
        & (df["split"] == split)
        & (df["product_locale"] == locale)
    ].fillna("")
    if num_queries is not None:
        query_ids = df["query_id"].unique()[:num_queries]
        df = df[df["query_id"].isin(query_ids)]
    print(f"Loaded {len(df):,} candidate rows | {df['query_id'].nunique():,} queries")
    all_products = products[products["product_locale"] == locale].fillna("").reset_index(drop=True)
    print(f"Full product pool: {len(all_products):,} products")
    return df.reset_index(drop=True), all_products


def group_by_query(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [grp.reset_index(drop=True) for _, grp in df.groupby("query_id", sort=False)]


def build_pool(grp: pd.DataFrame, all_products: pd.DataFrame,
               rng: np.random.Generator, add_fillers: bool) -> pd.DataFrame:
    candidates = grp[_FIELDS + ["esci_label"]].copy()
    if not add_fillers:
        return candidates.iloc[rng.permutation(len(candidates))].reset_index(drop=True)
    candidate_ids = set(grp["product_id"])
    n_fill = max(0, POOL_SIZE - len(grp))
    non_candidates = all_products[~all_products["product_id"].isin(candidate_ids)]
    fill_idx = rng.choice(len(non_candidates), size=min(n_fill, len(non_candidates)), replace=False)
    fillers = non_candidates.iloc[fill_idx][_FIELDS].copy()
    fillers["esci_label"] = "I"
    pool = pd.concat([candidates, fillers], ignore_index=True)
    return pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)


def build_message(grp: pd.DataFrame, pool: pd.DataFrame,
                  system_prompt: str, user_template: str,
                  use_few_shot: bool) -> list[dict]:
    n_select = int((grp["esci_label"] == "E").sum()) or 1
    extra = _FEW_SHOT_CHAR_OVERHEAD if use_few_shot else 0
    char_budget = int((MAX_MODEL_LEN - 512) * 3.0) - extra
    chars_used = len(system_prompt) + 400

    lines = []
    n = len(pool)
    for i, row in pool.iterrows():
        remaining = n - i
        per_product = max(100, (char_budget - chars_used) // remaining)
        field_chars = max(30, min(_MAX_FIELD_CHARS, (per_product - 150) // 2))
        bp   = str(row["product_bullet_point"])[:field_chars]
        desc = str(row["product_description"])[:field_chars]
        entry = "\n".join([
            f"{i + 1}. {row['product_title']}",
            f"   Brand: {row['product_brand']}",
            f"   Color: {row['product_color']}",
            f"   Bullet points: {bp}",
            f"   Description: {desc}",
        ])
        lines.append(entry)
        chars_used += len(entry) + 2

    user = user_template.format(
        query=grp["query"].iloc[0],
        n_select=n_select,
        pool_size=len(pool),
        product_list="\n\n".join(lines),
    )
    messages = [{"role": "system", "content": system_prompt}]
    if use_few_shot:
        for fs_user, fs_answer in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user",      "content": fs_user})
            messages.append({"role": "assistant", "content": fs_answer})
    messages.append({"role": "user", "content": user})
    return messages


def parse_selection(text: str, pool_size: int, n_select: int) -> list[int]:
    numbers = re.findall(r"\d+", text)
    seen, selected = set(), []
    for n in numbers:
        idx = int(n) - 1
        if 0 <= idx < pool_size and idx not in seen:
            seen.add(idx)
            selected.append(idx)
        if len(selected) == n_select:
            break
    return selected


def get_ttft_ms(output) -> float | None:
    try:
        m = output.metrics
        if m.first_token_time is not None and m.first_scheduled_time is not None:
            return (m.first_token_time - m.first_scheduled_time) * 1000
    except Exception:
        pass
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant",        required=True, choices=list(VARIANTS))
    p.add_argument("--model",          default=MODEL)
    p.add_argument("--locale",         default="us", choices=["us", "es", "jp"])
    p.add_argument("--split",          default="test", choices=["train", "test"])
    p.add_argument("--num-queries",    type=int, default=500)
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--tensor-parallel",type=int, default=1)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--output",         type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    system_prompt, user_template, add_fillers, use_few_shot = VARIANTS[args.variant]

    print(f"Variant      : {args.variant}")
    print(f"Add fillers  : {add_fillers}  (pool_size={'up to ' + str(POOL_SIZE) if add_fillers else 'ESCI candidates only'})")
    print(f"Few-shot     : {use_few_shot}")

    df, all_products = load_data(args.locale, args.split, args.num_queries)
    groups = group_by_query(df)

    rng   = np.random.default_rng(args.seed)
    pools = [build_pool(grp, all_products, rng, add_fillers) for grp in groups]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _chk = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True)
    print(f"Tokenizer sanity: {len(tokenizer.encode(_chk))} tokens for test msg (expected >5)")
    assert len(tokenizer.encode(_chk)) > 5, "Tokenizer broken"

    msgs = [build_message(grp, pool, system_prompt, user_template, use_few_shot)
            for grp, pool in zip(groups, pools)]

    # pre-filter over-length prompts
    valid_idx = []
    for i, msg in enumerate(msgs):
        try:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(text)) <= MAX_MODEL_LEN:
                valid_idx.append(i)
        except Exception:
            valid_idx.append(i)
    n_skipped = len(msgs) - len(valid_idx)
    if n_skipped:
        print(f"Skipped {n_skipped} queries exceeding {MAX_MODEL_LEN}-token limit")
        msgs   = [msgs[i]   for i in valid_idx]
        groups = [groups[i] for i in valid_idx]
        pools  = [pools[i]  for i in valid_idx]

    print("\n--- SAMPLE PROMPT (first 800 chars) ---")
    print(msgs[0][-1]["content"][:800], "...")
    print("---------------------------------------\n")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    _CSV_FIELDS = ["query_id", "query", "n_exact_gt", "product_id", "product_title",
                   "esci_label", "type", "pred_rank", "precision", "recall", "f1", "latency_ms"]
    csv_fh = open(args.output, "w", newline="") if args.output else None
    csv_w  = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS) if csv_fh else None
    if csv_w:
        csv_w.writeheader()

    precisions, recalls, f1s, latencies = [], [], [], []
    t0 = time.time()

    for i in tqdm(range(0, len(msgs), args.batch_size), desc="Querying"):
        batch   = msgs[i : i + args.batch_size]
        t_batch = time.time()
        outputs = llm.chat(messages=batch, sampling_params=sampling_params)
        wall_per_query = (time.time() - t_batch) / len(batch) * 1000

        for j, out in enumerate(outputs):
            grp  = groups[i + j]
            pool = pools[i + j]

            text     = out.outputs[0].text
            n_select = int((grp["esci_label"] == "E").sum()) or 1
            selected = parse_selection(text, len(pool), n_select)

            exact_ids    = set(grp.loc[grp["esci_label"] == "E", "product_id"])
            selected_ids = {pool.loc[idx, "product_id"] for idx in selected}
            tp = len(exact_ids & selected_ids)

            prec = tp / len(selected_ids) if selected_ids else 0.0
            rec  = tp / len(exact_ids)    if exact_ids    else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            ttft = get_ttft_ms(out) or wall_per_query

            precisions.append(prec); recalls.append(rec)
            f1s.append(f1);          latencies.append(ttft)

            if csv_w:
                qid   = grp["query_id"].iloc[0]
                query = grp["query"].iloc[0]
                base  = {"query_id": qid, "query": query, "n_exact_gt": len(exact_ids),
                         "precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4), "latency_ms": round(ttft, 2)}
                for rank_pos, idx in enumerate(selected):
                    pid = pool.loc[idx, "product_id"]
                    csv_w.writerow({**base,
                        "product_id":    pid,
                        "product_title": pool.loc[idx, "product_title"],
                        "esci_label":    pool.loc[idx, "esci_label"],
                        "type":          "TP" if pid in exact_ids else "FP",
                        "pred_rank":     rank_pos + 1,
                    })
                pid_to_title = {r["product_id"]: r["product_title"] for _, r in grp.iterrows()}
                for pid in exact_ids - selected_ids:
                    csv_w.writerow({**base,
                        "product_id":    pid,
                        "product_title": pid_to_title.get(pid, ""),
                        "esci_label":    "E",
                        "type":          "FN",
                        "pred_rank":     "",
                    })
        if csv_fh:
            csv_fh.flush()

    if csv_fh:
        csv_fh.close()

    elapsed = time.time() - t0
    print(f"\nInference: {len(msgs):,} queries in {elapsed:.1f}s")
    print("\n" + "=" * 50)
    print(f"Model  : {args.model}")
    print(f"Variant: {args.variant}")
    print("=" * 50)
    print(f"  {'Precision':<14}  {np.mean(precisions):.4f}")
    print(f"  {'Recall':<14}  {np.mean(recalls):.4f}")
    print(f"  {'F1':<14}  {np.mean(f1s):.4f}")
    print(f"  {'Latency (ms)':<14}  {np.mean(latencies):.1f}")
    print("=" * 50)
    if args.output:
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
