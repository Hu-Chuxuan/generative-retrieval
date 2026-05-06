#!/usr/bin/env python3
"""
Qwen3-32B baseline: Exact-class retrieval from a pool of 5000 products.

Same task as eval_qwen_recall_precision.py but scaled to Qwen3-32B's 128K
context window, allowing ~5000 product titles per query.

Metrics: Precision, Recall, F1, Latency (TTFT in ms).

Usage
  python eval_qwen32b_recall_precision.py --num-queries 200 --output results/qwen32b_rp.csv
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
MODEL         = "Qwen/Qwen3-32B"
POOL_SIZE     = 3000
MAX_MODEL_LEN = 40960

SYSTEM_PROMPT = "You are a product search expert. Follow instructions exactly."

USER_TEMPLATE = """\
Given the customer search query below, identify the {n_select} most relevant products \
from the list of {pool_size} products.

Query: {query}

Products:
{product_list}

Output ONLY a comma-separated list of exactly {n_select} product numbers, \
with no explanation."""


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


def build_pool(grp: pd.DataFrame, all_products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    candidate_ids = set(grp["product_id"])
    n_fill = max(0, POOL_SIZE - len(grp))
    non_candidates = all_products[~all_products["product_id"].isin(candidate_ids)]
    fill_idx = rng.choice(len(non_candidates), size=min(n_fill, len(non_candidates)), replace=False)
    fillers = non_candidates.iloc[fill_idx][["product_id", "product_title"]].copy()
    fillers["esci_label"] = "I"
    candidates = grp[["product_id", "product_title", "esci_label"]].copy()
    pool = pd.concat([candidates, fillers], ignore_index=True)
    return pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)


def build_message(grp: pd.DataFrame, pool: pd.DataFrame) -> list[dict]:
    n_select = int((grp["esci_label"] == "E").sum()) or 1
    # 4 chars ≈ 1 token; reserve 512 tokens for template + output
    char_budget = int((MAX_MODEL_LEN - 512) * 3.0)
    chars_used = len(SYSTEM_PROMPT) + 300  # rough template overhead

    lines = []
    shown = 0
    for i, row in pool.iterrows():
        line = f"{i + 1}. {row['product_title']}"
        if chars_used + len(line) + 1 > char_budget:
            break
        lines.append(line)
        chars_used += len(line) + 1
        shown += 1

    user = USER_TEMPLATE.format(
        query=grp["query"].iloc[0],
        n_select=n_select,
        pool_size=shown,
        product_list="\n".join(lines),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]


def parse_selection(text: str, pool_size: int, n_select: int) -> list[int]:
    if text and "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default=MODEL)
    p.add_argument("--locale",          default="us", choices=["us", "es", "jp"])
    p.add_argument("--split",           default="test", choices=["train", "test"])
    p.add_argument("--num-queries",     type=int, default=200)
    p.add_argument("--batch-size",      type=int, default=1)
    p.add_argument("--thinking",        action="store_true")
    p.add_argument("--tensor-parallel", type=int, default=4,
                   help="32B model requires multiple GPUs; 4x H100 recommended")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--output",          type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df, all_products = load_data(args.locale, args.split, args.num_queries)
    groups = group_by_query(df)

    rng = np.random.default_rng(args.seed)
    pools = [build_pool(grp, all_products, rng) for grp in groups]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _chk = tokenizer.apply_chat_template([{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True)
    print(f"Tokenizer sanity: {len(tokenizer.encode(_chk))} tokens for test msg (expected >5)")
    assert len(tokenizer.encode(_chk)) > 5, "Tokenizer broken"
    msgs  = [build_message(grp, pool) for grp, pool in zip(groups, pools)]

    token_limit = MAX_MODEL_LEN
    valid_idx = []
    for i, msg in enumerate(msgs):
        try:
            try:
                text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            tc = len(tokenizer.encode(text))
            if tc <= token_limit:
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
    )

    if args.thinking:
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512)
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    chat_template_kwargs = {"enable_thinking": args.thinking}

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
        outputs = llm.chat(
            messages=batch,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
        wall_per_query = (time.time() - t_batch) / len(batch) * 1000

        for j, out in enumerate(outputs):
            grp  = groups[i + j]
            pool = pools[i + j]

            text      = out.outputs[0].text
            n_select  = int((grp["esci_label"] == "E").sum()) or 1
            selected  = parse_selection(text, len(pool), n_select)

            exact_ids    = set(grp.loc[grp["esci_label"] == "E", "product_id"])
            selected_ids = {pool.loc[idx, "product_id"] for idx in selected}
            tp = len(exact_ids & selected_ids)

            prec = tp / len(selected_ids) if selected_ids else 0.0
            rec  = tp / len(exact_ids)    if exact_ids    else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            ttft = get_ttft_ms(out) or wall_per_query

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            latencies.append(ttft)

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
    print(f"Model     : {args.model}")
    print(f"Pool size : {POOL_SIZE}  |  Locale: {args.locale}  |  Queries: {len(groups):,}")
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
