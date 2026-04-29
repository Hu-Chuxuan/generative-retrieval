#!/usr/bin/env python3
"""
Large-pool listwise ranking with Gemma.

Each query is presented with a pool of POOL_SIZE products (default 50):
  - the ~20 ESCI-labeled candidates (shuffled in)
  - ~30 randomly sampled distractor products

The model must SELECT and RANK the N most relevant products (N = number of
ESCI candidates for that query). Full product in fo is shown (title, brand,
color, bullet points, description) — same fields as eval_zero_shot.py.

At 100 products × ~300 tokens each, prompts are ~30K tokens, just within
the 32K context window. Reduce --pool-size if you hit context limit errors.

Evaluation maps each selection back to its ESCI label (distractors get 0).
NDCG@K and Recall@K are computed over the full pool.

Usage
  python eval_gemma_large_pool.py --num-queries 200 --output results/gemma_large_pool.csv
"""

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score as sklearn_ndcg
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "shopping_queries_dataset"
MODEL          = "google/gemma-4-E4B-it"
LABEL_MAP      = {"E": 3, "S": 2, "C": 1, "I": 0}
POOL_SIZE       = 100  # total products shown per query (~20 candidates + distractors)
_MAX_FIELD_CHARS = 500  # truncate bullet_point and description per product

SYSTEM_PROMPT = "You are a product search expert. Follow instructions exactly."

USER_TEMPLATE = """\
Given the customer search query below, select and rank the {n_select} most relevant products \
from the list of {pool_size} products.

Query: {query}

Products:
{product_list}

Output ONLY a comma-separated list of exactly {n_select} product numbers \
(e.g. "42, 7, 315, ..."), ranked from most to least relevant."""


# ── Data ──────────────────────────────────────────────────────────────────────
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
    print(f"Loaded {len(df):,} candidate rows | {df['query_id'].nunique():,} queries | "
          f"locale={locale} split={split} avg_depth={len(df)/df['query_id'].nunique():.1f}")

    all_products = products[products["product_locale"] == locale].fillna("").reset_index(drop=True)
    print(f"Full product pool: {len(all_products):,} products")
    return df.reset_index(drop=True), all_products


def group_by_query(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [grp.reset_index(drop=True) for _, grp in df.groupby("query_id", sort=False)]


# ── Build pool ────────────────────────────────────────────────────────────────
def build_pool(
    grp: pd.DataFrame,
    all_products: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Returns a shuffled pool of POOL_SIZE products and the set of candidate product_ids.
    Pool columns: product_id, product_title, esci_label (0 for distractors).
    """
    candidate_ids = set(grp["product_id"])
    n_fill = max(0, POOL_SIZE - len(grp))

    # sample distractors from the full locale pool, excluding candidates
    _FIELDS = ["product_id", "product_title", "product_brand", "product_color",
               "product_bullet_point", "product_description"]
    non_candidates = all_products[~all_products["product_id"].isin(candidate_ids)]
    fill_idx = rng.choice(len(non_candidates), size=min(n_fill, len(non_candidates)), replace=False)
    fillers = non_candidates.iloc[fill_idx][_FIELDS].copy()
    fillers["esci_label"] = "I"  # distractors are irrelevant

    candidates = grp[_FIELDS + ["esci_label"]].copy()
    pool = pd.concat([candidates, fillers], ignore_index=True)
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    return pool, candidate_ids


# ── Prompts ───────────────────────────────────────────────────────────────────
def build_messages(
    groups: list[pd.DataFrame],
    pools: list[pd.DataFrame],
) -> list[list[dict]]:
    msgs = []
    for grp, pool in zip(groups, pools):
        lines = []
        for i, row in pool.iterrows():
            bp   = str(row["product_bullet_point"])[:_MAX_FIELD_CHARS]
            desc = str(row["product_description"])[:_MAX_FIELD_CHARS]
            parts = [
                f"{i + 1}. {row['product_title']}",
                f"Brand: {row['product_brand']}",
                f"Color: {row['product_color']}",
                f"Bullet points: {bp}",
                f"Description: {desc}",
            ]
            lines.append("\n   ".join(parts))
        product_list = "\n\n".join(lines)
        user = USER_TEMPLATE.format(
            query=grp["query"].iloc[0],
            n_select=len(grp),
            pool_size=len(pool),
            product_list=product_list,
        )
        msgs.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ])
    return msgs


# ── Parse output ──────────────────────────────────────────────────────────────
def parse_selection(text: str, pool_size: int, n_select: int) -> list[int]:
    """Returns 0-based pool indices in predicted rank order (length = n_select)."""
    numbers = re.findall(r"\d+", text)
    seen, selected = set(), []
    for n in numbers:
        idx = int(n) - 1
        if 0 <= idx < pool_size and idx not in seen:
            seen.add(idx)
            selected.append(idx)
        if len(selected) == n_select:
            break
    # pad with random unselected if model output too short
    for i in range(pool_size):
        if len(selected) == n_select:
            break
        if i not in seen:
            selected.append(i)
            seen.add(i)
    return selected


# ── Metrics ───────────────────────────────────────────────────────────────────
def ndcg_at_k(
    groups: list[pd.DataFrame],
    pools: list[pd.DataFrame],
    selections: list[list[int]],
    k: int,
) -> float:
    scores = []
    for grp, pool, selection in zip(groups, pools, selections):
        true_rel = np.array([LABEL_MAP.get(l, 0) for l in pool["esci_label"]])
        if true_rel.max() == 0:
            continue
        pred_rel = np.zeros(len(pool))
        for rank_pos, idx in enumerate(selection):
            pred_rel[idx] = len(selection) - rank_pos
        scores.append(sklearn_ndcg([true_rel], [pred_rel], k=k))
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(
    groups: list[pd.DataFrame],
    pools: list[pd.DataFrame],
    selections: list[list[int]],
    k: int,
) -> float:
    scores = []
    for grp, pool, selection in zip(groups, pools, selections):
        total_exact = (pool["esci_label"] == "E").sum()
        if total_exact == 0:
            continue
        top_k = set(selection[:k])
        exact_in_top_k = sum(1 for i in top_k if pool.loc[i, "esci_label"] == "E")
        scores.append(exact_in_top_k / total_exact)
    return float(np.mean(scores)) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default=MODEL)
    p.add_argument("--locale",          default="us", choices=["us", "es", "jp"])
    p.add_argument("--split",           default="test", choices=["train", "test"])
    p.add_argument("--num-queries",     type=int, default=200)
    p.add_argument("--pool-size",       type=int, default=POOL_SIZE,
                   help="Total products shown per query (candidates + distractors). 100 ≈ 30K tokens.")
    p.add_argument("--batch-size",      type=int, default=8)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--output",          type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global POOL_SIZE
    POOL_SIZE = args.pool_size

    df, all_products = load_data(args.locale, args.split, args.num_queries)
    groups = group_by_query(df)

    rng = np.random.default_rng(args.seed)
    pools = [build_pool(grp, all_products, rng)[0] for grp in groups]

    msgs = build_messages(groups, pools)

    print("\n--- SAMPLE PROMPT (first 1000 chars) ---")
    print(msgs[0][-1]["content"][:1000], "...")
    print("----------------------------------------\n")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=32768,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    selections: list[list[int]] = []
    t0 = time.time()

    _CSV_FIELDS = ["query_id", "query", "product_id", "product_title", "esci_label",
                   "is_candidate", "pred_rank"]
    csv_fh = open(args.output, "w", newline="") if args.output else None
    csv_w  = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS) if csv_fh else None
    if csv_w:
        csv_w.writeheader()

    for i in tqdm(range(0, len(msgs), args.batch_size), desc="Ranking queries"):
        batch   = msgs[i : i + args.batch_size]
        outputs = llm.chat(
            messages=batch,
            sampling_params=sampling_params,
        )
        for j, out in enumerate(outputs):
            text      = out.outputs[0].text
            grp       = groups[i + j]
            pool      = pools[i + j]
            selection = parse_selection(text, len(pool), len(grp))
            selections.append(selection)
            if csv_w:
                qid, query = grp["query_id"].iloc[0], grp["query"].iloc[0]
                candidate_ids = set(grp["product_id"])
                for rank_pos, idx in enumerate(selection):
                    csv_w.writerow({
                        "query_id":     qid,
                        "query":        query,
                        "product_id":   pool.loc[idx, "product_id"],
                        "product_title": pool.loc[idx, "product_title"],
                        "esci_label":   pool.loc[idx, "esci_label"],
                        "is_candidate": pool.loc[idx, "product_id"] in candidate_ids,
                        "pred_rank":    rank_pos + 1,
                    })
        if csv_fh:
            csv_fh.flush()

    if csv_fh:
        csv_fh.close()

    elapsed = time.time() - t0
    print(f"\nInference: {len(msgs):,} queries in {elapsed:.1f}s ({len(msgs)/elapsed:.1f} q/s)")

    results = {
        "NDCG@5":    ndcg_at_k(groups, pools, selections, k=5),
        "NDCG@10":   ndcg_at_k(groups, pools, selections, k=10),
        "Recall@5":  recall_at_k(groups, pools, selections, k=5),
        "Recall@10": recall_at_k(groups, pools, selections, k=10),
    }

    print("\n" + "=" * 50)
    print(f"Model     : {args.model}")
    print(f"Pool size : {POOL_SIZE}  |  Locale: {args.locale}  |  Queries: {len(groups):,}")
    print("=" * 50)
    for name, val in results.items():
        print(f"  {name:<12}  {val:.4f}")
    print("=" * 50)

    if args.output:
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
