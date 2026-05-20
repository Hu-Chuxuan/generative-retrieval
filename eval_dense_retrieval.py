#!/usr/bin/env python3
"""
Dense retrieval baseline for ESCI.

Encodes all US products using a bi-encoder (same full-info text format as the
ablation script: title + brand + color + bullets + description), builds a FAISS
flat index, then retrieves the top-K products per query.

Timing is reported for three phases:
  1. Product encoding  (one-time; saved to disk and reused on reruns)
  2. FAISS index build
  3. Query encoding + retrieval

GPU note:
  Encoding 1.2M products on CPU takes ~1-2 hours for a large model.
  On a single H100 it takes ~3-5 minutes.  FAISS retrieval runs on CPU (fast).
  Use --device cpu only for quick testing with a small model.

Usage:
  python eval_dense_retrieval.py --output results/dense_bge_top100.csv
  python eval_dense_retrieval.py --model BAAI/bge-large-en-v1.5 --top-k 100
  python eval_dense_retrieval.py --device cpu --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import csv
import datetime
import hashlib
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR    = Path(__file__).parent.parent / "shopping_queries_dataset"
CACHE_DIR   = Path(__file__).parent / "embeddings_cache"
_MAX_CHARS  = 500   # per field, matching ablation script


# ── text formatting ───────────────────────────────────────────────────────────

def product_text(row: pd.Series) -> str:
    """Same fields as the full-info ablation prompt."""
    bp   = str(row.get("product_bullet_point", ""))[:_MAX_CHARS]
    desc = str(row.get("product_description",  ""))[:_MAX_CHARS]
    parts = [str(row.get("product_title", ""))]
    if row.get("product_brand"):  parts.append(f"Brand: {row['product_brand']}")
    if row.get("product_color"):  parts.append(f"Color: {row['product_color']}")
    if bp:                        parts.append(f"Bullets: {bp}")
    if desc:                      parts.append(f"Description: {desc}")
    return " | ".join(parts)


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

    all_products = (
        products[products["product_locale"] == locale]
        .fillna("")
        .reset_index(drop=True)
    )
    print(f"Queries   : {df['query_id'].nunique():,}")
    print(f"Catalogue : {len(all_products):,} products")
    return df, all_products


# ── encoding with cache ───────────────────────────────────────────────────────

def _cache_path(model_name: str, locale: str) -> Path:
    key = hashlib.md5(f"{model_name}_{locale}_{_MAX_CHARS}".encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"products_{key}.npy"


def encode_products(
    products: pd.DataFrame,
    model: SentenceTransformer,
    model_name: str,
    locale: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    cache = _cache_path(model_name, locale)
    if cache.exists():
        print(f"Loading cached embeddings from {cache.name} …")
        t0 = time.time()
        embs = np.load(cache)
        elapsed = time.time() - t0
        print(f"  Loaded {embs.shape} in {elapsed:.1f}s")
        return embs, 0.0   # 0 = encoding time (cached)

    print(f"Encoding {len(products):,} products (batch_size={batch_size}, device={device}) …")
    texts = [product_text(row) for _, row in tqdm(products.iterrows(),
             total=len(products), desc="  building texts")]

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine = inner product after L2-norm
        device=device,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    np.save(cache, embs)
    print(f"  Encoded in {elapsed:.1f}s  →  saved to {cache.name}")
    return embs.astype("float32"), elapsed


def encode_queries(
    queries: list[str],
    model: SentenceTransformer,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    embs = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    return embs.astype("float32"), elapsed


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_index(embs: np.ndarray) -> tuple[faiss.Index, float]:
    t0 = time.time()
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine (embeddings are L2-normalised)
    index.add(embs)
    elapsed = time.time() - t0
    print(f"FAISS index: {index.ntotal:,} vectors, dim={dim}, built in {elapsed:.2f}s")
    return index, elapsed


# ── retrieval + metrics ───────────────────────────────────────────────────────

def retrieve_and_evaluate(
    groups: list[pd.DataFrame],
    query_embs: np.ndarray,
    index: faiss.Index,
    products: pd.DataFrame,
    top_k: int,
    csv_w,
) -> tuple[list, list, list, float]:
    pid_to_idx = {pid: i for i, pid in enumerate(products["product_id"])}

    precisions, recalls, f1s = [], [], []
    t0 = time.time()

    for qi, grp in enumerate(tqdm(groups, desc="Retrieving")):
        q_emb = query_embs[qi : qi + 1]
        scores, indices = index.search(q_emb, top_k)
        scores   = scores[0].tolist()
        indices  = indices[0].tolist()

        retrieved_pids = [products.loc[idx, "product_id"] for idx in indices if idx >= 0]
        exact_ids      = set(grp.loc[grp["esci_label"] == "E", "product_id"])
        retrieved_set  = set(retrieved_pids)

        tp   = len(exact_ids & retrieved_set)
        prec = tp / len(retrieved_set) if retrieved_set else 0.0
        rec  = tp / len(exact_ids)     if exact_ids     else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        if csv_w:
            qid   = grp["query_id"].iloc[0]
            query = grp["query"].iloc[0]
            n_e   = len(exact_ids)
            base  = {"query_id": qid, "query": query, "n_exact_gt": n_e,
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4)}
            pid_to_label = dict(zip(grp["product_id"], grp["esci_label"]))
            for rank, (pid, sc) in enumerate(zip(retrieved_pids, scores), 1):
                csv_w.writerow({**base,
                    "product_id":    pid,
                    "product_title": products.loc[pid_to_idx[pid], "product_title"]
                                     if pid in pid_to_idx else "",
                    "esci_label":    pid_to_label.get(pid, "unmatched"),
                    "type":          "TP" if pid in exact_ids else "FP",
                    "retrieval_rank": rank,
                    "score":          round(sc, 5),
                })

    elapsed = time.time() - t0
    return precisions, recalls, f1s, elapsed


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="BAAI/bge-large-en-v1.5",
                   help="Sentence-transformers compatible bi-encoder")
    p.add_argument("--locale",      default="us", choices=["us", "es", "jp"])
    p.add_argument("--split",       default="test", choices=["train", "test"])
    p.add_argument("--num-queries", type=int, default=None)
    p.add_argument("--top-k",       type=int, default=100)
    p.add_argument("--batch-size",  type=int, default=512)
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--output",      default=None)
    p.add_argument("--no-cache",    action="store_true",
                   help="Re-encode products even if cache exists")
    return p.parse_args()


def main():
    args = parse_args()

    t_job_start = time.time()
    print("=" * 60)
    print(f"  eval_dense_retrieval.py  —  started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  model        : {args.model}")
    print(f"  locale       : {args.locale}  |  split: {args.split}")
    print(f"  num_queries  : {args.num_queries or 'all'}")
    print(f"  top_k        : {args.top_k}  |  batch_size: {args.batch_size}  |  device: {args.device}")
    print(f"  output       : {args.output or '(none)'}")
    print("=" * 60)

    if args.no_cache:
        c = _cache_path(args.model, args.locale)
        if c.exists():
            c.unlink()
            print(f"Deleted cache {c.name}")

    df, all_products = load_data(args.locale, args.split, args.num_queries)
    groups = [grp.reset_index(drop=True)
              for _, grp in df.groupby("query_id", sort=False)]
    queries = [grp["query"].iloc[0] for grp in groups]

    print(f"\nLoading model: {args.model}  (device={args.device})")
    model = SentenceTransformer(args.model, device=args.device)
    print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

    print(f"\n── Phase 1: Product encoding  [{datetime.datetime.now():%H:%M:%S}] ─────────────────")
    prod_embs, t_encode = encode_products(
        all_products, model, args.model, args.locale, args.batch_size, args.device)

    print(f"\n── Phase 2: FAISS index build  [{datetime.datetime.now():%H:%M:%S}] ────────────────")
    index, t_index = build_index(prod_embs)

    print(f"\n── Phase 3: Query encoding + retrieval (top-{args.top_k})  [{datetime.datetime.now():%H:%M:%S}] ─")
    q_embs, t_query_enc = encode_queries(queries, model, args.batch_size, args.device)
    print(f"Query encoding: {t_query_enc:.2f}s  ({t_query_enc/len(queries)*1000:.1f} ms/query)")

    _CSV_FIELDS = ["query_id", "query", "n_exact_gt", "product_id", "product_title",
                   "esci_label", "type", "retrieval_rank", "score",
                   "precision", "recall", "f1"]
    csv_fh = open(args.output, "w", newline="") if args.output else None
    csv_w  = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS) if csv_fh else None
    if csv_w:
        csv_w.writeheader()

    precisions, recalls, f1s, t_retrieval = retrieve_and_evaluate(
        groups, q_embs, index, all_products, args.top_k, csv_w)

    if csv_fh:
        csv_fh.close()

    # ── timing summary ────────────────────────────────────────────────────────
    t_per_q = t_retrieval / len(groups) * 1000
    print("\n" + "═" * 52)
    print(f"  TIMING SUMMARY")
    print("═" * 52)
    if t_encode > 0:
        print(f"  Product encoding  : {t_encode:.1f}s  ({len(all_products)/t_encode:.0f} products/s)")
    else:
        print(f"  Product encoding  : (loaded from cache)")
    print(f"  FAISS index build : {t_index:.2f}s")
    print(f"  Query enc+retrieve: {t_retrieval:.2f}s total  |  {t_per_q:.1f} ms/query")
    print("═" * 52)
    print(f"  RETRIEVAL METRICS  (top-{args.top_k})")
    print("═" * 52)
    print(f"  Model     : {args.model}")
    print(f"  Queries   : {len(groups):,}  |  Catalogue: {len(all_products):,}")
    print(f"  Precision : {np.mean(precisions):.4f}")
    print(f"  Recall    : {np.mean(recalls):.4f}")
    print(f"  F1        : {np.mean(f1s):.4f}")
    print("═" * 52)
    if args.output:
        print(f"  Saved to  : {args.output}")
    print(f"\n  Total job time: {time.time() - t_job_start:.1f}s")
    print(f"  Finished at  : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
