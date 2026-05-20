#!/usr/bin/env python3
"""
End-to-end pipeline: dense retrieval → LLM selection.

For each query:
  1. (One-time) encode all catalogue products with a bi-encoder  → FAISS index
  2. Encode query → FAISS top-100 retrieval
  3. Run LLM on the top-100 products with BOTH prompt variants

Timing reported per query:
  T_total   : from database-encoding start to first token (first run includes encoding)
  T_pipeline: from query-encoding start to first token (after DB is cached)

GPU note:
  Product encoding needs GPU (1.2M products takes ~3-5 min on H100, 1-2 h on CPU).
  FAISS retrieval and LLM run on GPU automatically.

Usage:
  python eval_e2e_pipeline.py --num-queries 10 --output results/e2e_pipeline.csv
"""

import argparse
import csv
import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    import faiss
except ImportError:
    raise SystemExit("faiss not found — install faiss-gpu or faiss-cpu first")

DATA_DIR    = Path(__file__).parent.parent / "shopping_queries_dataset"
CACHE_DIR   = Path(__file__).parent / "embeddings_cache"
_MAX_CHARS  = 500

_FIELDS = ["product_id", "product_title", "product_brand", "product_color",
           "product_bullet_point", "product_description"]

# ── prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {
    "advanced": {
        "system": (
            "You are an Amazon product search expert. A product is relevant ONLY if it "
            "satisfies EVERY requirement in the query. Treat all attributes as hard filters: "
            "wrong color, wrong product type, or violating a 'without X' / 'no X' constraint "
            "makes a product irrelevant regardless of other similarities."
        ),
        "user": """\
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
with no explanation.""",
    },
    "baseline": {
        "system": "You are a product search expert. Follow instructions exactly.",
        "user": """\
Given the customer search query below, identify the {n_select} most relevant products \
from the list of {pool_size} products.

Query: {query}

Products:
{product_list}

Output ONLY a comma-separated list of exactly {n_select} product numbers, \
with no explanation.""",
    },
}

# ── text helpers ──────────────────────────────────────────────────────────────

def product_text_embed(row: pd.Series) -> str:
    """Text sent to the bi-encoder (full info, same as ablation)."""
    bp   = str(row.get("product_bullet_point", ""))[:_MAX_CHARS]
    desc = str(row.get("product_description",  ""))[:_MAX_CHARS]
    parts = [str(row.get("product_title", ""))]
    if row.get("product_brand"): parts.append(f"Brand: {row['product_brand']}")
    if row.get("product_color"): parts.append(f"Color: {row['product_color']}")
    if bp:                       parts.append(f"Bullets: {bp}")
    if desc:                     parts.append(f"Description: {desc}")
    return " | ".join(parts)


def product_text_llm(i: int, row: pd.Series, field_chars: int) -> str:
    """Text shown to the LLM (numbered, truncated fields)."""
    bp   = str(row.get("product_bullet_point", ""))[:field_chars]
    desc = str(row.get("product_description",  ""))[:field_chars]
    return "\n".join([
        f"{i}. {row['product_title']}",
        f"   Brand: {row.get('product_brand', '')}",
        f"   Color: {row.get('product_color', '')}",
        f"   Bullet points: {bp}",
        f"   Description: {desc}",
    ])

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
        qids = df["query_id"].unique()[:num_queries]
        df = df[df["query_id"].isin(qids)]
    all_products = (products[products["product_locale"] == locale]
                    .fillna("").reset_index(drop=True))
    print(f"Queries   : {df['query_id'].nunique():,}  |  Catalogue: {len(all_products):,}")
    return df, all_products

# ── product encoding (cached) ─────────────────────────────────────────────────

def _cache_path(model_name: str, locale: str) -> Path:
    key = hashlib.md5(f"{model_name}_{locale}_{_MAX_CHARS}".encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"products_{key}.npy"


def get_product_embeddings(products, model, model_name, locale,
                           batch_size, device) -> tuple[np.ndarray, float]:
    cache = _cache_path(model_name, locale)
    if cache.exists():
        print(f"Loading cached product embeddings ({cache.name}) …")
        t0 = time.time()
        embs = np.load(cache).astype("float32")
        t_load = time.time() - t0
        print(f"  {embs.shape}  loaded in {t_load:.1f}s")
        return embs, 0.0   # 0 = encoding was skipped

    print(f"Encoding {len(products):,} products …")
    texts = [product_text_embed(row) for _, row in products.iterrows()]
    t0 = time.time()
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True, device=device,
                        convert_to_numpy=True).astype("float32")
    t_enc = time.time() - t0
    np.save(cache, embs)
    print(f"  Encoded in {t_enc:.1f}s ({len(products)/t_enc:.0f} prod/s)  →  {cache.name}")
    return embs, t_enc

# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_faiss_index(embs: np.ndarray) -> tuple[object, float]:
    t0 = time.time()
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    t = time.time() - t0
    print(f"FAISS index: {idx.ntotal:,} vectors, dim={embs.shape[1]}, built in {t:.2f}s")
    return idx, t

# ── LLM message building ──────────────────────────────────────────────────────

MAX_MODEL_LEN = 32768

def build_llm_message(query: str, n_select: int,
                       retrieved_rows: pd.DataFrame,
                       prompt_key: str) -> list[dict]:
    tmpl  = PROMPTS[prompt_key]
    char_budget = int((MAX_MODEL_LEN - 512) * 3.0)
    chars_used  = len(tmpl["system"]) + 400
    n = len(retrieved_rows)
    lines = []
    for i, (_, row) in enumerate(retrieved_rows.iterrows(), 1):
        remaining  = n - (i - 1)
        per_prod   = max(100, (char_budget - chars_used) // remaining)
        field_chars = max(30, min(_MAX_CHARS, (per_prod - 150) // 2))
        entry = product_text_llm(i, row, field_chars)
        lines.append(entry)
        chars_used += len(entry) + 2
    user = tmpl["user"].format(
        query=query, n_select=n_select,
        pool_size=len(lines), product_list="\n\n".join(lines),
    )
    return [{"role": "system", "content": tmpl["system"]},
            {"role": "user",   "content": user}]


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


def get_ttft_ms(out) -> float | None:
    try:
        m = out.metrics
        if m.first_token_time is not None and m.first_scheduled_time is not None:
            return (m.first_token_time - m.first_scheduled_time) * 1000
    except Exception:
        pass
    return None

# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model",   default="BAAI/bge-large-en-v1.5")
    p.add_argument("--llm-model",     default="google/gemma-4-E4B-it")
    p.add_argument("--locale",        default="us", choices=["us", "es", "jp"])
    p.add_argument("--split",         default="test")
    p.add_argument("--num-queries",   type=int, default=10)
    p.add_argument("--top-k",           type=int, default=100)
    p.add_argument("--batch-size",      type=int, default=512)
    p.add_argument("--device",          default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--prompt-variant",  default="both",
                   choices=["both", "advanced", "baseline"],
                   help="Which LLM prompt variant(s) to run")
    p.add_argument("--output",          default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    df, all_products = load_data(args.locale, args.split, args.num_queries)
    groups  = [grp.reset_index(drop=True)
               for _, grp in df.groupby("query_id", sort=False)]
    queries = [grp["query"].iloc[0] for grp in groups]

    # ── bi-encoder ────────────────────────────────────────────────────────────
    print(f"\nLoading bi-encoder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model, device=args.device)

    # ── PHASE 1: product encoding (wall clock starts here) ───────────────────
    T_wall_start = time.time()
    prod_embs, t_encode = get_product_embeddings(
        all_products, embedder, args.embed_model, args.locale,
        args.batch_size, args.device)

    # ── PHASE 2: FAISS index ─────────────────────────────────────────────────
    faiss_index, t_index = build_faiss_index(prod_embs)
    pid_list = all_products["product_id"].tolist()

    # ── PHASE 3: query encoding ───────────────────────────────────────────────
    print(f"\nEncoding {len(queries)} queries …")
    T_query_start = time.time()
    q_embs = embedder.encode(queries, batch_size=64, normalize_embeddings=True,
                              device=args.device, convert_to_numpy=True,
                              show_progress_bar=True).astype("float32")
    t_q_enc = time.time() - T_query_start

    # ── PHASE 4: FAISS retrieval ──────────────────────────────────────────────
    print(f"FAISS retrieval (top-{args.top_k}) …")
    t0 = time.time()
    scores_all, idx_all = faiss_index.search(q_embs, args.top_k)
    t_faiss = time.time() - t0
    print(f"  {t_faiss*1000:.1f} ms total  ({t_faiss/len(queries)*1000:.1f} ms/query)")

    # build retrieved product dataframes per query + retrieval recall@K
    pid_to_row = {row["product_id"]: row
                  for _, row in all_products[_FIELDS].iterrows()}
    retrieved_pools = []
    retrieval_recalls = []
    for qi in range(len(groups)):
        rows = []
        retrieved_pids = []
        for idx in idx_all[qi]:
            if idx >= 0:
                pid = pid_list[idx]
                r   = pid_to_row.get(pid)
                if r is not None:
                    rows.append(r)
                    retrieved_pids.append(pid)
        retrieved_pools.append(pd.DataFrame(rows).reset_index(drop=True))
        exact_ids = set(groups[qi].loc[groups[qi]["esci_label"] == "E", "product_id"])
        rec_at_k = (len(exact_ids & set(retrieved_pids)) / len(exact_ids)
                    if exact_ids else 0.0)
        retrieval_recalls.append(rec_at_k)

    # ── PHASE 5: load LLM ────────────────────────────────────────────────────
    print(f"\nLoading LLM: {args.llm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm = LLM(model=args.llm_model, tensor_parallel_size=args.tensor_parallel,
               max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90,
               disable_log_stats=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # ── PHASE 6: LLM inference ────────────────────────────────────────────────
    _CSV_FIELDS = ["query_id", "query", "prompt_variant", "n_exact_gt",
                   "retrieval_recall_at_k",
                   "product_id", "product_title", "esci_label", "type",
                   "pred_rank", "retrieval_rank", "precision", "recall", "f1",
                   "t_encode_db_s", "t_query_enc_ms", "t_faiss_ms", "t_ttft_ms",
                   "t_total_first_run_s"]
    csv_fh = open(args.output, "w", newline="") if args.output else None
    csv_w  = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS) if csv_fh else None
    if csv_w:
        csv_w.writeheader()

    active_prompts = (list(PROMPTS.keys()) if args.prompt_variant == "both"
                      else [args.prompt_variant])
    results_by_variant = {k: {"prec": [], "rec": [], "f1": [], "ttft": []}
                          for k in active_prompts}
    first_token_wall = None  # wall time to first LLM token (first query)

    for variant_name in active_prompts:
        print(f"\n── LLM variant: {variant_name} ──────────────────────────────")
        msgs_all = []
        for qi, (grp, pool) in enumerate(zip(groups, retrieved_pools)):
            n_select = int((grp["esci_label"] == "E").sum()) or 1
            msgs_all.append(build_llm_message(
                queries[qi], n_select, pool, variant_name))

        for qi in tqdm(range(len(msgs_all)), desc=f"  {variant_name}"):
            grp   = groups[qi]
            pool  = retrieved_pools[qi]
            t_q_start = time.time()

            out = llm.chat([msgs_all[qi]], sampling_params=sampling_params)[0]
            ttft = get_ttft_ms(out) or (time.time() - t_q_start) * 1000

            if first_token_wall is None:
                first_token_wall = time.time() - T_wall_start

            text      = out.outputs[0].text
            n_select  = int((grp["esci_label"] == "E").sum()) or 1
            selected  = parse_selection(text, len(pool), n_select)

            exact_ids    = set(grp.loc[grp["esci_label"] == "E", "product_id"])
            selected_pids = [pool.loc[idx, "product_id"] for idx in selected
                             if idx < len(pool)]
            selected_set  = set(selected_pids)
            tp = len(exact_ids & selected_set)

            prec = tp / len(selected_set) if selected_set else 0.0
            rec  = tp / len(exact_ids)    if exact_ids    else 0.0
            f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0

            results_by_variant[variant_name]["prec"].append(prec)
            results_by_variant[variant_name]["rec"].append(rec)
            results_by_variant[variant_name]["f1"].append(f1)
            results_by_variant[variant_name]["ttft"].append(ttft)

            if csv_w:
                qid   = grp["query_id"].iloc[0]
                query = queries[qi]
                # retrieval rank lookup
                ret_rank = {pid_list[idx]: r+1
                            for r, idx in enumerate(idx_all[qi]) if idx >= 0}
                base = {"query_id": qid, "query": query,
                        "prompt_variant": variant_name,
                        "n_exact_gt": len(exact_ids),
                        "retrieval_recall_at_k": round(retrieval_recalls[qi], 4),
                        "precision": round(prec,4), "recall": round(rec,4),
                        "f1": round(f1,4),
                        "t_encode_db_s":     round(t_encode, 1),
                        "t_query_enc_ms":    round(t_q_enc/len(queries)*1000, 2),
                        "t_faiss_ms":        round(t_faiss/len(queries)*1000, 2),
                        "t_ttft_ms":         round(ttft, 2),
                        "t_total_first_run_s": round(first_token_wall, 1)
                                               if first_token_wall else ""}
                pid_to_label = dict(zip(grp["product_id"], grp["esci_label"]))
                pid_to_title = {r["product_id"]: r["product_title"]
                                for _, r in grp.iterrows()}
                for rank_pos, pid in enumerate(selected_pids, 1):
                    row_data = pool[pool["product_id"] == pid]
                    csv_w.writerow({**base,
                        "product_id":    pid,
                        "product_title": row_data["product_title"].iloc[0]
                                         if not row_data.empty else "",
                        "esci_label":    pid_to_label.get(pid, "unmatched"),
                        "type":          "TP" if pid in exact_ids else "FP",
                        "pred_rank":     rank_pos,
                        "retrieval_rank": ret_rank.get(pid, ""),
                    })
                for pid in exact_ids - selected_set:
                    csv_w.writerow({**base,
                        "product_id":    pid,
                        "product_title": pid_to_title.get(pid, ""),
                        "esci_label":    "E",
                        "type":          "FN",
                        "pred_rank":     "",
                        "retrieval_rank": ret_rank.get(pid, "not_retrieved"),
                    })
        if csv_fh:
            csv_fh.flush()

    if csv_fh:
        csv_fh.close()

    # ── timing + metrics summary ──────────────────────────────────────────────
    t_q_enc_ms  = t_q_enc / len(queries) * 1000
    t_faiss_ms  = t_faiss / len(queries) * 1000

    print("\n" + "═"*60)
    print("  TIMING BREAKDOWN")
    print("═"*60)
    if t_encode > 0:
        print(f"  DB encoding (1-time)  : {t_encode:.1f}s  "
              f"({len(all_products)/t_encode:.0f} prod/s)")
    else:
        print(f"  DB encoding           : loaded from cache")
    print(f"  FAISS index build     : {t_index:.2f}s")
    print(f"  Query encoding        : {t_q_enc_ms:.1f} ms/query")
    print(f"  FAISS retrieval       : {t_faiss_ms:.2f} ms/query")
    if first_token_wall:
        print(f"  ──────────────────────────────────────────────")
        print(f"  Total (DB→first token): {first_token_wall:.1f}s  (first query, incl. DB encoding)")
    print("═"*60)
    print(f"  RETRIEVAL  recall@{args.top_k}  : {np.mean(retrieval_recalls):.4f}  "
          f"(fraction of E-class products in top-{args.top_k})")
    print("═"*60)
    print("  LLM METRICS  (top-100 retrieved pool)")
    print("═"*60)
    for vname, res in results_by_variant.items():
        print(f"  [{vname}]")
        print(f"    Precision : {np.mean(res['prec']):.4f}")
        print(f"    Recall    : {np.mean(res['rec']):.4f}")
        print(f"    F1        : {np.mean(res['f1']):.4f}")
        print(f"    TTFT      : {np.mean(res['ttft']):.1f} ms")
    print("═"*60)
    if args.output:
        print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
