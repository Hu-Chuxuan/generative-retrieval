#!/usr/bin/env python3
"""
Case study: title-only vs full-info baselines.

Two cases per model:
  A) Title-only picked a WRONG product (FP) that full-info did NOT pick
     → shows what the full product info looks like for that FP
     → hypothesis: the title was misleading, richer info would have helped

  B) Full-info picked a WRONG product (FP) that title-only did NOT pick
     → shows what the full product info looks like for that FP
     → hypothesis: something in the extra fields confused the model

Usage:
    python case_study_full_vs_nonfull.py
    python case_study_full_vs_nonfull.py --top-k 5 --model qwen3-8b
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR    = Path(__file__).parent.parent / "shopping_queries_dataset"
RESULTS_DIR = Path(__file__).parent / "results"

PAIRS = {
    "qwen3-8b":   ("qwen3-8b_baseline_rp.csv",   "qwen3-8b_baseline_rp_full.csv"),
    "gemma4-e4b": ("gemma4-e4b_baseline_rp.csv", "gemma4-e4b_baseline_rp_full.csv"),
    "gemma4-31b": ("gemma4-31b_baseline_rp.csv", "gemma4-31b_baseline_rp_full.csv"),
    "qwen3-32b":  ("qwen3-32b_baseline_rp.csv",  "qwen3-32b_baseline_rp_full.csv"),
}

_MAX_FIELD = 300   # chars to show per field in product info


def load_products(locale: str = "us") -> pd.DataFrame:
    products = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_products.parquet")
    return (
        products[products["product_locale"] == locale]
        .fillna("")
        .set_index("product_id")
    )


def show_product(pid: str, products: pd.DataFrame, indent: int = 6) -> str:
    pad = " " * indent
    if pid not in products.index:
        return f"{pad}[product {pid} not found in dataset]"
    row = products.loc[pid]
    fields = [
        ("Title",   row.get("product_title", "")),
        ("Brand",   row.get("product_brand", "")),
        ("Color",   row.get("product_color", "")),
        ("Bullets", row.get("product_bullet_point", "")),
        ("Desc",    row.get("product_description", "")),
    ]
    parts = []
    for label, val in fields:
        val = str(val).strip()
        if val:
            short = val[:_MAX_FIELD] + ("…" if len(val) > _MAX_FIELD else "")
            parts.append(f"{pad}{label}: {short}")
    return "\n".join(parts) if parts else f"{pad}[no product fields]"


def analyze_pair(name: str, path_nf: Path, path_full: Path,
                 products: pd.DataFrame, top_k: int):

    if not path_nf.exists():
        print(f"[SKIP {name}] {path_nf.name} not found"); return
    if not path_full.exists():
        print(f"[SKIP {name}] {path_full.name} not found"); return

    df_nf   = pd.read_csv(path_nf)
    df_full = pd.read_csv(path_full)

    if df_nf.empty or df_full.empty:
        print(f"[SKIP {name}] empty CSV"); return

    # per-query F1 from each file
    pq_nf   = df_nf.drop_duplicates("query_id").set_index("query_id")
    pq_full = df_full.drop_duplicates("query_id").set_index("query_id")

    shared = pq_nf.index.intersection(pq_full.index)
    nf_f1   = pq_nf.loc[shared, "f1"]
    full_f1 = pq_full.loc[shared, "f1"]
    delta   = full_f1 - nf_f1

    print(f"\n{'═'*72}")
    print(f"  MODEL: {name}   ({len(shared)} shared queries)")
    print(f"  title-only mean F1: {nf_f1.mean():.4f}  |  "
          f"full-info mean F1: {full_f1.mean():.4f}  |  "
          f"mean Δ: {delta.mean():+.4f}")
    print(f"  Queries where full helps: {(delta > 0.001).sum()}  |  "
          f"hurts: {(delta < -0.001).sum()}  |  "
          f"tied: {(delta.abs() <= 0.001).sum()}")

    # FP sets per query
    nf_fps   = df_nf[df_nf["type"]   == "FP"].groupby("query_id")["product_id"].apply(set)
    full_fps = df_full[df_full["type"] == "FP"].groupby("query_id")["product_id"].apply(set)

    # ── CASE A: title-only FP that full did NOT pick ─────────────────────────
    print(f"\n{'─'*72}")
    print(f"  CASE A: Title-only selected a WRONG product that full-info did NOT pick")
    print(f"  (title was misleading → richer info would have avoided the mistake)")
    print(f"{'─'*72}")

    case_a = []   # (delta_f1, qid, pid)
    for qid in shared:
        fps_nf   = nf_fps.get(qid, set())
        fps_full = full_fps.get(qid, set())
        # products that were FP in title-only but not selected at all in full
        full_selected = df_full[df_full["query_id"] == qid]["product_id"]
        fp_nf_not_full = fps_nf - set(full_selected)
        for pid in fp_nf_not_full:
            case_a.append((delta.get(qid, 0), qid, pid))

    # sort: show queries where full improved most (interesting contrast)
    case_a.sort(key=lambda x: -x[0])

    shown = 0
    for df1, qid, pid in case_a:
        if shown >= top_k:
            break
        query = pq_nf.loc[qid, "query"]
        nf_val = nf_f1.get(qid, float("nan"))
        fu_val = full_f1.get(qid, float("nan"))
        print(f"\n  query_id={qid}  title-only F1={nf_val:.3f} → full-info F1={fu_val:.3f}  (Δ={df1:+.3f})")
        print(f"  Query: \"{query}\"")
        print(f"  FP product (in title-only, not in full-info): {pid}")
        print(show_product(pid, products))
        shown += 1

    if shown == 0:
        print("  [none found for this model]")

    # ── CASE B: full-info FP that title-only did NOT pick ────────────────────
    print(f"\n{'─'*72}")
    print(f"  CASE B: Full-info selected a WRONG product that title-only did NOT pick")
    print(f"  (extra fields confused the model → richer info introduced a mistake)")
    print(f"{'─'*72}")

    case_b = []   # (delta_f1, qid, pid)
    for qid in shared:
        fps_nf   = nf_fps.get(qid, set())
        fps_full = full_fps.get(qid, set())
        # products that were FP in full but not selected at all in title-only
        nf_selected = df_nf[df_nf["query_id"] == qid]["product_id"]
        fp_full_not_nf = fps_full - set(nf_selected)
        for pid in fp_full_not_nf:
            case_b.append((delta.get(qid, 0), qid, pid))

    # sort: show queries where full hurt most
    case_b.sort(key=lambda x: x[0])

    shown = 0
    for df1, qid, pid in case_b:
        if shown >= top_k:
            break
        query = pq_full.loc[qid, "query"]
        nf_val = nf_f1.get(qid, float("nan"))
        fu_val = full_f1.get(qid, float("nan"))
        print(f"\n  query_id={qid}  title-only F1={nf_val:.3f} → full-info F1={fu_val:.3f}  (Δ={df1:+.3f})")
        print(f"  Query: \"{query}\"")
        print(f"  FP product (in full-info, not in title-only): {pid}")
        print(show_product(pid, products))
        shown += 1

    if shown == 0:
        print("  [none found for this model]")


MODEL_COMPARISONS = [
    # (label_small, file_small, label_large, file_large)
    ("Gemma4-E4B (title-only)", "gemma4-e4b_baseline_rp.csv",
     "Gemma4-31B (title-only)", "gemma4-31b_baseline_rp.csv"),
    ("Gemma4-E4B (full-info)", "gemma4-e4b_baseline_rp_full.csv",
     "Gemma4-31B (full-info)", "gemma4-31b_baseline_rp_full.csv"),
]


def analyze_model_size(label_sm: str, path_sm: Path,
                        label_lg: str, path_lg: Path,
                        products: pd.DataFrame, top_k: int):
    """Show where the larger model beats the smaller model on shared queries."""

    if not path_sm.exists() or not path_lg.exists():
        print(f"[SKIP] missing files for model-size comparison"); return

    df_sm = pd.read_csv(path_sm)
    df_lg = pd.read_csv(path_lg)

    if df_sm.empty or df_lg.empty:
        print(f"[SKIP] empty CSV"); return

    pq_sm = df_sm.drop_duplicates("query_id").set_index("query_id")
    pq_lg = df_lg.drop_duplicates("query_id").set_index("query_id")
    shared = pq_sm.index.intersection(pq_lg.index)

    f1_sm = pq_sm.loc[shared, "f1"]
    f1_lg = pq_lg.loc[shared, "f1"]
    delta  = f1_lg - f1_sm

    print(f"\n{'═'*72}")
    print(f"  MODEL SIZE COMPARISON")
    print(f"  small: {label_sm}   ({len(f1_sm)} queries)  mean F1={f1_sm.mean():.4f}")
    print(f"  large: {label_lg}   ({len(f1_lg)} queries)  mean F1={f1_lg.mean():.4f}")
    print(f"  Shared queries: {len(shared)}  |  large better: {(delta>0.001).sum()}  "
          f"|  small better: {(delta<-0.001).sum()}  |  tied: {(delta.abs()<=0.001).sum()}")

    # ── CASE C: large got a TP that small missed ──────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  CASE C: Large model found a correct product (TP) that small model missed (FN)")
    print(f"{'─'*72}")

    sm_selected = df_sm.groupby("query_id")["product_id"].apply(set)
    lg_selected = df_lg.groupby("query_id")["product_id"].apply(set)
    sm_tps = df_sm[df_sm["type"] == "TP"].groupby("query_id")["product_id"].apply(set)
    lg_tps = df_lg[df_lg["type"] == "TP"].groupby("query_id")["product_id"].apply(set)
    sm_fns = df_sm[df_sm["type"] == "FN"].groupby("query_id")["product_id"].apply(set)

    case_c = []
    for qid in shared:
        fns_sm   = sm_fns.get(qid, set())
        tps_lg   = lg_tps.get(qid, set())
        # products the large model got right that small missed entirely
        recovered = fns_sm & tps_lg
        for pid in recovered:
            case_c.append((delta.get(qid, 0), qid, pid))

    case_c.sort(key=lambda x: -x[0])

    shown = 0
    for df1, qid, pid in case_c:
        if shown >= top_k: break
        query  = pq_sm.loc[qid, "query"]
        v_sm   = f1_sm.get(qid, float("nan"))
        v_lg   = f1_lg.get(qid, float("nan"))
        print(f"\n  query_id={qid}  small F1={v_sm:.3f} → large F1={v_lg:.3f}  (Δ={df1:+.3f})")
        print(f"  Query: \"{query}\"")
        print(f"  Product correctly found by large model (TP), missed by small (FN): {pid}")
        print(show_product(pid, products))
        shown += 1

    if shown == 0:
        print("  [none found]")

    # ── CASE D: small picked a FP that large avoided ─────────────────────────
    print(f"\n{'─'*72}")
    print(f"  CASE D: Small model selected a WRONG product (FP) that large model avoided")
    print(f"{'─'*72}")

    sm_fps = df_sm[df_sm["type"] == "FP"].groupby("query_id")["product_id"].apply(set)

    case_d = []
    for qid in shared:
        fps_sm = sm_fps.get(qid, set())
        lg_sel = lg_selected.get(qid, set())
        # FP in small but not even selected by large
        avoided = fps_sm - lg_sel
        for pid in avoided:
            case_d.append((delta.get(qid, 0), qid, pid))

    case_d.sort(key=lambda x: -x[0])

    shown = 0
    for df1, qid, pid in case_d:
        if shown >= top_k: break
        query  = pq_sm.loc[qid, "query"]
        v_sm   = f1_sm.get(qid, float("nan"))
        v_lg   = f1_lg.get(qid, float("nan"))
        print(f"\n  query_id={qid}  small F1={v_sm:.3f} → large F1={v_lg:.3f}  (Δ={df1:+.3f})")
        print(f"  Query: \"{query}\"")
        print(f"  FP selected by small model, avoided by large: {pid}")
        print(show_product(pid, products))
        shown += 1

    if shown == 0:
        print("  [none found]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k",  type=int, default=5)
    ap.add_argument("--model",  type=str, default=None,
                    help="restrict to one model key, e.g. qwen3-8b")
    ap.add_argument("--locale", default="us")
    ap.add_argument("--mode",   choices=["full-vs-nonfull", "model-size", "all"],
                    default="all")
    args = ap.parse_args()

    print("Loading product catalog…")
    products = load_products(args.locale)
    print(f"  {len(products):,} products loaded.")

    if args.mode in ("full-vs-nonfull", "all"):
        pairs = PAIRS
        if args.model:
            if args.model not in PAIRS:
                print(f"Unknown model '{args.model}'. Choose from: {list(PAIRS)}")
                return
            pairs = {args.model: PAIRS[args.model]}

        for name, (nf_file, full_file) in pairs.items():
            analyze_pair(
                name,
                RESULTS_DIR / nf_file,
                RESULTS_DIR / full_file,
                products,
                top_k=args.top_k,
            )

    if args.mode in ("model-size", "all"):
        for label_sm, file_sm, label_lg, file_lg in MODEL_COMPARISONS:
            analyze_model_size(
                label_sm, RESULTS_DIR / file_sm,
                label_lg, RESULTS_DIR / file_lg,
                products, top_k=args.top_k,
            )


if __name__ == "__main__":
    main()
