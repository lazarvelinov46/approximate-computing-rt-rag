#!/usr/bin/env python3
# scripts/plot_cross_model.py — NEW FILE
"""Knobs 1 and 2 across generators: per-model report figures, comparison
figures and tables. Regime 2 (A1 pooled corpus), short mode, n=1000.

Per model  results/figures/knob{1,2}_curves_<tag>.png — the layout of the
           report figures from notebooks 07 / 08, driven by that model's CSVs.
Across     results/figures/knob{1,2}_cross_model.png
           (a) EM vs cost, (b) loss vs evidence delivered with each model's
           evidence-loss projection, (c) dEM(model) - dEM(reference), 95% CI.
Tables     results/cross_model/knob12_tables.md + one CSV per table.

Retrieval-side quantities (dense sweeps, distance computations, ANN recall,
compression, memory, complete_frac) do not depend on the generator: every
model's knob-1/2 runs used the reference model's stored retrieval (replayed,
notebook 35). They are read once from the reference summaries, and each
model's complete_frac is asserted equal to them. Only generation-side
quantities come from each model's result CSVs.

Bootstrap: paired over questions, one index matrix (B=10,000, seed 42)
shared by every setting and model, so all four hit vectors of a comparison
are resampled jointly.

Pure CPU. From the repo root:
    python scripts/plot_cross_model.py --self-test
    python scripts/plot_cross_model.py
    python scripts/plot_cross_model.py --models qwen25_3b llama32_3b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.config import MODEL_TAGS, REFERENCE_GENERATOR, results_dir   # noqa: E402
import src.metrics.quality as Q                                       # noqa: E402
import src.pipeline.rag as R                                          # noqa: E402
from plot_phase2 import verdict, SIGNIFICANT                          # noqa: E402

TAG2GEN = {t: g for g, t in MODEL_TAGS.items()}
REF = MODEL_TAGS[REFERENCE_GENERATOR]
DISPLAY = {"qwen25_3b": "Qwen2.5-3B-Instruct", "llama32_3b": "Llama-3.2-3B-Instruct",
           "qwen25_7b": "Qwen2.5-7B-Instruct"}
SHORT = {"qwen25_3b": "Qwen-3B", "llama32_3b": "Llama-3B", "qwen25_7b": "Qwen-7B"}
COLOUR = {"qwen25_3b": "tab:blue", "llama32_3b": "tab:red", "qwen25_7b": "tab:green"}

N, TOP_K, N_BOOT, SEED = 1000, 5, 10_000, 42
FIG_DIR = REPO / "results" / "figures"
TAB_DIR = REPO / "results" / "cross_model"
BASE_FILE, BASE_TAG = "regime2_a1_baseline_results.csv", "regime2_a1_baseline"

KNOBS = {
    1: dict(file="knob1_results.csv", summary="knob1_summary.csv",
            retrieval=("index", "ndis", "ann_recall"), cost="ndis",
            families=(("hnsw", "HNSW", "o"), ("ivf", "IVF", "s"))),
    2: dict(file="knob2_results.csv", summary="knob2_summary.csv",
            retrieval=("index", "compression", "corpus_mib", "ann_recall"), cost="corpus_mib",
            families=(("pq", "PQ", "o"), ("sq", "SQ", "s"))),
}
XLABEL = {1: "distance computations / query", 2: "corpus vector memory (MiB)"}
KNOB_NAME = {1: "search effort", 2: "embedding precision"}

# (b, c) published in notebooks 07 / 08 — the self-test must reproduce them.
QWEN_PUBLISHED = {"knob1_hnsw_ef32": (2, 11), "knob1_hnsw_ef5": (25, 56),
                  "knob1_ivf_nprobe1": (39, 139), "knob2_pq_m64": (47, 47),
                  "knob2_pq_m32": (42, 73)}


# --- data -------------------------------------------------------------------

def short(setting: str) -> str:
    return (setting.replace("knob1_hnsw_", "HNSW ").replace("knob1_ivf_", "IVF ")
                   .replace("knob2_pq_", "PQ ").replace("knob2_sq_", "SQ "))


def summary(knob: int) -> pd.DataFrame:
    """Reference model's summary: the model-independent retrieval columns."""
    return pd.read_csv(REPO / "results" / KNOBS[knob]["summary"])


def exact(knob: int) -> pd.Series:
    s = summary(knob)
    row = s[s["index"] == "flat"]
    assert len(row) == 1, f"{KNOBS[knob]['summary']}: expected exactly one flat row"
    return row.iloc[0]


def _load(tag: str, fname: str, setting: str) -> pd.DataFrame:
    df = R.load_results(str(results_dir(TAG2GEN[tag]) / fname), setting=setting)
    assert len(df) == N and df.qid.nunique() == N, \
        f"{tag}/{fname}:{setting}: {len(df)} rows, {df.qid.nunique()} unique qids"
    return df.sort_values("qid").reset_index(drop=True)


def _hits(df: pd.DataFrame) -> np.ndarray:
    return np.array([Q.exact_match(p, g) for p, g in zip(df.prediction, df.gold_answer)],
                    dtype=float)


def _mcnemar(h_ref: np.ndarray, h_set: np.ndarray) -> dict:
    b = int(((h_set == 1) & (h_ref == 0)).sum())     # setting-only correct
    c = int(((h_ref == 1) & (h_set == 0)).sum())     # exact-only correct
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"b": b, "c": c, "d": b + c, "p": round(float(p), 4), "verdict": verdict(b, c, p)}


def load_model(tag: str) -> dict:
    """Everything the figures and tables need for one model, knobs 1 and 2."""
    base = _load(tag, BASE_FILE, BASE_TAG)
    agg0 = Q.aggregate(base.to_dict("records"), k=TOP_K)
    m = {"tag": tag, "qids": base.qid.tolist(), "h0": _hits(base), "agg0": agg0,
         "gap": agg0["em_complete"] - agg0["em_incomplete"], "hits": {}, "tables": {}}
    for knob, spec in KNOBS.items():
        summ = summary(knob)
        path = results_dir(TAG2GEN[tag]) / spec["file"]
        rows = []
        for s in sorted(R.load_results(str(path)).setting.unique()):
            df = _load(tag, spec["file"], s)
            assert df.qid.tolist() == m["qids"], f"{tag}:{s}: question set differs from baseline"
            ref = summ[summ["setting"] == s]
            assert len(ref) == 1, f"{s}: not in {spec['summary']}"
            ref = ref.iloc[0]
            a, h = Q.aggregate(df.to_dict("records"), k=TOP_K), _hits(df)
            assert abs(a["complete_frac"] - float(ref["complete_frac"])) < 1e-4, \
                (f"{tag}:{s}: complete_frac {a['complete_frac']} vs reference "
                 f"{ref['complete_frac']} — retrieval differs from the reference run")
            m["hits"][s] = h
            rows.append({"setting": s, **{c: ref[c] for c in spec["retrieval"]},
                         "recall@5": a[f"recall@{TOP_K}"], "complete_frac": a["complete_frac"],
                         "em": a["em"], "f1": a["f1"], "em_complete": a["em_complete"],
                         "em_incomplete": a["em_incomplete"], "abstain_rate": a["abstain_rate"],
                         "dEM": round(a["em"] - agg0["em"], 4), **_mcnemar(m["h0"], h)})
        m["tables"][knob] = (pd.DataFrame(rows).sort_values(spec["cost"])
                             .reset_index(drop=True))
    return m


def boundary_mib(tab: pd.DataFrame):
    """Memory of the last setting (increasing compression) before the first
    significant loss. Reproduces notebook 08's red line at m48 (3.0 MiB)."""
    t = tab.sort_values("compression").reset_index(drop=True)
    for i, r in t.iterrows():
        if r["verdict"] == SIGNIFICANT and r["c"] > r["b"]:
            # Midpoint in log space between the last passing setting and the
            # first loss: the boundary lies between them, not on either point.
            if i == 0:
                return None
            lo, hi = float(t.loc[i, "corpus_mib"]), float(t.loc[i - 1, "corpus_mib"])
            return float(np.sqrt(lo * hi))
    return None


# --- bootstrap --------------------------------------------------------------

_IDX = None

def boot_idx() -> np.ndarray:
    global _IDX
    if _IDX is None:
        _IDX = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))
    return _IDX


def delta_boot(h_set: np.ndarray, h_ref: np.ndarray):
    """-> (dEM replicates, EM-ratio replicates) for one model and setting."""
    idx = boot_idx()
    es, er = h_set[idx].mean(axis=1), h_ref[idx].mean(axis=1)
    return es - er, es / er


def _ci(x: np.ndarray):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def cross_table(knob: int, models: dict) -> pd.DataFrame:
    ref = models[REF]
    rt = ref["tables"][knob].set_index("setting")
    shared = [s for s in ref["tables"][knob]["setting"]
              if all(s in m["hits"] for m in models.values())]
    rows = []
    for s in shared:
        dR, rR = delta_boot(ref["hits"][s], ref["h0"])
        row = {"setting": s, "complete_frac": float(rt.loc[s, "complete_frac"])}
        for tag, m in models.items():
            t = m["tables"][knob].set_index("setting").loc[s]
            d, r = delta_boot(m["hits"][s], m["h0"])
            row[f"{tag}_em"] = float(t["em"])
            row[f"{tag}_dEM"] = float(t["dEM"])
            row[f"{tag}_dEM_lo"], row[f"{tag}_dEM_hi"] = _ci(d)
            row[f"{tag}_verdict"] = t["verdict"]
            if tag != REF:
                row[f"{tag}_ddEM"] = round(float(t["dEM"]) - float(rt.loc[s, "dEM"]), 4)
                row[f"{tag}_ddEM_lo"], row[f"{tag}_ddEM_hi"] = _ci(d - dR)
                row[f"{tag}_ddrel"] = round(float(t["em"]) / m["agg0"]["em"]
                                            - float(rt.loc[s, "em"]) / ref["agg0"]["em"], 4)
                row[f"{tag}_ddrel_lo"], row[f"{tag}_ddrel_hi"] = _ci(r - rR)
        rows.append(row)
    return pd.DataFrame(rows)


def projection_table(models: dict, cross: pd.DataFrame) -> pd.DataFrame:
    """Observed dEM vs dcf x (em_complete - em_incomplete) at each model's exact."""
    cf0 = models[REF]["agg0"]["complete_frac"]
    for tag, m in models.items():
        assert abs(m["agg0"]["complete_frac"] - cf0) < 1e-4, f"{tag}: baseline retrieval differs"
    rows = []
    for _, c in cross.iterrows():
        dcf = c["complete_frac"] - cf0
        row = {"setting": c["setting"], "d_complete_frac": round(dcf, 4)}
        for tag, m in models.items():
            proj = dcf * m["gap"]
            row[f"{tag}_proj"] = round(proj, 4)
            row[f"{tag}_obs"] = c[f"{tag}_dEM"]
            row[f"{tag}_resid"] = round(c[f"{tag}_dEM"] - proj, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# --- per-model report figures (layout of notebooks 07 / 08) -----------------

def fig_knob1_curves(m: dict, out_path: Path) -> None:
    tab, e = m["tables"][1], exact(1)
    dense = pd.read_csv(REPO / "results" / "knob1_dense_sweep.csv")
    dh = dense[dense["index"] == "hnsw"].sort_values("ndis_per_query")
    di = dense[dense["index"] == "ivf"].sort_values("ndis_per_query")
    hn = tab[tab["index"] == "hnsw"].sort_values("ndis")
    iv = tab[tab["index"] == "ivf"].sort_values("ndis")
    em0 = m["agg0"]["em"]
    se0 = np.sqrt(em0 * (1 - em0) / N)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].semilogx(dh["ndis_per_query"], dh["ann_recall"], "o-", ms=3, label="HNSW (ef)")
    ax[0].semilogx(di["ndis_per_query"], di["ann_recall"], "s-", ms=3, label="IVF (nprobe)")
    ax[0].axvline(float(e["ndis"]), ls=":", c="k", lw=1)
    ax[0].set_xlabel("distance computations / query")
    ax[0].set_ylabel("ANN recall vs exact")
    ax[0].set_title("(a) retrieval fidelity, dense sweep")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(dh["ann_recall"], dh["recall@5"], "o-", ms=3, c="C0", label="recall@5 (HNSW)")
    ax[1].plot(di["ann_recall"], di["recall@5"], "s-", ms=3, c="C1", label="recall@5 (IVF)")
    ax[1].plot(hn["ann_recall"], hn["em"], "o--", ms=6, c="C0", label="EM (HNSW)")
    ax[1].plot(iv["ann_recall"], iv["em"], "s--", ms=6, c="C1", label="EM (IVF)")
    ax[1].scatter([1.0], [em0], marker="*", s=180, c="k", zorder=5, label="exact")
    ax[1].set_xlabel("ANN recall vs exact")
    ax[1].set_ylabel("recall@5  /  EM")
    ax[1].set_title("(b) propagation to answer quality")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].semilogx(hn["ndis"], hn["em"], "o-", ms=6, label="HNSW")
    ax[2].semilogx(iv["ndis"], iv["em"], "s-", ms=6, label="IVF")
    ax[2].axhline(em0, ls="--", c="k", lw=1, label=f"exact ({em0:.3f})")
    ax[2].axhspan(em0 - se0, em0 + se0, color="k", alpha=.08)
    ax[2].scatter([float(e["ndis"])], [em0], marker="*", s=180, c="k", zorder=5)
    ax[2].set_xlabel("distance computations / query")
    ax[2].set_ylabel("EM")
    ax[2].set_title("(c) quality vs work  (band = ±1 SE)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    n_gpu = len(hn) + len(iv)
    fig.suptitle("Knob 1 — search effort, A1 pooled corpus (66,581 passages), short mode"
                 f" · {DISPLAY[m['tag']]}\n(a) retrieval-only sweep, generator-independent"
                 f" · (b, c) the {n_gpu} settings generated with this model", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_knob2_curves(m: dict, out_path: Path) -> None:
    tab, e = m["tables"][2], exact(2)
    d2 = pd.read_csv(REPO / "results" / "knob2_dense_sweep.csv")
    d2["nbits"] = d2["nbits"].astype(str)
    dpq = d2[(d2["index"] == "pq") & (d2["nbits"] == "8")].sort_values("compression")
    dsq = d2[d2["index"] == "sq"].sort_values("compression")
    pq_ = tab[tab["index"] == "pq"].sort_values("compression")
    sq_ = tab[tab["index"] == "sq"].sort_values("compression")
    em0 = m["agg0"]["em"]
    se0 = np.sqrt(em0 * (1 - em0) / N)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].semilogx(dpq["compression"], dpq["ann_recall"], "o-", ms=3, label="PQ ANN recall")
    ax[0].semilogx(dsq["compression"], dsq["ann_recall"], "s-", ms=3, label="SQ ANN recall")
    ax[0].semilogx(dpq["compression"], dpq["exact_match_topk"], "o--", ms=3, alpha=.6,
                   label="PQ list identity")
    ax[0].set_xlabel("compression vs fp32")
    ax[0].set_ylabel("agreement with exact search")
    ax[0].set_title("(a) fidelity: recall vs ordering")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].semilogx(dpq["compression"], dpq["complete_frac"], "o-", ms=3, label="PQ")
    ax[1].semilogx(dsq["compression"], dsq["complete_frac"], "s-", ms=3, label="SQ")
    ax[1].axhline(float(e["complete_frac"]), ls="--", c="k", lw=1, label="exact")
    ax[1].set_xlabel("compression vs fp32")
    ax[1].set_ylabel("complete_frac")
    ax[1].set_title("(b) evidence delivered")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].semilogx(pq_["corpus_mib"], pq_["em"], "o-", ms=6, label="PQ")
    if not sq_.empty:
        ax[2].semilogx(sq_["corpus_mib"], sq_["em"], "s-", ms=6, label="SQ")
    ax[2].axhline(em0, ls="--", c="k", lw=1, label=f"exact ({em0:.3f})")
    ax[2].axhspan(em0 - se0, em0 + se0, color="k", alpha=.08)
    bm = boundary_mib(tab)
    if bm is not None:
        ax[2].axvline(bm, ls=":", c="r", lw=1.5)
    ax[2].scatter([float(e["corpus_mib"])], [em0], marker="*", s=180, c="k", zorder=5)
    ax[2].set_xlabel("corpus vector memory (MiB)")
    ax[2].set_ylabel("EM")
    ax[2].set_title("(c) quality vs memory  (red: significance boundary)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    n_gpu = len(pq_) + len(sq_)
    fig.suptitle("Knob 2 — embedding precision, A1 corpus (66,581 passages), short mode"
                 f" · {DISPLAY[m['tag']]}\n(a, b) retrieval-only sweep, generator-independent"
                 f" · (c) the {n_gpu} settings generated with this model", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- comparison figure ------------------------------------------------------

def fig_cross(knob: int, models: dict, cross: pd.DataFrame, out_path: Path) -> None:
    """Shared settings only; cost ascends left to right in every panel."""
    spec, e = KNOBS[knob], exact(knob)
    cost = spec["cost"]
    shared = list(cross["setting"])
    fam = {s: next(f[0] for f in spec["families"] if f"_{f[0]}_" in s) for s in shared}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) EM vs cost, each model with its own exact line and band
    for tag, m in models.items():
        col = COLOUR[tag]
        t = m["tables"][knob]
        t = t[t["setting"].isin(shared)]
        for fkey, flabel, mk in spec["families"]:
            d = t[t["index"] == fkey].sort_values(cost)
            if not d.empty:
                ax[0].plot(d[cost], d["em"], "-", marker=mk, ms=6, color=col,
                           label=f"{SHORT[tag]} · {flabel}")
        em0 = m["agg0"]["em"]
        se0 = np.sqrt(em0 * (1 - em0) / N)
        ax[0].axhline(em0, ls="--", c=col, lw=1)
        ax[0].axhspan(em0 - se0, em0 + se0, color=col, alpha=.08)
        ax[0].scatter([float(e[cost])], [em0], marker="*", s=180, color=col, zorder=5)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(XLABEL[knob])
    ax[0].set_ylabel("EM")
    ax[0].set_title(f"(a) quality vs {'work' if knob == 1 else 'memory'}  "
                    "(star: exact, band = ±1 SE)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    # (b) loss vs evidence delivered, with each model's projection
    cf0 = models[REF]["agg0"]["complete_frac"]
    xs = np.linspace(cross["complete_frac"].min() - 0.02, cf0, 50)
    for tag, m in models.items():
        col = COLOUR[tag]
        for fkey, flabel, mk in spec["families"]:
            c = cross[cross["setting"].map(fam) == fkey]
            if c.empty:
                continue
            y = c[f"{tag}_dEM"].to_numpy()
            err = np.clip([y - c[f"{tag}_dEM_lo"].to_numpy(),
                           c[f"{tag}_dEM_hi"].to_numpy() - y], 0, None)
            ax[1].errorbar(c["complete_frac"], y, yerr=err, fmt=mk, ms=6, color=col,
                           capsize=3, lw=1, label=f"{SHORT[tag]} · {flabel}")
        ax[1].plot(xs, (xs - cf0) * m["gap"], "--", color=col, lw=1,
                   label=f"{SHORT[tag]} · projection")
    ax[1].axhline(0, c="k", lw=.8)
    ax[1].scatter([cf0], [0], marker="*", s=180, c="k", zorder=5)
    ax[1].set_xlabel("complete_frac (evidence delivered)")
    ax[1].set_ylabel("ΔEM vs own exact baseline")
    ax[1].set_title("(b) loss vs evidence  (dashed: projection, bars: 95% CI)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

    # (c) difference in degradation vs the reference model
    others = [t for t in models if t != REF]
    xpos = np.arange(len(shared))
    for j, tag in enumerate(others):
        off = (j - (len(others) - 1) / 2) * 0.15
        y = cross[f"{tag}_ddEM"].to_numpy()
        err = np.clip([y - cross[f"{tag}_ddEM_lo"].to_numpy(),
                       cross[f"{tag}_ddEM_hi"].to_numpy() - y], 0, None)
        ax[2].errorbar(xpos + off, y, yerr=err, fmt="o", ms=6, color=COLOUR[tag],
                       capsize=4, lw=1, label=f"{SHORT[tag]} − {SHORT[REF]}")
    ax[2].axhline(0, c="k", lw=.8)
    ax[2].set_xticks(xpos)
    ax[2].set_xticklabels([short(s) for s in shared])
    ax[2].set_xlabel(f"setting (ordered by {'work' if knob == 1 else 'memory'}, ascending)")
    ax[2].set_ylabel("ΔEM(model) − ΔEM(reference)")
    ax[2].set_title("(c) difference in degradation  (95% CI)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.suptitle(f"Knob {knob} — {KNOB_NAME[knob]} across generators, A1 corpus "
                 "(66,581 passages), short mode, n=1000")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- tables -----------------------------------------------------------------

def _md(rows: list, head: list) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _ci_str(pt, lo, hi) -> str:
    return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def model_table_md(knob: int, m: dict) -> str:
    """Rows in increasing aggressiveness: exact first."""
    t, e, a0 = m["tables"][knob].iloc[::-1], exact(knob), m["agg0"]
    tail = ["—"] * 5
    if knob == 1:
        head = ["setting", "ndis / query", "ANN recall", "recall@5", "complete_frac",
                "EM", "F1", "ΔEM", "b", "c", "p", "verdict"]
        rows = [["exact (Flat)", f"{float(e['ndis']):,.0f}", "1.000",
                 f"{a0['recall@5']:.3f}", f"{a0['complete_frac']:.3f}",
                 f"{a0['em']:.3f}", f"{a0['f1']:.3f}", *tail]]
        for _, r in t.iterrows():
            rows.append([short(r["setting"]), f"{float(r['ndis']):,.0f}",
                         f"{r['ann_recall']:.3f}", f"{r['recall@5']:.3f}",
                         f"{r['complete_frac']:.3f}", f"{r['em']:.3f}", f"{r['f1']:.3f}",
                         f"{r['dEM']:+.3f}", r["b"], r["c"], f"{r['p']:.4f}", r["verdict"]])
    else:
        head = ["setting", "compression", "memory (MiB)", "ANN recall", "complete_frac",
                "EM", "F1", "ΔEM", "b", "c", "p", "verdict"]
        rows = [["exact (fp32)", "1×", f"{float(e['corpus_mib']):.1f}", "1.000",
                 f"{a0['complete_frac']:.3f}", f"{a0['em']:.3f}", f"{a0['f1']:.3f}", *tail]]
        for _, r in t.iterrows():
            rows.append([short(r["setting"]), f"{r['compression']:.0f}×",
                         f"{r['corpus_mib']:.2f}", f"{r['ann_recall']:.3f}",
                         f"{r['complete_frac']:.3f}", f"{r['em']:.3f}", f"{r['f1']:.3f}",
                         f"{r['dEM']:+.3f}", r["b"], r["c"], f"{r['p']:.4f}", r["verdict"]])
    return _md(rows, head)


def cross_md(cross: pd.DataFrame, models: dict) -> str:
    others = [t for t in models if t != REF]
    head = ["setting", "complete_frac"]
    for t in models:
        head += [f"EM {SHORT[t]}", f"ΔEM {SHORT[t]} [95% CI]"]
    for t in others:
        head += [f"ΔΔ abs {SHORT[t]} − {SHORT[REF]} [95% CI]", "ΔΔ rel [95% CI]"]
    head += [f"verdict {SHORT[t]}" for t in models]
    rows = []
    for _, r in cross.iloc[::-1].iterrows():
        row = [short(r["setting"]), f"{r['complete_frac']:.3f}"]
        for t in models:
            row += [f"{r[f'{t}_em']:.3f}",
                    _ci_str(r[f"{t}_dEM"], r[f"{t}_dEM_lo"], r[f"{t}_dEM_hi"])]
        for t in others:
            row += [_ci_str(r[f"{t}_ddEM"], r[f"{t}_ddEM_lo"], r[f"{t}_ddEM_hi"]),
                    _ci_str(r[f"{t}_ddrel"], r[f"{t}_ddrel_lo"], r[f"{t}_ddrel_hi"])]
        row += [r[f"{t}_verdict"] for t in models]
        rows.append(row)
    return _md(rows, head)


def proj_md(proj: pd.DataFrame, models: dict) -> str:
    head = ["setting", "Δcf"]
    for t in models:
        head += [f"{SHORT[t]} projected", f"{SHORT[t]} observed", f"{SHORT[t]} residual"]
    rows = []
    for _, r in proj.iloc[::-1].iterrows():
        row = [short(r["setting"]), f"{r['d_complete_frac']:+.3f}"]
        for t in models:
            row += [f"{r[f'{t}_proj']:+.3f}", f"{r[f'{t}_obs']:+.3f}", f"{r[f'{t}_resid']:+.3f}"]
        rows.append(row)
    return _md(rows, head)


def write_tables(models: dict, crosses: dict, projs: dict) -> Path:
    cf0 = models[REF]["agg0"]["complete_frac"]
    gaps = ", ".join(f"{SHORT[t]} {m['gap']:.3f}" for t, m in models.items())
    md = ["# Knobs 1 and 2 across generators", "",
          "Regime 2 (A1 pooled corpus, 66,581 passages), short mode, n=1000. Retrieval is "
          "identical across models (the reference model's stored retrieval, replayed), so "
          "distance computations, ANN recall, compression, memory and complete_frac are "
          "shared. ΔEM is against each model's own exact baseline; McNemar b = "
          "setting-only correct, c = exact-only correct. CIs: paired bootstrap over "
          f"questions, B={N_BOOT:,}, seed {SEED}.", ""]
    for knob in KNOBS:
        for tag, m in models.items():
            md += [f"## Knob {knob} — {DISPLAY[tag]}", "", model_table_md(knob, m), ""]
            m["tables"][knob].to_csv(TAB_DIR / f"knob{knob}_{tag}.csv", index=False)
        if knob in crosses:
            md += [f"## Knob {knob} — degradation compared with {DISPLAY[REF]}", "",
                   cross_md(crosses[knob], models), "",
                   f"## Knob {knob} — observed ΔEM vs evidence-loss projection", "",
                   f"Projected ΔEM = Δcf × (em_complete − em_incomplete) at each model's "
                   f"exact baseline; Δcf against complete_frac {cf0:.3f}. Gaps: {gaps}.", "",
                   proj_md(projs[knob], models), ""]
            crosses[knob].to_csv(TAB_DIR / f"knob{knob}_cross.csv", index=False)
            projs[knob].to_csv(TAB_DIR / f"knob{knob}_projection.csv", index=False)
    path = TAB_DIR / "knob12_tables.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


# --- self-test --------------------------------------------------------------

def _paired_se(b: int, c: int, n: int = N) -> float:
    return float(np.sqrt(((b + c) / n - ((b - c) / n) ** 2) / n))


def self_test() -> int:
    """Every check compares against an independent number and can fail."""
    fails, n_checks = [], 0
    q = load_model(REF)
    for knob in KNOBS:
        summ = summary(knob)
        for _, r in q["tables"][knob].iterrows():
            s = r["setting"]
            n_checks += 3
            want = float(summ.loc[summ["setting"] == s, "em"].iloc[0])
            if abs(r["em"] - want) > 1e-4:
                fails.append(f"{s}: EM from CSV {r['em']} vs summary {want}")
            d, _ = delta_boot(q["hits"][s], q["h0"])
            se = _paired_se(r["b"], r["c"])
            if se > 0 and abs(d.std() / se - 1) > 0.10:
                fails.append(f"{s}: bootstrap SD {d.std():.4f} vs paired SE {se:.4f}")
            if abs(d.mean() - r["dEM"]) > 0.002:
                fails.append(f"{s}: bootstrap mean {d.mean():+.4f} vs point {r['dEM']:+.4f}")
            if s in QWEN_PUBLISHED:
                n_checks += 1
                if (r["b"], r["c"]) != QWEN_PUBLISHED[s]:
                    fails.append(f"{s}: b,c = {(r['b'], r['c'])}, published {QWEN_PUBLISHED[s]}")

    # Negative case: break the pairing. If the SD check cannot see this, it is vacuous.
    n_checks += 1
    s = "knob1_hnsw_ef32"
    r = q["tables"][1].set_index("setting").loc[s]
    d_bad, _ = delta_boot(np.random.default_rng(0).permutation(q["hits"][s]), q["h0"])
    if d_bad.std() < 2 * _paired_se(r["b"], r["c"]):
        fails.append("unpaired resample not detected — the pairing check is vacuous")

    # The knob-2 boundary rule must reproduce notebook 08's line at m48 (3.0 MiB).
    n_checks += 1
    bm = boundary_mib(q["tables"][2])
    want = float(np.sqrt(2.0 * 3.0))     # midpoint between m32 (2.0) and m48 (3.0)
    if bm is None or abs(bm - want) > 0.02:
        fails.append(f"knob-2 boundary at {bm} MiB; expected {want:.3f}, between "
                     f"m48 (3.0, last non-loss) and m32 (2.0, first loss)")

    for f in fails:
        print(f"FAIL  {f}")
    print(f"self-test: {n_checks - len(fails)}/{n_checks} checks passed")
    return 1 if fails else 0


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    help="model tags; default: every tag with knob-1 and knob-2 results")
    ap.add_argument("--self-test", action="store_true", help="checks only, no files written")
    args = ap.parse_args()

    rc = self_test()
    if rc or args.self_test:
        return rc

    have = lambda t: all((results_dir(TAG2GEN[t]) / KNOBS[k]["file"]).exists() for k in KNOBS)
    tags = args.models or [t for t in TAG2GEN if have(t)]
    tags = [REF] + [t for t in tags if t != REF]
    models = {t: load_model(t) for t in tags}
    for t, m in models.items():
        assert m["qids"] == models[REF]["qids"], f"{t}: question set differs from {REF}"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    for m in models.values():
        for knob, fn in ((1, fig_knob1_curves), (2, fig_knob2_curves)):
            path = FIG_DIR / f"knob{knob}_curves_{m['tag']}.png"
            fn(m, path)
            print(f"wrote {path.relative_to(REPO)}")

    crosses, projs = {}, {}
    if len(models) > 1:
        for knob in KNOBS:
            crosses[knob] = cross_table(knob, models)
            projs[knob] = projection_table(models, crosses[knob])
            path = FIG_DIR / f"knob{knob}_cross_model.png"
            fig_cross(knob, models, crosses[knob], path)
            print(f"wrote {path.relative_to(REPO)}")

    path = write_tables(models, crosses, projs)
    print(f"wrote {path.relative_to(REPO)} (+ CSVs in {TAB_DIR.relative_to(REPO)}/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
