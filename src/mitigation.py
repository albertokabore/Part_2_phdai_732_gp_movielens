"""Minimum-support mitigation (Ethical Considerations and Real-World Application).

The problem this addresses, measured in fairness.py: the RMSE-tuned SVD pushes
movies with a handful of training ratings into most users' top-10 lists. Heavy
regularization shrinks the personal taste factors toward zero, so rankings
fall back on each movie's bias term, and a movie with two perfect ratings has
the highest bias in the catalog.

The mitigation is the one production systems use most often: do not recommend
an item until it has at least N ratings. The sweep over N reports what each
threshold buys (fewer thin-evidence recommendations, more personalized lists)
and what it costs (catalog coverage, and recall on relevant movies it can no
longer recommend), so the report can discuss a tradeoff rather than defend a
hand-picked number. N is never chosen by test-set performance here.

The untuned SVD is scored alongside at N = 0 as a reference point: it is
less accurate by RMSE but less regularized, so it shows what the tuning
traded away.
"""

import json

import numpy as np
import pandas as pd

from . import config as C
from .evaluation import precision_recall_at_k, score_matrix, top_n_from_scores
from .fairness import list_overlap
from .plotting import BLUE, INK_2, ORANGE, plt, save


def _list_metrics(top_n, counts):
    rec_counts = top_n[C.ITEM].value_counts()
    support = counts.reindex(top_n[C.ITEM]).to_numpy()
    return {
        "distinct_items_recommended": int(top_n[C.ITEM].nunique()),
        "catalog_coverage": float(top_n[C.ITEM].nunique() / len(counts)),
        "share_of_recs_to_thin_items": float((support < C.THIN_EVIDENCE).mean()),
        "mean_pairwise_list_overlap": list_overlap(top_n),
        "most_recommended_item_share_of_users": float(rec_counts.iloc[0] / top_n[C.USER].nunique()),
        "median_train_ratings_of_recommended_items": float(np.median(support)),
    }


def run_mitigation(tuned, tuned_frame, default_svd, default_frame, train_df):
    counts = train_df.groupby(C.ITEM).size()
    relevant = (tuned_frame[tuned_frame.actual >= C.RELEVANCE_THRESHOLD]
                .groupby(C.USER).size())
    scores, _ = score_matrix(tuned)
    rows = []

    def ranking(frame, eligible):
        f = frame[frame[C.ITEM].isin(eligible)]
        summary, _ = precision_recall_at_k(f, relevant=relevant)
        return summary

    # Reference: untuned SVD, no filter.
    d_scores, _ = score_matrix(default_svd)
    d_top = top_n_from_scores(default_svd.trainset, d_scores)
    r = ranking(default_frame, counts.index)
    rows.append({"model": "svd_default", "min_support": 0,
                 "precision_at_k": r["precision_at_k"], "recall_at_k": r["recall_at_k"],
                 **_list_metrics(d_top, counts)})

    for n in C.MIN_SUPPORT_SWEEP:
        eligible = counts.index[counts >= n]
        excluded = counts.index[counts < n]
        top = top_n_from_scores(tuned.trainset, scores, exclude_items=list(excluded))
        r = ranking(tuned_frame, eligible)
        rows.append({"model": "svd_tuned", "min_support": n,
                     "items_eligible": int(len(eligible)),
                     "share_of_catalog_eligible": float(len(eligible) / len(counts)),
                     "precision_at_k": r["precision_at_k"], "recall_at_k": r["recall_at_k"],
                     **_list_metrics(top, counts)})
        print(f"  mitigation: min_support={n}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(C.RESULTS_DIR / "mitigation_min_support.csv", index=False)
    _figure(table)

    base = table[(table.model == "svd_tuned") & (table.min_support == 0)].iloc[0]
    out = {"k": C.TOP_K, "thin_evidence_threshold": C.THIN_EVIDENCE,
           "sweep": table.to_dict("records"),
           "tuned_unfiltered": base.to_dict()}
    (C.RESULTS_DIR / "mitigation_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return table


def _figure(table):
    t = table[table.model == "svd_tuned"].reset_index(drop=True)
    x = np.arange(len(t))
    labels = ["none" if n == 0 else f"{n}+" for n in t.min_support]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    for col, color, lab in (("precision_at_k", BLUE, f"Precision@{C.TOP_K}"),
                            ("recall_at_k", ORANGE, f"Recall@{C.TOP_K}")):
        ax.plot(x, t[col], color=color, marker="o", ms=6, lw=1.4, label=lab)
        ax.annotate(f"{t[col].iloc[-1]:.3f}".replace("0.", "."), (x[-1], t[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    ax.set(title="Ranking quality", xlabel="Minimum training ratings to be recommended", ylim=(0, 1))
    ax.set_xticks(x, labels)
    ax.legend(loc="upper left")

    ax = axes[1]
    for col, color, lab in (("share_of_recs_to_thin_items", BLUE,
                             f"Recs to movies with <{C.THIN_EVIDENCE} ratings"),
                            ("mean_pairwise_list_overlap", ORANGE, "Overlap between users' lists")):
        ax.plot(x, t[col], color=color, marker="o", ms=6, lw=1.4, label=lab)
    ax.set(title="Exposure and personalization", xlabel="Minimum training ratings to be recommended",
           ylim=(0, 1))
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right")
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "mitigation_min_support")
