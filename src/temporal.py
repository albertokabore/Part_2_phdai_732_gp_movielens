"""Random versus chronological evaluation (evaluation-validity subsection).

The headline RMSE comes from a random split, like most published MovieLens
results. A random split lets the model train on ratings made after the ones it
is tested on: it has already seen how each movie was received later. A deployed
recommender never has that. This module refits the tuned SVD and the bias-only
baseline under three splits and reports how much the headline number depends
on that choice.

Hyperparameters are the ones selected on the random split. Nothing here is
tuned, so any difference is the split, not a search.
"""

import json

import numpy as np
import pandas as pd

from . import config as C
from .models import evaluate, make_baseline, make_svd
from .plotting import BLUE, INK_2, ORANGE, plt, save
from .splits import describe_split, make_split, train_test


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def run_temporal(ratings, tuned_params):
    rows = []
    for strategy in C.SPLIT_STRATEGIES:
        assignment = make_split(ratings, strategy)
        train, test = train_test(ratings, assignment)
        info = describe_split(ratings, assignment, strategy)
        for name, algo in (("baseline_only", make_baseline()),
                           ("svd_tuned", make_svd(tuned_params))):
            print(f"  temporal: {strategy} / {name}", flush=True)
            rec, frame = evaluate(algo, name, train, test, split=strategy, cv=False,
                                  tag="temporal")
            err = frame.predicted - frame.actual
            cold = ~frame[C.USER].isin(set(train[C.USER]))
            rows.append({**info, "model": name, "rmse": rec["rmse"], "mae": rec["mae"],
                         "rmse_warm_users": _rmse(err[~cold]),
                         "rmse_cold_users": _rmse(err[cold])})
    table = pd.DataFrame(rows)
    table.to_csv(C.RESULTS_DIR / "temporal_comparison.csv", index=False)

    piv = table.pivot(index="strategy", columns="model", values="rmse").reindex(C.SPLIT_STRATEGIES)
    labels = {"random": "Random", "user_chrono": "Per-user\nchronological",
              "global_chrono": "Global\nchronological"}
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(piv))
    # Dots, not bars: the differences are hundredths of a rating point, and a
    # bar must start at zero. A dot plot can use a tight axis honestly.
    for col, color, lab, off, va in (("baseline_only", ORANGE, "Bias-only baseline", 7, "bottom"),
                                     ("svd_tuned", BLUE, "Tuned SVD", -7, "top")):
        ax.plot(x, piv[col], color=color, marker="o", ms=7, lw=1.2, label=lab)
        for xi, v in zip(x, piv[col]):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(10, off), textcoords="offset points",
                        va=va, fontsize=8.5, color=INK_2)
    ax.set_xticks(x, [labels[s] for s in piv.index])
    ax.set_xlim(-0.4, len(piv) - 0.4)
    pad = (piv.to_numpy().max() - piv.to_numpy().min()) * 0.35 + 0.005
    ax.set(ylabel="Test RMSE", ylim=(piv.to_numpy().min() - pad, piv.to_numpy().max() + pad))
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "temporal_rmse_by_split")

    svd = table[table.model == "svd_tuned"].set_index("strategy")
    summary = {
        "svd_tuned_rmse_by_split": svd["rmse"].to_dict(),
        "rmse_increase_random_to_global": float(svd.loc["global_chrono", "rmse"] - svd.loc["random", "rmse"]),
        "rmse_increase_random_to_global_pct": float(
            (svd.loc["global_chrono", "rmse"] - svd.loc["random", "rmse"]) / svd.loc["random", "rmse"]),
        "global_chrono_warm_user_rmse": float(svd.loc["global_chrono", "rmse_warm_users"]),
        "global_chrono_cold_user_rmse": float(svd.loc["global_chrono", "rmse_cold_users"]),
        "global_chrono_cold_user_share_of_test": float(
            svd.loc["global_chrono", "test_ratings_cold_user"] / svd.loc["global_chrono", "n_test"]),
        "svd_minus_baseline_rmse_by_split": (
            svd["rmse"] - table[table.model == "baseline_only"].set_index("strategy")["rmse"]).to_dict(),
        "future_information_by_split": svd["test_ratings_with_future_train_ratings_of_same_item"].to_dict(),
    }
    (C.RESULTS_DIR / "temporal_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return table, summary
