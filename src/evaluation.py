"""Final model evaluation (Deliverable 2, Key Component 2).

Fits the tuned SVD on the full training partition and scores the held-out test
partition exactly once. Then asks where the errors are, because a single RMSE
says how wrong the model is on average and nothing about whom it is wrong for:

  - by true rating: does the model compress toward the mean?
  - by user activity and item popularity: is it worse for light users and
    niche movies, as matrix factorization theory predicts?
  - as a ranking: RMSE grades rating prediction, but a recommender is used to
    rank. precision@k and recall@k grade the lists users would actually see.
  - the largest individual misses, by title, for the "areas for improvement"
    discussion.

It also produces top-10 lists for every user, which fairness.py uses for the
popularity-bias analysis.
"""

import json

import numpy as np
import pandas as pd

from . import config as C
from .models import accuracy, evaluate, make_svd
from .plotting import BLUE, INK_2, plt, save


# --- Ranking metrics ------------------------------------------------------------
def precision_recall_at_k(frame, k=C.TOP_K, threshold=C.RELEVANCE_THRESHOLD, relevant=None):
    """Per-user precision@k and recall@k on the test partition.

    Follows the definition in Surprise's FAQ: a test item is relevant if its
    true rating is >= threshold, and recommended if it is in the user's top k
    by predicted rating and its predicted rating is >= threshold. Users with no
    recommended items score 0 precision, users with no relevant items 0 recall,
    as in the FAQ, so the averages are conservative.

    relevant, if given, maps user -> number of relevant test items before any
    filtering. A mitigation that removes candidates is then charged in recall
    for every relevant item it made unrecommendable.
    """
    rows = []
    for uid, g in frame.groupby(C.USER):
        g = g.sort_values("predicted", ascending=False)
        top = g.head(k)
        n_rel = int(relevant.get(uid, 0)) if relevant is not None else int((g.actual >= threshold).sum())
        n_rec_k = int((top.predicted >= threshold).sum())
        n_both = int(((top.actual >= threshold) & (top.predicted >= threshold)).sum())
        rows.append({C.USER: uid, "n_test": len(g), "n_relevant": n_rel,
                     "precision_at_k": n_both / n_rec_k if n_rec_k else 0.0,
                     "recall_at_k": n_both / n_rel if n_rel else 0.0})
    per_user = pd.DataFrame(rows)
    return {
        "k": k, "threshold": threshold,
        "precision_at_k": float(per_user.precision_at_k.mean()),
        "recall_at_k": float(per_user.recall_at_k.mean()),
        "users_scored": int(len(per_user)),
    }, per_user


# --- Top-N lists --------------------------------------------------------------
def score_matrix(algo):
    """Full user x item score matrix from a fitted SVD, checked against predict().

    est = mu + b_u + b_i + q_i . p_u, clipped to the rating scale, which is
    exactly what SVD.predict computes for known users and items. Scoring 1.5
    million pairs through predict() one at a time takes minutes; this takes
    under a second. The equivalence is checked on a random sample every run, so
    a Surprise version that changed the formula fails loudly here instead of
    producing wrong lists.
    """
    ts = algo.trainset
    lo, hi = ts.rating_scale
    scores = np.clip(ts.global_mean + algo.bu[:, None] + algo.bi[None, :] + algo.pu @ algo.qi.T, lo, hi)
    rng = np.random.default_rng(C.SEED)
    us = rng.integers(0, ts.n_users, 300)
    it = rng.integers(0, ts.n_items, 300)
    ref = np.array([algo.predict(ts.to_raw_uid(u), ts.to_raw_iid(i)).est for u, i in zip(us, it)])
    worst = float(np.max(np.abs(ref - scores[us, it])))
    if worst > 1e-6:
        raise AssertionError(f"Vectorized scores disagree with SVD.predict by {worst:.2e}.")
    return scores, worst


def top_n_from_scores(trainset, scores, n=C.TOP_K, exclude_items=None):
    """Top-n unseen movies per user. exclude_items: raw item ids never recommended."""
    masked = scores.copy()
    for u in trainset.all_users():
        masked[u, [i for i, _ in trainset.ur[u]]] = -np.inf
    if exclude_items is not None and len(exclude_items):
        cols = [trainset.to_inner_iid(i) for i in exclude_items]
        masked[:, cols] = -np.inf
    rows = []
    for u in trainset.all_users():
        top = np.argpartition(-masked[u], n)[:n]
        top = top[np.argsort(-masked[u, top])]
        rows += [{C.USER: trainset.to_raw_uid(u), "rank": r + 1, C.ITEM: trainset.to_raw_iid(i),
                  "predicted": float(masked[u, i])} for r, i in enumerate(top)]
    return pd.DataFrame(rows)


def top_n_all_users(algo, n=C.TOP_K):
    """Top-n unseen movies for every training user, from a fitted SVD."""
    scores, worst = score_matrix(algo)
    return top_n_from_scores(algo.trainset, scores, n), worst


# --- Error anatomy -----------------------------------------------------------
def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def error_by_true_rating(frame):
    f = frame.assign(err=frame.predicted - frame.actual)
    t = f.groupby("actual").agg(n=("err", "size"), mean_predicted=("predicted", "mean"),
                                mean_signed_error=("err", "mean"),
                                rmse=("err", _rmse)).reset_index()
    t.to_csv(C.RESULTS_DIR / "final_error_by_true_rating.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    data = [f.loc[f.actual == r, "predicted"] for r in t.actual]
    ax.boxplot(data, positions=t.actual, widths=0.5, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
               medianprops={"color": BLUE, "lw": 2},
               whiskerprops={"color": BLUE}, capprops={"color": BLUE})
    ax.plot(t.actual, t.actual, ls="none", marker="_", ms=22, mew=2, color=INK_2)
    ax.set(xlabel="True rating", ylabel="Predicted rating", ylim=(0.8, 5.2))
    ax.set_xticks(list(t.actual), [f"{int(v)}" for v in t.actual])
    ax.grid(axis="x", visible=False)
    ax.annotate("gray tick = perfect prediction", (1, 5.05), fontsize=8.5, color=INK_2)
    save(fig, "final_predicted_by_true_rating")
    return t


def error_by_support(frame, train_df):
    """RMSE by how much training data the user and the movie had."""
    u_n = train_df.groupby(C.USER).size().rename("user_train_ratings")
    i_n = train_df.groupby(C.ITEM).size().rename("item_train_ratings")
    f = (frame.merge(u_n, left_on=C.USER, right_index=True, how="left")
         .merge(i_n, left_on=C.ITEM, right_index=True, how="left")
         .fillna({"user_train_ratings": 0, "item_train_ratings": 0}))
    f["err"] = f.predicted - f.actual
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, col, title in ((axes[0], "user_train_ratings", "Users"),
                           (axes[1], "item_train_ratings", "Movies")):
        f["bin"] = pd.qcut(f[col], 5, duplicates="drop")
        t = f.groupby("bin", observed=True).agg(n=("err", "size"), rmse=("err", _rmse),
                                                lo=(col, "min"), hi=(col, "max")).reset_index()
        t["label"] = [f"{int(a)}-{int(b)}" for a, b in zip(t.lo, t.hi)]
        t.drop(columns="bin").to_csv(C.RESULTS_DIR / f"final_rmse_by_{col}.csv", index=False)
        ax.plot(range(len(t)), t.rmse, color=BLUE, marker="o", ms=7, lw=1.2)
        for x, v in enumerate(t.rmse):
            ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_2)
        ax.set_xticks(range(len(t)), t.label)
        ax.set_xlim(-0.5, len(t) - 0.5)
        ax.set(title=title, xlabel="Training ratings (quintile range)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", visible=False)
        out[col] = t[["label", "n", "rmse"]].to_dict("records")
    axes[0].set_ylabel("Test RMSE")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.15)
    save(fig, "final_rmse_by_support")
    return out


def largest_errors(frame, items, n=15):
    t = (frame.assign(abs_err=(frame.predicted - frame.actual).abs())
         .nlargest(n, "abs_err")
         .merge(items[[C.ITEM, "title"]], on=C.ITEM, how="left"))
    t.to_csv(C.RESULTS_DIR / "final_largest_errors.csv", index=False)
    return t


# --- Driver -------------------------------------------------------------------
def run_final_evaluation(train_df, test_df, items, tuned_params, roster_table, roster_frames):
    tuned = make_svd(tuned_params)
    rec, frame = evaluate(tuned, "svd", train_df, test_df, tag="tuned")
    train_preds = tuned.test(tuned.trainset.build_testset())
    rec["train_rmse"] = accuracy.rmse(train_preds, verbose=False)
    rec["train_test_rmse_gap"] = rec["rmse"] - rec["train_rmse"]
    frame.merge(items[[C.ITEM, "title"]], on=C.ITEM, how="left").to_csv(
        C.RESULTS_DIR / "final_test_predictions.csv", index=False)

    # Comparison: roster plus tuned model, on the same held-out partition.
    comp = pd.concat([roster_table, pd.DataFrame([rec])], ignore_index=True)
    comp = comp[["model", "rmse", "mae", "cv_rmse_mean", "cv_rmse_sd", "mean_signed_error",
                 "fit_seconds"]]
    comp.to_csv(C.RESULTS_DIR / "final_model_comparison.csv", index=False)

    default = roster_table.set_index("model").loc["svd"]
    improvement = {
        "default_svd_test_rmse": float(default.rmse),
        "tuned_svd_test_rmse": rec["rmse"],
        "rmse_reduction": float(default.rmse - rec["rmse"]),
        "rmse_reduction_pct": float((default.rmse - rec["rmse"]) / default.rmse),
        "default_svd_test_mae": float(default.mae),
        "tuned_svd_test_mae": rec["mae"],
        "mae_reduction": float(default.mae - rec["mae"]),
        "default_svd_cv_rmse_sd": float(default.cv_rmse_sd),
        # A reduction smaller than the fold-to-fold SD of the default model is
        # inside noise, and the report must not call it an improvement.
        "reduction_exceeds_default_cv_sd": bool(default.rmse - rec["rmse"] > default.cv_rmse_sd),
        "baseline_only_test_rmse": float(roster_table.set_index("model").loc["baseline_only"].rmse),
        "tuned_vs_baseline_only_rmse_reduction": float(
            roster_table.set_index("model").loc["baseline_only"].rmse - rec["rmse"]),
    }

    ranking = {}
    frames = {**roster_frames, "svd_tuned": frame}
    for name in ("baseline_only", "knn_item", "svd", "svd_tuned"):
        summary, per_user = precision_recall_at_k(frames[name])
        ranking[name] = summary
        if name == "svd_tuned":
            per_user.to_csv(C.RESULTS_DIR / "final_precision_recall_per_user.csv", index=False)
    pd.DataFrame(ranking).T.to_csv(C.RESULTS_DIR / "final_ranking_metrics.csv",
                                   index_label="model")

    by_rating = error_by_true_rating(frame)
    by_support = error_by_support(frame, train_df)
    worst = largest_errors(frame, items)

    top_n, check = top_n_all_users(tuned)
    top_n.merge(items[[C.ITEM, "title"]], on=C.ITEM, how="left").to_csv(
        C.RESULTS_DIR / "final_top10_all_users.csv", index=False)
    per_user_n = train_df.groupby(C.USER).size()
    examples = {"heavy": int(per_user_n.idxmax()),
                "median": int((per_user_n - per_user_n.median()).abs().idxmin()),
                "light": int(per_user_n.idxmin())}
    ex_rows = []
    for kind, uid in examples.items():
        t = top_n[top_n[C.USER] == uid].merge(items[[C.ITEM, "title"]], on=C.ITEM)
        ex_rows += [{"user_type": kind, C.USER: uid, "train_ratings": int(per_user_n[uid]),
                     **r} for r in t[["rank", "title", "predicted"]].to_dict("records")]
    pd.DataFrame(ex_rows).to_csv(C.RESULTS_DIR / "final_example_recommendations.csv", index=False)

    summary = {
        "tuned_params": tuned_params,
        "test": {k: rec[k] for k in ("rmse", "mae", "mean_signed_error", "train_rmse",
                                     "train_test_rmse_gap", "cv_rmse_mean", "cv_rmse_sd")},
        "improvement_over_default": improvement,
        "ranking": ranking,
        "compression": {
            "sd_actual": float(frame.actual.std()),
            "sd_predicted": float(frame.predicted.std()),
            "mean_predicted_when_true_is_1": float(by_rating.set_index("actual").loc[1, "mean_predicted"]),
            "mean_predicted_when_true_is_5": float(by_rating.set_index("actual").loc[5, "mean_predicted"]),
        },
        "rmse_by_support": by_support,
        "largest_error_titles": worst["title"].head(5).tolist(),
        "vectorized_topn_max_abs_diff_vs_predict": check,
    }
    (C.RESULTS_DIR / "final_evaluation.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary, tuned, frame, top_n
