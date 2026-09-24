"""Demographic error disparities and popularity bias (Ethical Considerations).

Two separate questions, because they are separate harms:

1. Is the model less accurate for some groups of users? Test RMSE by gender,
   age band, and occupation, with 95% confidence intervals from a bootstrap
   that resamples users, not ratings. Ratings from one user are correlated, so
   resampling ratings would make every interval too narrow.

   Each group's mean training ratings per user is reported alongside, because
   light raters are harder to model regardless of who they are. A gap that
   tracks activity is a data-volume problem; a gap that survives it is not.

2. Does the model concentrate exposure on already-popular movies? Compares how
   the top-10 lists distribute across item popularity deciles against how the
   training ratings do. Popularity bias is a fairness issue for items and the
   people who make them, and a quality issue for users whose tastes sit in the
   tail.
"""

import json

import numpy as np
import pandas as pd

from . import config as C
from .plotting import BLUE, INK_2, ORANGE, plt, save


def _cluster_bootstrap_rmse(sse, n, reps, rng):
    """RMSE resampled by user. sse and n are per-user sums over test ratings."""
    idx = rng.integers(0, len(sse), (reps, len(sse)))
    return np.sqrt(sse[idx].sum(axis=1) / n[idx].sum(axis=1))


def group_errors(frame, users, train_df):
    rng = np.random.default_rng(C.SEED)
    f = frame.merge(users, on=C.USER, how="left")
    f["sq"] = (f.predicted - f.actual) ** 2
    f["abs"] = (f.predicted - f.actual).abs()
    f["signed"] = f.predicted - f.actual
    per_user = f.groupby(C.USER).agg(sse=("sq", "sum"), n=("sq", "size"))
    train_n = train_df.groupby(C.USER).size().rename("train_ratings")
    per_user = per_user.join(train_n).join(users.set_index(C.USER)[["gender", "age_band", "occupation"]])

    overall = float(np.sqrt(f.sq.mean()))
    rows, boots = [], {}
    for attr in ("gender", "age_band", "occupation"):
        for grp, pu in per_user.groupby(attr, observed=True):
            g = f[f[attr] == grp]
            b = _cluster_bootstrap_rmse(pu.sse.to_numpy(), pu.n.to_numpy().astype(float),
                                        C.BOOTSTRAP_REPS, rng)
            boots[(attr, str(grp))] = b
            rows.append({
                "attribute": attr, "group": str(grp),
                "n_users": int(len(pu)), "n_test_ratings": int(len(g)),
                "rmse": float(np.sqrt(g.sq.mean())),
                "rmse_ci_low": float(np.percentile(b, 2.5)),
                "rmse_ci_high": float(np.percentile(b, 97.5)),
                "mae": float(g["abs"].mean()),
                "mean_signed_error": float(g.signed.mean()),
                "mean_actual": float(g.actual.mean()),
                "mean_predicted": float(g.predicted.mean()),
                "mean_train_ratings_per_user": float(pu.train_ratings.mean()),
                "rmse_minus_overall": float(np.sqrt(g.sq.mean()) - overall),
            })
    table = pd.DataFrame(rows)
    table.to_csv(C.RESULTS_DIR / "fairness_group_errors.csv", index=False)

    # Gender gap with a bootstrap CI on the difference itself.
    gap = {}
    if ("gender", "F") in boots and ("gender", "M") in boots:
        d = boots[("gender", "F")] - boots[("gender", "M")]
        gt = table[table.attribute == "gender"].set_index("group")
        gap = {"rmse_F_minus_M": float(gt.loc["F", "rmse"] - gt.loc["M", "rmse"]),
               "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
               "ci_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

    # Does error track activity? Correlation across users between training
    # volume and per-user RMSE, which is the confound for every group gap.
    per_user["rmse"] = np.sqrt(per_user.sse / per_user.n)
    activity_corr = float(per_user[["train_ratings", "rmse"]].corr(method="spearman").iloc[0, 1])

    # The direct test of the confound: compare genders within activity
    # quartiles. If the gap were a data-volume effect it would vanish here.
    f = f.join(train_n, on=C.USER)
    f["activity_quartile"] = pd.qcut(f.train_ratings, 4, labels=["Q1 (lightest)", "Q2", "Q3", "Q4 (heaviest)"])
    strat = (f.groupby(["activity_quartile", "gender"], observed=True)
             .agg(rmse=("sq", lambda x: float(np.sqrt(x.mean()))), n_test_ratings=("sq", "size"),
                  n_users=(C.USER, "nunique")).reset_index())
    strat.to_csv(C.RESULTS_DIR / "fairness_gender_by_activity.csv", index=False)
    wide = strat.pivot(index="activity_quartile", columns="gender", values="rmse")
    gap_by_quartile = (wide["F"] - wide["M"]).to_dict() if {"F", "M"} <= set(wide.columns) else {}

    _group_figure(table, overall)
    return table, {"overall_rmse": overall, "gender_gap": gap,
                   "gender_gap_by_activity_quartile": {str(k): float(v) for k, v in gap_by_quartile.items()},
                   "gender_gap_positive_in_every_quartile": bool(gap_by_quartile) and all(
                       v > 0 for v in gap_by_quartile.values()),
                   "spearman_user_train_ratings_vs_user_rmse": activity_corr,
                   "age_band_rmse_range": float(table[table.attribute == "age_band"].rmse.max()
                                                - table[table.attribute == "age_band"].rmse.min())}


def _group_figure(table, overall):
    occ = table[table.attribute == "occupation"].sort_values("rmse")
    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[2, 6], hspace=0.55, wspace=0.45)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    lo = table.rmse_ci_low.min() - 0.01
    hi = table.rmse_ci_high.max() + 0.01
    for ax, (attr, title) in zip(axes, [("gender", "Gender"), ("age_band", "Age band"),
                                        ("occupation", "Occupation")]):
        t = occ if attr == "occupation" else table[table.attribute == attr]
        y = np.arange(len(t))
        ax.hlines(y, t.rmse_ci_low, t.rmse_ci_high, color=BLUE, alpha=0.45, lw=3)
        ax.plot(t.rmse, y, "o", color=BLUE, ms=6)
        ax.axvline(overall, color=INK_2, lw=0.8, ls="--")
        ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(t.group, t.n_users)], fontsize=8.5)
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.set_xlim(lo, hi)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Test RMSE with 95% CI")
    axes[2].set_xlabel("Test RMSE with 95% CI (dashed line = overall)")
    save(fig, "fairness_rmse_by_group")


def list_overlap(top_n, pairs=20000):
    """Mean Jaccard overlap between two random users' top-n lists.

    0 means every user gets a different list; 1 means everyone gets the same
    one. The standard inverse of "personalization" in the recommender
    literature. Sampled pairs, seeded, so it is fast and reproducible.
    """
    lists = top_n.groupby(C.USER)[C.ITEM].apply(frozenset).to_numpy()
    rng = np.random.default_rng(C.SEED)
    a = rng.integers(0, len(lists), pairs)
    b = rng.integers(0, len(lists), pairs)
    keep = a != b
    return float(np.mean([len(lists[x] & lists[y]) / len(lists[x] | lists[y])
                          for x, y in zip(a[keep], b[keep])]))


def popularity_bias(top_n, train_df):
    counts = train_df.groupby(C.ITEM).size().sort_values(ascending=False)
    n_items = len(counts)
    decile = pd.Series(np.minimum(np.arange(n_items) * 10 // n_items, 9) + 1,
                       index=counts.index, name="popularity_decile")
    rec = top_n.join(decile, on=C.ITEM)
    rat = train_df.join(decile, on=C.ITEM)
    table = pd.DataFrame({
        "share_of_catalog": decile.value_counts(normalize=True).sort_index(),
        "share_of_train_ratings": rat.popularity_decile.value_counts(normalize=True).sort_index(),
        "share_of_recommendations": rec.popularity_decile.value_counts(normalize=True)
        .reindex(range(1, 11), fill_value=0).sort_index(),
    })
    table.index.name = "popularity_decile (1 = most popular)"
    table.to_csv(C.RESULTS_DIR / "fairness_popularity_deciles.csv")

    rec_counts = top_n[C.ITEM].value_counts()
    all_counts = rec_counts.reindex(counts.index, fill_value=0).to_numpy()
    x = np.sort(all_counts.astype(float))
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum()))
    pop_pct = counts.rank(pct=True)
    means = train_df.groupby(C.ITEM)[C.RATING].mean()
    top_items = pd.DataFrame({
        "users_recommended": rec_counts.head(15),
        "share_of_users": rec_counts.head(15) / top_n[C.USER].nunique(),
        "train_ratings": counts.reindex(rec_counts.head(15).index),
        "train_mean_rating": means.reindex(rec_counts.head(15).index),
    })
    top_items.index.name = C.ITEM
    top_items.to_csv(C.RESULTS_DIR / "fairness_most_recommended.csv")
    rec_support = counts.reindex(top_n[C.ITEM]).to_numpy()
    summary = {
        "catalog_coverage": float(top_n[C.ITEM].nunique() / n_items),
        "distinct_items_recommended": int(top_n[C.ITEM].nunique()),
        "share_of_recs_from_top_decile": float(table.loc[1, "share_of_recommendations"]),
        "share_of_train_ratings_on_top_decile": float(table.loc[1, "share_of_train_ratings"]),
        "share_of_recs_from_bottom_half": float(table.loc[6:10, "share_of_recommendations"].sum()),
        "gini_of_recommendation_counts": gini,
        "mean_popularity_percentile_of_recommended_items": float(pop_pct.reindex(top_n[C.ITEM]).mean()),
        "most_recommended_item_share_of_users": float(rec_counts.iloc[0] / top_n[C.USER].nunique()),
        "most_recommended_item_train_ratings": int(counts.get(rec_counts.index[0], 0)),
        # The small-sample problem the EDA predicted: movies the model knows
        # least about, pushed to the most people.
        "share_of_recs_to_items_under_10_train_ratings": float((rec_support < C.THIN_EVIDENCE).mean()),
        "share_of_catalog_under_10_train_ratings": float((counts < C.THIN_EVIDENCE).mean()),
        "mean_pairwise_list_overlap": list_overlap(top_n),
        "items_under_10_train_ratings_in_over_half_of_lists": int(
            ((top_items.train_ratings < 10) & (top_items.share_of_users > 0.5)).sum()),
    }

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x_ = np.arange(1, 11)
    w = 0.38
    b1 = ax.bar(x_ - w / 2, table.share_of_train_ratings, w, color=BLUE, label="Training ratings")
    b2 = ax.bar(x_ + w / 2, table.share_of_recommendations, w, color=ORANGE, label="Top-10 recommendations")
    ax.axhline(0.10, color=INK_2, lw=0.8, ls="--")
    ax.annotate("equal exposure (10%)", (10.4, 0.10), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK_2)
    ax.set(xlabel="Movie popularity decile (1 = most rated in training)", ylabel="Share",
           xticks=x_)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92))
    ax.grid(axis="x", visible=False)
    save(fig, "fairness_popularity_bias")
    return table, summary


def run_fairness(frame, users, train_df, top_n, items=None):
    table, group_summary = group_errors(frame, users, train_df)
    pop_table, pop_summary = popularity_bias(top_n, train_df)
    if items is not None:
        mr = pd.read_csv(C.RESULTS_DIR / "fairness_most_recommended.csv")
        mr.merge(items[[C.ITEM, "title"]], on=C.ITEM, how="left").to_csv(
            C.RESULTS_DIR / "fairness_most_recommended.csv", index=False)
    out = {"groups": group_summary, "popularity": pop_summary}
    (C.RESULTS_DIR / "fairness_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out
