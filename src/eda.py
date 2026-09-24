"""Exploratory analysis of the user-item rating matrix.

Reads the cleaned frames from load_data(); never re-reads raw files. Every
figure has a matching CSV or JSON under results/ so a drafter can cite the
number instead of reading it off a plot.
"""

import json

import numpy as np
import pandas as pd

from . import config as C
from .plotting import BLUE, INK_2, label_bars, plt, save


def _gini(counts):
    """Gini coefficient of a count vector. 0 = perfectly even, 1 = one item has everything."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def rating_distribution(ratings):
    dist = ratings[C.RATING].value_counts().sort_index()
    table = pd.DataFrame({"rating": dist.index.astype(int), "count": dist.values,
                          "share": dist.values / dist.sum()})
    table.to_csv(C.RESULTS_DIR / "eda_rating_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(table["rating"], table["count"], color=BLUE, width=0.62)
    for b, s in zip(bars, table["share"]):
        ax.annotate(f"{s:.1%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK_2)
    ax.set(xlabel="Rating", ylabel="Number of ratings", xticks=table["rating"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="x", visible=False)
    save(fig, "eda_rating_distribution")
    return table


def long_tails(ratings):
    per_user = ratings.groupby(C.USER).size().sort_values(ascending=False)
    per_item = ratings.groupby(C.ITEM).size().sort_values(ascending=False)

    def top_share(s, frac):
        k = int(np.ceil(len(s) * frac))
        return float(s.iloc[:k].sum() / s.sum())

    summary = {
        "users": {"median": float(per_user.median()), "mean": float(per_user.mean()),
                  "min": int(per_user.min()), "max": int(per_user.max()),
                  "gini": _gini(per_user),
                  "share_of_ratings_from_top_10pct_users": top_share(per_user, 0.10)},
        "items": {"median": float(per_item.median()), "mean": float(per_item.mean()),
                  "min": int(per_item.min()), "max": int(per_item.max()),
                  "gini": _gini(per_item),
                  "share_of_ratings_on_top_10pct_items": top_share(per_item, 0.10),
                  "share_of_ratings_on_top_20pct_items": top_share(per_item, 0.20),
                  "items_with_1_rating": int((per_item == 1).sum()),
                  "items_with_under_10_ratings": int((per_item < 10).sum())},
    }
    (C.RESULTS_DIR / "eda_long_tail.json").write_text(json.dumps(summary, indent=2))
    per_user.rename("n_ratings").to_csv(C.RESULTS_DIR / "eda_ratings_per_user.csv")
    per_item.rename("n_ratings").to_csv(C.RESULTS_DIR / "eda_ratings_per_item.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, s, who in ((axes[0], per_user, "Users"), (axes[1], per_item, "Movies")):
        ax.plot(np.arange(1, len(s) + 1), s.values, color=BLUE)
        ax.set_yscale("log")
        ax.set(xlabel=f"{who} ranked by number of ratings", title=who)
        ax.axhline(s.median(), color=INK_2, lw=0.8, ls="--")
        ax.annotate(f"median {s.median():.0f}", (len(s) * 0.62, s.median()),
                    xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    axes[0].set_ylabel("Ratings (log scale)")
    save(fig, "eda_long_tail")
    return summary


def popularity_vs_mean(ratings):
    """Rare movies have extreme means. This is the case for bias terms and regularization."""
    g = ratings.groupby(C.ITEM)[C.RATING].agg(n="size", mean="mean")
    g.to_csv(C.RESULTS_DIR / "eda_item_mean_vs_count.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.scatter(g["n"], g["mean"], s=9, color=BLUE, alpha=0.45, linewidths=0)
    ax.set_xscale("log")
    ax.axhline(ratings[C.RATING].mean(), color=INK_2, lw=0.8, ls="--")
    ax.annotate(f"global mean {ratings[C.RATING].mean():.2f}", (g["n"].max(), ratings[C.RATING].mean()),
                xytext=(0, -14), textcoords="offset points", ha="right", fontsize=8.5, color=INK_2,
                bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set(xlabel="Ratings per movie (log scale)", ylabel="Mean rating", ylim=(0.8, 5.2))
    save(fig, "eda_item_mean_vs_count")
    return {
        "sd_of_item_means_under_10_ratings": float(g.loc[g.n < 10, "mean"].std()),
        "sd_of_item_means_100_plus_ratings": float(g.loc[g.n >= 100, "mean"].std()),
        "items_with_mean_5_and_under_5_ratings": int(((g["mean"] == 5) & (g.n < 5)).sum()),
    }


def activity_over_time(ratings):
    weekly = ratings.set_index("datetime")[C.RATING].resample("W").size()
    weekly.rename("n_ratings").to_csv(C.RESULTS_DIR / "eda_weekly_ratings.csv")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(weekly.index, weekly.values, color=BLUE)
    ax.set(ylabel="Ratings per week")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.autofmt_xdate()
    save(fig, "eda_weekly_ratings")


def demographics(ratings, users):
    """Who the users are, and how much of the rating signal each group contributes."""
    merged = ratings.merge(users, on=C.USER, how="left")
    rows = []
    for col in ("gender", "age_band", "occupation"):
        by = merged.groupby(col, observed=True)
        t = pd.DataFrame({
            "users": by[C.USER].nunique(),
            "ratings": by.size(),
            "mean_rating": by[C.RATING].mean(),
            "ratings_per_user": by.size() / by[C.USER].nunique(),
        })
        t["share_of_users"] = t["users"] / users[C.USER].nunique()
        t["share_of_ratings"] = t["ratings"] / len(ratings)
        t.insert(0, "attribute", col)
        rows.append(t.reset_index().rename(columns={col: "group"}))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(C.RESULTS_DIR / "eda_demographics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 2.6, 2.6]})
    g = table[table.attribute == "gender"]
    bars = axes[0].bar(g["group"].astype(str), g["share_of_ratings"], color=BLUE, width=0.55)
    label_bars(axes[0], bars, fmt="{:.0%}")
    axes[0].set(title="Gender", ylabel="Share of all ratings")
    a = table[table.attribute == "age_band"]
    bars = axes[1].bar(a["group"].astype(str), a["share_of_ratings"], color=BLUE, width=0.62)
    label_bars(axes[1], bars, fmt="{:.0%}")
    axes[1].set(title="Age band")
    axes[1].tick_params(axis="x", labelsize=8.5)
    o = table[table.attribute == "occupation"].sort_values("share_of_ratings").tail(10)
    axes[2].barh(o["group"].astype(str), o["share_of_ratings"], color=BLUE, height=0.62)
    axes[2].set(title="Occupation (top 10)", xlabel="Share of all ratings")
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", visible=False)
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[2].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "eda_demographics")
    return table


def genres(ratings, items):
    merged = ratings.merge(items[[C.ITEM] + C.GENRES], on=C.ITEM)
    counts = merged[C.GENRES].sum().sort_values()
    means = pd.Series({g: merged.loc[merged[g] == 1, C.RATING].mean() for g in C.GENRES})
    table = pd.DataFrame({"ratings": counts, "mean_rating": means.reindex(counts.index),
                          "movies": items[C.GENRES].sum().reindex(counts.index)})
    table.to_csv(C.RESULTS_DIR / "eda_genres.csv", index_label="genre")
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.62)
    ax.set(xlabel="Ratings (a movie counts once per genre it carries)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", visible=False)
    save(fig, "eda_genres")
    return table


def run_eda(ratings, users, items):
    out = {"long_tail": long_tails(ratings), "popularity_vs_mean": popularity_vs_mean(ratings)}
    rating_distribution(ratings)
    activity_over_time(ratings)
    demographics(ratings, users)
    genres(ratings, items)
    (C.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    from .data import load_data
    r, u, i, _ = load_data()
    print(json.dumps(run_eda(r, u, i), indent=2))
