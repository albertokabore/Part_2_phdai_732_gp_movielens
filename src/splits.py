"""Train/test partitions, frozen once and written to disk.

Every model in the report is fit and scored on the same partition. Regenerate a
split with a different seed and the tuning table stops being comparable to the
baseline table, and nobody notices until the night before.

Three strategies:

  random         Each rating assigned to test with probability .20, independent
                 of time. The headline split, because it is what the assignment
                 and most published MovieLens results use.
  user_chrono    Each user's latest 20% of ratings go to test. Every user stays
                 in training, but the model still sees other users' future
                 ratings of the same movies.
  global_chrono  One cutoff date. Everything after it is test. The only split
                 where the model never sees the future, and the only one that
                 resembles deployment. It also produces cold-start users, which
                 is the point.

Only the random split is used to select or tune a model. The chronological
splits exist for the evaluation-validity subsection and are scored, never tuned.
"""

import numpy as np
import pandas as pd

from . import config as C


def split_path(strategy):
    return C.RESULTS_DIR / f"split_{strategy}.csv"


def _assign(ratings, strategy):
    n = len(ratings)
    test = pd.Series(False, index=ratings.index)
    if strategy == "random":
        rng = np.random.default_rng(C.SEED)
        test_idx = rng.choice(ratings.index.to_numpy(), size=int(round(n * C.TEST_SIZE)),
                              replace=False)
        test.loc[test_idx] = True
    elif strategy == "user_chrono":
        # Ties on timestamp are broken by file order so the split is fully
        # determined by the data, with no randomness.
        ordered = ratings.assign(_row=np.arange(n)).sort_values([C.USER, C.TIMESTAMP, "_row"])
        pos = ordered.groupby(C.USER).cumcount()
        size = ordered.groupby(C.USER)[C.USER].transform("size")
        n_test = np.ceil(size * C.TEST_SIZE).astype(int)
        test.loc[ordered.index[(pos >= size - n_test).to_numpy()]] = True
    elif strategy == "global_chrono":
        ordered = ratings.assign(_row=np.arange(n)).sort_values([C.TIMESTAMP, "_row"])
        n_test = int(round(n * C.TEST_SIZE))
        test.loc[ordered.index[-n_test:]] = True
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Use one of {C.SPLIT_STRATEGIES}.")
    return test.map({True: "test", False: "train"}).rename("split")


def make_split(ratings, strategy=C.HEADLINE_SPLIT, overwrite=False):
    """Return a train/test assignment Series aligned to ratings.index.

    Cached per strategy under results/. A cached split that does not cover the
    current frame row for row raises instead of silently reindexing to NaN,
    which would push unmatched rows into the test set.
    """
    cache = split_path(strategy)
    if cache.exists() and not overwrite:
        cached = pd.read_csv(cache, index_col=0)["split"]
        if not ratings.index.equals(cached.index):
            raise ValueError(
                f"{cache.name} does not match the loaded ratings "
                f"(cached n={len(cached)}, current n={len(ratings)}). The data or the "
                "cleaning changed. Resolve that; do not pass overwrite=True to hide it."
            )
        return cached
    assignment = _assign(ratings, strategy)
    assignment.to_frame().to_csv(cache)
    return assignment


def train_test(ratings, assignment):
    """Split the ratings frame. Refuses an assignment with gaps."""
    if assignment.isna().any() or not ratings.index.equals(assignment.index):
        raise ValueError("Split assignment and ratings frame disagree. Fix before fitting.")
    return ratings[assignment == "train"], ratings[assignment == "test"]


def describe_split(ratings, assignment, strategy):
    """Counts the evaluation-validity subsection needs, including cold start."""
    train, test = train_test(ratings, assignment)
    train_users, train_items = set(train[C.USER]), set(train[C.ITEM])
    cold_user = ~test[C.USER].isin(train_users)
    cold_item = ~test[C.ITEM].isin(train_items)
    return {
        "strategy": strategy,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_share": float(len(test) / len(ratings)),
        "train_users": len(train_users),
        "train_items": len(train_items),
        "test_ratings_cold_user": int(cold_user.sum()),
        "test_ratings_cold_item": int(cold_item.sum()),
        "test_ratings_warm": int((~cold_user & ~cold_item).sum()),
        "test_users_unseen_in_train": int(test.loc[cold_user, C.USER].nunique()),
        "train_end_utc": train["datetime"].max().isoformat(),
        "test_start_utc": test["datetime"].min().isoformat(),
        **_future_information(train, test),
    }


def _future_information(train, test):
    """How much the training set knows about the future of each test rating.

    For each test rating (u, i, t), count training ratings of the same item i
    made after time t. A model fit on those ratings has seen how the movie was
    received after the moment it is being asked to predict. This is the leakage
    the evaluation-validity subsection measures. It is zero by construction for
    global_chrono and large for random.
    """
    later_share = np.zeros(len(test))
    any_later = np.zeros(len(test), dtype=bool)
    train_ts = {i: np.sort(g.to_numpy()) for i, g in train.groupby(C.ITEM)[C.TIMESTAMP]}
    for k, (i, t) in enumerate(zip(test[C.ITEM].to_numpy(), test[C.TIMESTAMP].to_numpy())):
        ts = train_ts.get(i)
        if ts is None or len(ts) == 0:
            continue
        n_later = len(ts) - np.searchsorted(ts, t, side="right")
        later_share[k] = n_later / len(ts)
        any_later[k] = n_later > 0
    return {
        "test_ratings_with_future_train_ratings_of_same_item": float(any_later.mean()),
        "mean_share_of_item_train_ratings_from_future": float(later_share.mean()),
    }
