"""Model roster, the Surprise adapters, and the evaluation contract.

Every result in the report comes from evaluate(), which writes a JSON record
to results/. Report writers read those files. Nobody transcribes a number from
a Colab cell into a Word document by hand.

The roster brackets SVD from below so its selection can be justified rather
than assumed:

  normal_predictor   random draws from the rating distribution; the floor
  baseline_only      global mean + user bias + item bias, no interactions
  knn_item           item-based neighborhood CF, the classic alternative
  svd                matrix factorization with biases (Funk-style SVD)

If SVD cannot beat baseline_only, the latent factors are adding nothing over
"this user rates high, this movie is popular," and the report has to say so.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

try:
    import surprise
    from surprise import SVD, BaselineOnly, Dataset, KNNBasic, NormalPredictor, Reader, accuracy
    from surprise.model_selection import GridSearchCV, KFold, cross_validate
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scikit-surprise is not installed. In Colab run `!pip install scikit-surprise`, "
        "then restart the runtime if NumPy was upgraded. See README, 'Setup'."
    ) from e


# --- Adapters ----------------------------------------------------------------
def to_dataset(df):
    """A Surprise Dataset from a ratings frame. Column order is user, item, rating."""
    return Dataset.load_from_df(df[[C.USER, C.ITEM, C.RATING]],
                                Reader(rating_scale=C.RATING_SCALE))


def to_trainset(df):
    return to_dataset(df).build_full_trainset()


def to_testset(df):
    """Surprise testsets are lists of (raw user id, raw item id, true rating)."""
    return [(int(u), int(i), float(r)) for u, i, r in
            df[[C.USER, C.ITEM, C.RATING]].itertuples(index=False)]


def predictions_frame(predictions):
    return pd.DataFrame({
        C.USER: [p.uid for p in predictions],
        C.ITEM: [p.iid for p in predictions],
        "actual": [p.r_ui for p in predictions],
        "predicted": [p.est for p in predictions],
        "was_impossible": [bool(p.details.get("was_impossible", False)) for p in predictions],
    })


def cv_splitter(n_splits):
    """Seeded folds. Surprise's default KFold is unseeded, so pass this explicitly."""
    return KFold(n_splits=n_splits, random_state=C.SEED, shuffle=True)


# --- Roster ------------------------------------------------------------------
def make_baseline():
    """Bias-only model. verbose=False silences 'Estimating biases using als...'
    on Surprise versions that accept it; older versions just print."""
    try:
        return BaselineOnly(bsl_options={"method": "als"}, verbose=False)
    except TypeError:
        return BaselineOnly(bsl_options={"method": "als"})


def build_roster():
    return {
        "normal_predictor": NormalPredictor(),
        "baseline_only": make_baseline(),
        "knn_item": KNNBasic(k=40, sim_options={"name": "msd", "user_based": False},
                             verbose=False),
        "svd": SVD(random_state=C.SEED),
    }


def make_svd(params):
    """SVD with the project seed. Every SVD in the project is built here."""
    p = {k: v for k, v in params.items() if k != "random_state"}
    return SVD(random_state=C.SEED, **p)


# --- Evaluation contract ------------------------------------------------------
def evaluate(algo, name, train_df, test_df, split=C.HEADLINE_SPLIT, cv=True, tag=None):
    """Fit on the full training partition, score the test partition once, persist.

    Returns (record, predictions_df). The record also carries k-fold CV RMSE on
    the training partition, which is the number GridSearchCV optimizes and the
    only fair comparison against tuned scores.
    """
    name_out = f"{name}_{tag}" if tag else name
    trainset = to_trainset(train_df)
    t0 = time.perf_counter()
    algo.fit(trainset)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    preds = algo.test(to_testset(test_df))
    test_s = time.perf_counter() - t0
    frame = predictions_frame(preds)

    record = {
        "model": name_out,
        "split": split,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "rmse": accuracy.rmse(preds, verbose=False),
        "mae": accuracy.mae(preds, verbose=False),
        "mean_signed_error": float((frame.predicted - frame.actual).mean()),
        "n_impossible": int(frame.was_impossible.sum()),
        "fit_seconds": fit_s,
        "test_seconds": test_s,
        "seed": C.SEED,
        "surprise_version": getattr(surprise, "__version__", "unknown"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Surprise's SVD stores the per-term rates (lr_bu, reg_bu, ...) rather than
    # lr_all and reg_all, which it expands at construction.
    params = {k: getattr(algo, k) for k in ("n_factors", "n_epochs", "lr_bu", "reg_bu")
              if hasattr(algo, k)}
    if params:
        record["params"] = params

    if cv and name != "normal_predictor":
        res = cross_validate(algo, to_dataset(train_df), measures=["rmse", "mae"],
                             cv=cv_splitter(C.CV_FOLDS), n_jobs=1, verbose=False)
        record["cv_folds"] = C.CV_FOLDS
        record["cv_rmse_mean"] = float(np.mean(res["test_rmse"]))
        record["cv_rmse_sd"] = float(np.std(res["test_rmse"], ddof=1))
        record["cv_mae_mean"] = float(np.mean(res["test_mae"]))
        # cross_validate refits the object on each fold, so refit on the full
        # training partition before anyone uses it downstream.
        algo.fit(trainset)

    out = C.RESULTS_DIR / f"metrics_{split}_{name_out}.json"
    out.write_text(json.dumps(record, indent=2, default=float))
    return record, frame


def run_roster(train_df, test_df, split=C.HEADLINE_SPLIT):
    records, frames, fitted = [], {}, {}
    for name, algo in build_roster().items():
        print(f"  roster: {name}", flush=True)
        rec, frame = evaluate(algo, name, train_df, test_df, split=split)
        records.append(rec)
        frames[name] = frame
        fitted[name] = algo
    table = pd.DataFrame(records)
    table.to_csv(C.RESULTS_DIR / f"roster_comparison_{split}.csv", index=False)
    return table, frames, fitted
