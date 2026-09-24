"""Download, verify, and load MovieLens 100K. One cleaning path for everyone.

Nobody calls pd.read_csv on the raw files anywhere else. Import load_data()
and work from what it returns, so five people report the same numbers.

    from src.data import load_data
    ratings, users, items, log = load_data()
"""

import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from . import config as C


def _download(url, dest):
    """Download to a .part file and rename, so a failed transfer never leaves a
    truncated zip that blocks every later run.

    Uses the operating system's certificate verifier through truststore when it
    is installed. Python's default TLS stack cannot fetch a missing intermediate
    certificate, which the OS verifier (and every browser) can.
    """
    ctx = None
    try:
        import ssl

        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.replace(dest)


def ensure_downloaded():
    """Fetch and extract ml-100k into data/ if it is not already there.

    If the download fails, save the zip from a browser as data/ml-100k.zip and
    rerun. Never work around a certificate error by disabling verification.
    """
    if C.RATINGS_FILE.exists() and C.USERS_FILE.exists() and C.ITEMS_FILE.exists():
        return
    archive = C.DATA_DIR / "ml-100k.zip"
    manual = (f"Download {C.ML100K_URL} in a browser, save it as {archive}, and rerun. "
              "Do not disable certificate verification to get around this.")
    if not archive.exists():
        print(f"Downloading {C.ML100K_URL}", flush=True)
        try:
            _download(C.ML100K_URL, archive)
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Could not download MovieLens 100K: {e}. {manual}") from e
    try:
        with zipfile.ZipFile(archive) as z:
            z.extractall(C.DATA_DIR)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"{archive} is not a valid zip. Delete it. {manual}") from e
    missing = [p.name for p in (C.RATINGS_FILE, C.USERS_FILE, C.ITEMS_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Archive extracted but {missing} not found under {C.ML100K_DIR}.")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_raw():
    ratings = pd.read_csv(
        C.RATINGS_FILE, sep="\t", header=None,
        names=[C.USER, C.ITEM, C.RATING, C.TIMESTAMP],
        dtype={C.USER: "int64", C.ITEM: "int64", C.RATING: "float64", C.TIMESTAMP: "int64"},
    )
    users = pd.read_csv(
        C.USERS_FILE, sep="|", header=None,
        names=[C.USER, "age", "gender", "occupation", "zip"],
        dtype={C.USER: "int64", "age": "int64", "gender": "string",
               "occupation": "string", "zip": "string"},
    )
    items = pd.read_csv(
        C.ITEMS_FILE, sep="|", header=None, encoding="latin-1",
        names=[C.ITEM, "title", "release_date", "video_release_date", "imdb_url"] + C.GENRES,
    )
    return ratings, users, items


def load_data(write_log=True):
    """Return (ratings, users, items, log). Raises if the data is not ml-100k."""
    ensure_downloaded()
    ratings, users, items = _read_raw()
    sha = _sha256(C.RATINGS_FILE)
    log = {"ratings_sha256": sha, "source": C.ML100K_URL}

    # --- Identity checks. Fail loudly rather than model the wrong file. ---
    problems = []
    if len(ratings) != C.EXPECTED_RATINGS:
        problems.append(f"{len(ratings)} ratings, expected {C.EXPECTED_RATINGS}")
    if ratings[C.USER].nunique() != C.EXPECTED_USERS:
        problems.append(f"{ratings[C.USER].nunique()} users, expected {C.EXPECTED_USERS}")
    if ratings[C.ITEM].nunique() != C.EXPECTED_ITEMS:
        problems.append(f"{ratings[C.ITEM].nunique()} items, expected {C.EXPECTED_ITEMS}")
    if C.EXPECTED_RATINGS_SHA256 and sha != C.EXPECTED_RATINGS_SHA256:
        problems.append(f"u.data SHA-256 {sha[:12]}... does not match config")
    if problems:
        raise ValueError("This is not the expected MovieLens 100K file: " + "; ".join(problems))
    log["fingerprint_status"] = "verified" if C.EXPECTED_RATINGS_SHA256 else "recorded_only"

    # --- Cleaning screen. Each check is reported whether or not it fires. ---
    lo, hi = C.RATING_SCALE
    per_user = ratings.groupby(C.USER).size()
    dup_pairs = ratings.duplicated([C.USER, C.ITEM]).sum()
    title_counts = items["title"].value_counts()
    dup_titles = title_counts[title_counts > 1]
    log["cleaning"] = {
        "rows_in": int(len(ratings)),
        "missing_values_ratings": int(ratings.isna().sum().sum()),
        "missing_values_users": int(users.isna().sum().sum()),
        "duplicate_user_item_pairs": int(dup_pairs),
        "ratings_outside_scale": int((~ratings[C.RATING].between(lo, hi)).sum()),
        "non_integer_ratings": int((ratings[C.RATING] % 1 != 0).sum()),
        "users_below_min_ratings": int((per_user < C.MIN_RATINGS_PER_USER).sum()),
        "ratings_referencing_unknown_users": int((~ratings[C.USER].isin(users[C.USER])).sum()),
        "ratings_referencing_unknown_items": int((~ratings[C.ITEM].isin(items[C.ITEM])).sum()),
        # Known quirk: some titles appear under two item ids. Kept separate on
        # purpose (see README, "Data decisions"), because merging them changes
        # the published benchmark and each id carries its own rating history.
        "duplicate_titles": int(len(dup_titles)),
        "item_ids_with_duplicate_titles": int(dup_titles.sum()),
        "duplicate_title_examples": dup_titles.index[:5].tolist(),
        "items_with_unknown_genre_flag": int(items["unknown"].sum()),
        "items_missing_release_date": int(items["release_date"].isna().sum()),
        "non_numeric_zip_codes": int((~users["zip"].str.fullmatch(r"\d{5}")).sum()),
    }

    # Duplicate pairs would mean the same user rated the same item twice. The
    # published file has none; if one ever appears, keep the latest rating,
    # which is the user's current opinion, and record how many were dropped.
    if dup_pairs:
        ratings = (ratings.sort_values(C.TIMESTAMP)
                   .drop_duplicates([C.USER, C.ITEM], keep="last")
                   .sort_index())
    log["cleaning"]["rows_out"] = int(len(ratings))

    # --- Derived columns used downstream. ---
    ratings["datetime"] = pd.to_datetime(ratings[C.TIMESTAMP], unit="s", utc=True)
    users["age_band"] = pd.cut(users["age"], bins=C.AGE_BINS, labels=C.AGE_LABELS, right=False)
    items["n_genres"] = items[C.GENRES].sum(axis=1)

    n_u, n_i = ratings[C.USER].nunique(), ratings[C.ITEM].nunique()
    log["shape"] = {
        "n_ratings": int(len(ratings)),
        "n_users": int(n_u),
        "n_items": int(n_i),
        "density": float(len(ratings) / (n_u * n_i)),
        "sparsity": float(1 - len(ratings) / (n_u * n_i)),
        "first_rating_utc": ratings["datetime"].min().isoformat(),
        "last_rating_utc": ratings["datetime"].max().isoformat(),
        "mean_rating": float(ratings[C.RATING].mean()),
        "sd_rating": float(ratings[C.RATING].std(ddof=1)),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_item_median": float(ratings.groupby(C.ITEM).size().median()),
        "items_with_fewer_than_5_ratings": int((ratings.groupby(C.ITEM).size() < 5).sum()),
    }

    if write_log:
        (C.RESULTS_DIR / "data_log.json").write_text(json.dumps(log, indent=2, default=_json_default))
    return ratings, users, items, log


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


if __name__ == "__main__":
    r, u, i, lg = load_data()
    print(json.dumps(lg, indent=2, default=_json_default))
