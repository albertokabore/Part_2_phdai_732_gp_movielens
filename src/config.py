"""Single source of truth for paths, the seed, split proportions, and grids.

Nothing else in the project hardcodes a seed, a path, a column name, or a
hyperparameter grid. If one of these needs to change, change it here and tell
the group, because every number in the report depends on them.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset -----------------------------------------------------------------
# MovieLens 100K, GroupLens Research (Harper & Konstan, 2015). This is the same
# archive Surprise's Dataset.load_builtin("ml-100k") downloads. We read the raw
# files directly because the report needs three of them (ratings, users,
# items) and load_builtin exposes only the ratings.
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_DIR = DATA_DIR / "ml-100k"
RATINGS_FILE = ML100K_DIR / "u.data"   # user id | item id | rating | timestamp (tab)
USERS_FILE = ML100K_DIR / "u.user"     # user id | age | gender | occupation | zip (pipe)
ITEMS_FILE = ML100K_DIR / "u.item"     # movie id | title | release date | ... | 19 genre flags (pipe, latin-1)

# Published properties of the dataset (README distributed with ml-100k).
# load_data() refuses to continue if the files on disk disagree.
EXPECTED_RATINGS = 100_000
EXPECTED_USERS = 943
EXPECTED_ITEMS = 1_682
MIN_RATINGS_PER_USER = 20
RATING_SCALE = (1, 5)

# SHA-256 of u.data, pinned 2026-09-23 from the GroupLens archive (whose zip
# MD5, 0e33842e24a9c977be4e0107933c0723, matches the published checksum).
# Every run on every teammate's machine verifies it received identical bytes.
# Set to None only to switch to record-only mode.
EXPECTED_RATINGS_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Column names used everywhere downstream.
USER, ITEM, RATING, TIMESTAMP = "user_id", "item_id", "rating", "timestamp"

# --- Reproducibility ---------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5          # cross-validation on the training partition, baseline roster
TUNE_CV_FOLDS = 3     # GridSearchCV folds, as the assignment specifies

# Headline split. The chronological splits exist for the evaluation-validity
# subsection and are never used to select a model.
HEADLINE_SPLIT = "random"
SPLIT_STRATEGIES = ("random", "user_chrono", "global_chrono")

# --- Evaluation --------------------------------------------------------------
# A rating of 4 or 5 counts as "relevant" for precision@k and recall@k. This is
# the convention in Surprise's own FAQ and in most MovieLens ranking work.
RELEVANCE_THRESHOLD = 4.0
TOP_K = 10
BOOTSTRAP_REPS = 1000

# Minimum-support mitigation (mitigation.py): movies with fewer than N training
# ratings are excluded from recommendation lists. Swept rather than fixed, so the
# report shows the tradeoff instead of a single hand-picked value. 0 = no filter.
MIN_SUPPORT_SWEEP = [0, 5, 10, 20, 50]
THIN_EVIDENCE = 10   # "thin evidence" = fewer than this many training ratings

# Age bands for the demographic analysis. MovieLens 100K ages run 7 to 73.
AGE_BINS = [0, 18, 25, 35, 45, 56, 200]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-55", "56+"]

# --- Hyperparameter grids ----------------------------------------------------
# The assignment's grid, verbatim. Run first and reported as specified.
ASSIGNMENT_GRID = {
    "n_factors": [50, 100],
    "n_epochs": [20, 40],
    "lr_all": [0.002, 0.005],
}

# The expanded grid adds the regularization term, which the assignment's grid
# leaves at Surprise's default of .02. On a matrix this sparse, regularization
# is the parameter most likely to matter, so leaving it fixed would search the
# wrong axes.
EXPANDED_GRID = {
    "n_factors": [50, 100, 150],
    "n_epochs": [20, 40],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1],
}

# Coarse-to-fine refinement. On the real data the expanded grid's winner sat on
# the upper edge of all four axes (150 factors, 40 epochs, lr .01, reg .1),
# which means the optimum may lie outside the searched range. This grid extends
# every axis past that edge. It contains the expanded winner, so on the same
# folds its best score can only match or beat it. If the winner lands on an
# edge again, tuning.py flags it and KEY_NUMBERS.md says so.
REFINED_GRID = {
    "n_factors": [100, 150, 200],
    "n_epochs": [40, 60, 80],
    "lr_all": [0.01, 0.015, 0.02],
    "reg_all": [0.1, 0.15, 0.2],
}

# Surprise's SVD defaults, used to locate the untuned configuration inside each
# search so tuned and untuned are compared on identical folds.
SVD_DEFAULTS = {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}

# Shrunken grids for a smoke test. Never report numbers from a --quick run.
QUICK_GRID = {"n_factors": [20, 50], "n_epochs": [10], "lr_all": [0.005]}
