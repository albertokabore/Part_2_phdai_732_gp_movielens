# PhDAI 732 Group 5, Part 2: MovieLens 100K Recommender

Collaborative-filtering recommender built with Surprise's SVD on MovieLens 100K,
tuned with GridSearchCV and evaluated with RMSE and MAE, as the Deliverable 2
assignment specifies. The Part 2 report also carries the Part 1 requirements
(EDA, cleaning, model justification) for a recommender, which Part 1 did not
cover.

**If you are drafting a section, read [`DRAFTING_GUIDE.md`](DRAFTING_GUIDE.md) first.** It explains the code, the results, and what each section needs.

**Take every number from `results/KEY_NUMBERS.md`.** Every
number the report needs is there, with the file it came from. If a number in
the paper is not in that file, it did not come from the pipeline.

## Setup

### Colab (recommended)

Open `notebooks/part2_colab.ipynb` and run the cells in order. The notebook
clones this repo, installs Surprise, runs the pipeline, and shows the results.

### Local

```
pip install -r requirements.txt
python run_all.py
```

Surprise compiles C extensions against NumPy when it installs, and it is the
one fragile dependency here. If `pip install scikit-surprise` fails, or
`import surprise` raises an error mentioning NumPy's ABI or `dtype size
changed`, install a NumPy 1.x first and restart the Python session:

```
pip install "numpy<2" scikit-surprise
```

On Windows, a source build also needs the Microsoft C++ Build Tools. Colab
avoids that entirely. Do not install the package named `surprise` on PyPI; it
is not this library.

## Running

```
python run_all.py            # full pipeline; the expanded grid takes several minutes
python run_all.py --quick    # shrunken grids to check the wiring; never cite these numbers
```

The MovieLens archive is downloaded from GroupLens on the first run and cached
in `data/`. It is never committed: the GroupLens license does not permit
redistribution. If the download fails (a certificate error on some Windows
setups), download https://files.grouplens.org/datasets/movielens/ml-100k.zip in
a browser, save it as `data/ml-100k.zip`, and rerun. `src/config.py` pins the
SHA-256 of `u.data`, so every run on every machine verifies it is reading the
same data, and refuses to continue if not.

## Results in this repo

`results/` and `figures/` are committed from a full run of `run_all.py` on the
frozen splits. `results/run_manifest.json` records when, on which library
versions, and against which data and split hashes. If you change code in `src/`,
rerun the pipeline and commit the regenerated results in the same commit, so
the numbers on GitHub always match the code that produced them.

## Repo contract

Carried over from Part 1, because it is what the Part 1 feedback credited.

1. One seed (`config.SEED = 42`) and frozen splits in `results/split_*.csv`.
   Nobody regenerates a split.
2. Nobody reads the raw files directly. `from src.data import load_data`.
3. Logic lives in `src/`. The notebook is a thin driver.
4. Every result is written to `results/` as JSON or CSV before anyone cites it.
   Nobody transcribes a number from a notebook cell into the paper.
5. The dataset is never committed. `results/` and `figures/` are.

## Layout

| Path | What it does |
|---|---|
| `src/config.py` | Paths, seed, split proportions, hyperparameter grids. The only place any of these are set. |
| `src/data.py` | Downloads, verifies, and loads ratings, users, and movies. Cleaning screen and data log. |
| `src/splits.py` | Random, per-user chronological, and global chronological splits, frozen to disk. |
| `src/eda.py` | Rating distribution, long tails, sparsity, demographics, genres, activity over time. |
| `src/models.py` | Surprise adapters, the model roster, and `evaluate()`, which every result passes through. |
| `src/tuning.py` | GridSearchCV on SVD: the assignment's grid, then an expanded grid. |
| `src/evaluation.py` | Final model on the held-out test set: RMSE, MAE, error anatomy, precision@k, top-10 lists. |
| `src/fairness.py` | Error by gender, age, and occupation with bootstrap CIs; popularity bias in recommendations. |
| `src/temporal.py` | How much the headline RMSE depends on a random split versus a chronological one. |
| `src/mitigation.py` | Minimum-support threshold for recommendations, swept over several values, with its costs and benefits. |
| `src/key_numbers.py` | Writes `results/KEY_NUMBERS.md` from the result files. |
| `run_all.py` | Runs everything in order and writes `results/run_manifest.json`. |

## Where each report section gets its material

| Report section | Pages | Numbers | Figures |
|---|---|---|---|
| Setup: dataset, cleaning, EDA | carries Part 1 | `data_log.json`, `eda_summary.json`, `eda_*.csv` | `eda_*.png` |
| Model selection (setup) | carries Part 1 | `roster_comparison_random.csv` | none |
| 1. Hyperparameter Tuning | 0.5 | `tuning_assignment.json`, `tuning_expanded.json`, `tuning_surface_*.csv` | `tuning_sensitivity_expanded.png` |
| 2. Final Model Evaluation | 1 | `final_evaluation.json`, `final_*.csv`, `temporal_summary.json` | `final_*.png`, `temporal_rmse_by_split.png` |
| 3. Ethical Considerations | 0.5 | `fairness_summary.json`, `fairness_group_errors.csv`, `fairness_gender_by_activity.csv`, `fairness_most_recommended.csv`, `mitigation_min_support.csv` | `fairness_*.png`, `mitigation_min_support.png` |
| 4. Real-World Application | 1 | `final_example_recommendations.csv`, `mitigation_min_support.csv`, `fairness_summary.json` (coverage) | none required |
| 5. Final Thoughts and Conclusion | 0.5 | `KEY_NUMBERS.md` | none |

## Methodological decisions the report should state

**Grid search sees only the training data.** The assignment's example calls
`grid_search.fit(data_surprise)` on the full dataset. That lets the test
ratings choose the hyperparameters and then grade them. Here GridSearchCV runs
on the 80% training partition, and the 20% test partition is scored once,
afterward. One sentence in Section 1 should say so, or a grader comparing
against the snippet will read it as an error.

**Three grids, coarse to fine, all on the same folds.** The assignment's grid
runs exactly as written and is reported as such. The expanded grid adds
`reg_all`, which the assignment's grid leaves at Surprise's default of .02. On
a matrix that is 94% empty, regularization is the parameter most likely to
matter, and on the real data it moved CV RMSE four times more than any other
parameter. The expanded grid's winner sat on the upper edge of all four axes,
which means the optimum could lie outside the range searched, so the refined
grid extends every axis past it. Each grid's record flags any winner that
still sits on an edge, and reports the untuned defaults' score on the same
folds so the improvement is measured like for like.

**Improvement is judged against noise.** `KEY_NUMBERS.md` states whether the
tuned model's RMSE reduction exceeds the default model's fold-to-fold standard
deviation. If it does not, the report calls it within noise, not an
improvement.

**Lower RMSE is not a better recommender, and the report should say so.** On
the real data, each round of tuning raised regularization, which lowered RMSE
and also shrank the personal taste factors. Rankings then fell back on each
movie's bias term, and movies with two to seven perfect ratings entered most
users' top-10 lists. The RMSE-tuned model has lower precision@10 than the
untuned one. `mitigation.py` measures the standard production fix. Do not swap
in a different configuration because its test numbers look better: every
selection here is made on training folds, and choosing by test results would
leak the test set into the model.

**The roster brackets SVD from below.** A random predictor, a bias-only
baseline, and item-based KNN. The bias-only baseline is the important one: the
gap between it and SVD is what the latent factors add.

**Random split for the headline, chronological splits for validity.** Random
80/20 is what the assignment and most published MovieLens results use, so it
drives model selection. The chronological splits are scored, never tuned, and
measure how much a random split flatters the model by letting it train on
ratings made after the ones it is tested on.

**Duplicate titles kept.** MovieLens 100K lists some movies under two ids.
They are kept separate because merging them changes the published benchmark
and each id carries its own rating history. The count is in
`results/data_log.json`.

**Fairness bootstraps resample users, not ratings.** Ratings from one user are
correlated, so resampling individual ratings would make every confidence
interval too narrow.

## Data citation

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and
context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), Article
19. https://doi.org/10.1145/2827872

Hug, N. (2020). Surprise: A Python library for recommender systems. *Journal of
Open Source Software, 5*(52), Article 2174. https://doi.org/10.21105/joss.02174
