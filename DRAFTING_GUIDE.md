# Part 2 Drafting Guide

This guide explains the code and the results, then covers what each section of the report needs. Read sections 1 through 4 whatever your section is, because the findings connect across the whole paper. Then read your own part of section 5.

Every number here comes from `results/KEY_NUMBERS.md`. When you write, copy numbers from that file, not from this guide or from memory.

---

## 1. What the code does

You don't need to run anything. The results are already in `results/` and the figures in `figures/`. If you're curious, `python run_all.py` reproduces everything in about two minutes (see the README for setup).

`run_all.py` runs these steps in order. Each is a module in `src/`.

1. **Load and verify the data** (`data.py`). Reads MovieLens 100K's three files: ratings, user demographics, and movie details. Refuses to continue unless the counts and a SHA-256 fingerprint match the published dataset, and records a cleaning screen.
2. **Freeze the train/test splits** (`splits.py`). One random 80/20 split (80,000 training ratings, 20,000 test ratings, seed 42) drives every result. Two chronological splits exist only for the evaluation-validity comparison. All three are saved to disk so no one can regenerate them differently.
3. **Explore the data** (`eda.py`). Rating distribution, sparsity, how ratings concentrate on a few users and movies, and demographics.
4. **Compare four models** (`models.py`): a random predictor, a bias-only baseline, item-based nearest neighbors (KNN), and SVD, all on the same split.
5. **Tune SVD** (`tuning.py`). Three GridSearchCV searches on the training ratings only.
6. **Evaluate the final model** (`evaluation.py`). Scores the tuned SVD once on the test set, breaks down where its errors are, measures ranking quality, and builds a top-10 list for every user.
7. **Check fairness** (`fairness.py`). Error by gender, age, and occupation, and which movies the lists push to users.
8. **Test a mitigation** (`mitigation.py`). A minimum-ratings threshold for recommending a movie, swept across five values.
9. **Test evaluation validity** (`temporal.py`). Reruns the tuned model on the chronological splits.
10. **Write `KEY_NUMBERS.md`** (`key_numbers.py`). Collects every reportable number, by section, with its source file.

---

## 2. Concepts you'll write about

**Collaborative filtering.** Predicting how a user will rate a movie from the pattern of ratings across all users, not from the movie's content. The data is a user-by-movie matrix. Here it is 93.7% empty, and the model's job is to fill in the gaps.

**SVD (as Surprise implements it).** Each prediction is the global average rating, plus a *user bias* (does this person rate high or low?), plus a *movie bias* (is this movie rated above or below average?), plus a match score between the user's and the movie's *latent factors* (learned taste dimensions). The four hyperparameters:

- `n_factors`: how many taste dimensions.
- `n_epochs`: how many passes over the training data.
- `lr_all`: the learning rate, or how big each update step is.
- `reg_all`: regularization, a penalty that keeps the learned values small so the model doesn't overfit.

**RMSE and MAE.** Both measure prediction error in rating points on the 1 to 5 scale. RMSE squares errors before averaging, so it punishes large misses more. Lower is better.

**Cross-validation versus the test set.** Tuning never sees the 20,000 test ratings. GridSearchCV splits the 80,000 training ratings into 3 folds and scores each candidate on those. The chosen model is then scored once on the test set. "CV RMSE" and "test RMSE" are different numbers; always say which one you mean.

**Precision@10 and recall@10.** Ranking metrics. For each user, the model's ten highest-predicted movies among that user's test ratings are taken as its recommendations. A movie counts as "relevant" if the user actually rated it 4 or higher. Precision@10 is the share of recommendations that were relevant. Recall@10 is the share of the user's relevant movies that got recommended. RMSE grades rating predictions; these grade the lists a user would actually see.

**Thin-evidence movie.** Our term for a movie with fewer than 10 ratings in the training data. Use it consistently.

---

## 3. The results, as one story

**The model choice holds up.** Untuned SVD beats every alternative on the test set: RMSE 0.941, against 0.950 for the bias-only baseline, 0.981 for item KNN, and 1.521 for random guessing.

**Tuning worked by the assignment's measure.** On identical cross-validation folds, the best configuration cut CV RMSE from 0.9534 (Surprise's defaults) to 0.9294. On the held-out test set, RMSE fell from 0.941 to 0.921. That reduction (0.0198) is nearly five times the fold-to-fold noise (0.0041). Regularization mattered four times more than any other hyperparameter.

**But the tuned model is a worse recommender.** Precision@10 fell from .619 to .596. The untuned model sends 0.1% of its recommendations to thin-evidence movies; the tuned model sends 56.5%. *Pather Panchali* has 7 training ratings and appears in 97% of users' top-10 lists. *The Saint of Fort Washington* has 2 and appears in 87%. Lists also became less personal: the overlap between two random users' lists doubled from .177 to .360.

**The likely mechanism.** In Surprise's SVD, regularization is applied one rating at a time, so it shrinks a movie with 2 ratings about as much as a movie with 500. It doesn't pull small samples toward the average harder, the way the bias-only baseline does. The untuned model trains briefly (20 passes at a small step size), so thin movies' biases never grow far from zero. Tuning chose 80 passes at a larger step, which let those biases climb to their raw averages of 4.6 to 5.0, the highest in the catalog. The pattern across configurations fits: 0.1% thin recommendations at 20 passes, 25.5% at 40, and 56.5% at 80. Call this the *likely* mechanism; it wasn't isolated experimentally.

**A simple filter fixes the symptom but not the cause.** Refusing to recommend movies with fewer than 10 training ratings drops thin-evidence recommendations to 0% with no change in precision (.596) and a recall loss of .002. It does not restore personalization (list overlap .318, versus .177 untuned).

**The model is less accurate for women.** RMSE is 0.977 for women and 0.902 for men, and the 95% confidence interval on the gap excludes zero. The gap holds in every activity quartile, so it isn't because women rated fewer movies. The likely cause is that women contribute only about 26% of the training ratings.

**A random split flatters the model.** RMSE is 0.921 on the random split, 0.972 when each user's latest ratings are held out, and 1.021 with a single cutoff date. The random split lets the model train on ratings made after the ones it is tested on.

---

## 4. Rules for every section

- **Numbers.** Copy from `KEY_NUMBERS.md` exactly. In prose, round RMSE and MAE to three decimals (0.921).
- **APA number format.** RMSE and MAE can exceed 1, so they keep the leading zero (0.921). Precision, recall, proportions, and correlations cannot exceed 1, so they drop it (.596).
- **Consistent terms.** "Tuned SVD" and "untuned SVD" (not "default model," "baseline SVD," or "final model" in one place and "optimized model" in another). "Test set" for the held-out 20%. "Training partition" for the 80%. "Thin-evidence movies" as defined above. "Bias-only baseline" for BaselineOnly.
- **Figures.** At most one per section; the page budget is tight. Label it **Figure X** in bold with the title in italics on the next line, and put a *Note.* under it explaining what it shows. Part 1 lost small points on figure and table notes. Leave the number as X; Eric numbers them at assembly.
- **Sources.** Bring at least one peer-reviewed source, and read it before citing it. Section 7 lists candidates. Add yours to the References page in alphabetical order with a hanging indent.
- **Claims.** Match the evidence. "Likely mechanism" stays likely, and a measured gap is not a proven cause. Hedge only where the evidence actually leaves room.
- **Length.** Hyperparameter Tuning 0.5 page, Final Model Evaluation 1 page, Ethical Considerations 0.5 page, Real-World Application 1 page, Conclusion 0.5 page. Keep the setup sections short too.

---

## 5. Section by section

### Div: Dataset and Preprocessing

**Cover:** MovieLens 100K (Harper & Konstan, 2015):

- 100,000 ratings on a 1 to 5 integer scale from 943 users on 1,682 movies, September 1997 to April 1998. Every user has at least 20 ratings.
- The three files used: ratings with timestamps, user demographics, and movie details.
- The cleaning screen found 0 missing values, 0 duplicate user-movie pairs, and 0 out-of-range ratings, so no rows were removed.
- 18 titles appear under two movie IDs each. They were kept separate because merging them would change the published benchmark and each ID has its own rating history.
- Every run verifies the data against a pinned fingerprint.
- The random 80/20 split (seed 42; 80,000 and 20,000 ratings) is frozen. The two chronological splits exist only for the evaluation-validity comparison.

**Watch out:** A clean dataset is a result, not an empty section. Say what was checked and that it passed. Don't describe cleaning steps that never happened.

**Source:** Harper and Konstan (2015) is already in the References.

### Div: Model Selection (one short paragraph)

**Cover:** The four-model comparison on the test set (RMSE / MAE):

| Model | Test RMSE | Test MAE |
|---|---|---|
| Random predictor | 1.521 | 1.224 |
| Item KNN | 0.981 | 0.774 |
| Bias-only baseline | 0.950 | 0.752 |
| SVD | 0.941 | 0.741 |

The comparison that matters is SVD against the bias-only baseline, because the gap between them is what the latent factors add: 0.009 untuned, growing to 0.029 after tuning. Item KNN ranked worst (precision@10 .371).

**Watch out:** The CV column in the KEY_NUMBERS table is 5-fold. Tuning uses 3-fold. Never put numbers from the two side by side.

**Source:** Sarwar et al. (2001) for item-based KNN.

### Zubair: Exploratory Data Analysis

**Cover:**

- **Ratings skew positive:** 4 is the most common rating (34.2%) and 1 the least (6.1%); mean 3.53, SD 1.13.
- **Sparsity:** 93.7% of the user-movie matrix is empty.
- **Activity is concentrated.** Users: median 65 ratings, and the top 10% of users supply 32.2% of all ratings. Movies: median 27 ratings, the top 20% receive 64.7% of ratings, 141 have exactly one rating, and 530 have fewer than 10.
- **Demographics:** 71% of users are male, and they supply 74.3% of ratings.

**The most important point for the rest of the paper:** averages from few ratings are unreliable. The SD of movie averages is 0.96 for movies with fewer than 10 ratings and 0.42 for movies with 100 or more. Ten movies average a perfect 5.0 on fewer than 5 ratings. This sets up the thin-evidence finding that runs through Sections 2 and 3.

**Figure:** `eda_item_mean_vs_count.png` (the funnel shape shows the point at a glance). Use `eda_long_tail.png` if you want the concentration instead. Skip genres and weekly activity unless there's room.

**Watch out:** EDA earns its space by setting up later sections, not by cataloging everything in the data.

**Source:** Park and Tuzhilin (2008) on the long tail in recommender systems.

### Zubair: Real-World Application (1 page)

**Cover:**

- **A concrete deployment:** a "recommended for you" row on a streaming or rental service's home screen.
- **What happens if the tuned model ships as-is:** 97% of users see the same obscure 1955 film, based on 7 ratings. The example lists in `results/final_example_recommendations.csv` make this vivid. User 651 has 11 training ratings, and their top five are *Pather Panchali*, *The Saint of Fort Washington*, *The World of Apu*, *The Wrong Trousers*, and *Paths of Glory*.
- **What production would change, each tied to our evidence:**
  - Select models on ranking quality or live A/B tests, not RMSE alone.
  - Regularize movie biases more heavily than taste factors (Surprise exposes `reg_bi`), or use damped averages.
  - Keep a minimum-support floor. At 10 ratings it cost nothing measurable.
  - Handle new users separately. In the single-cutoff-date test, 85% of test ratings came from users with no history.
  - Retrain on a schedule. Untuned SVD fits 80,000 ratings in about a second, and top-10 lists for every user come from one matrix multiplication.

**Watch out:** This section is graded on "realistic and compelling," so every recommendation should point back to a number from our results.

**Source:** Gomez-Uribe and Hunt (2015) on Netflix's production recommender.

### Albert: Hyperparameter Tuning (0.5 page)

**Cover:**

- **Method:** GridSearchCV with 3-fold CV on the training partition only. **State in one sentence** that this departs from the assignment's example, which fits the search on all the data. Fitting on all the data lets the test ratings choose the hyperparameters and then grade them.
- **Three grids, all on the same folds:**
  - **Assignment grid,** exactly as written: 8 candidates, best CV RMSE 0.9502, against 0.9534 for Surprise's defaults.
  - **Expanded grid,** adding `reg_all`: 54 candidates, best 0.9306. The winner sat on the upper edge of all four axes, so the true optimum could lie outside the grid.
  - **Refined grid,** extending past that edge: 81 candidates, best 0.9294 at 200 factors, 80 epochs, learning rate .01, and regularization .15.
- **Regularization dominates.** Across the expanded grid it moved CV RMSE by 0.0235, against 0.0059 for the next parameter.
- **The refined grid reached a plateau.** The top 10 candidates lie within 0.0008 of each other, and the parameters still on an edge move RMSE by 0.0011 or less, so further search would not change the result.

**Figure:** `tuning_sensitivity_expanded.png`.

**Watch out:** Compare tuned and untuned only on the same folds (0.9534 to 0.9294) or on the test set. Not against the 5-fold numbers in the Model Selection table.

**Source:** Bergstra and Bengio (2012) on grid versus random search.

### Albert: Final Model Evaluation (1 page)

**Cover:**

- **Headline:** Tuned SVD test RMSE 0.921 and MAE 0.729, against 0.941 and 0.741 untuned. The 0.0198 reduction is nearly five times the fold noise of 0.0041. Tuned SVD beats the bias-only baseline by 0.029. Training RMSE is 0.791, so the train-to-test gap is 0.131.
- **Compression toward the mean:** predictions have an SD of 0.58 against 1.13 for true ratings. True 1s are predicted at 2.77 on average and true 5s at 3.97. Figure: `final_predicted_by_true_rating.png`.
- **Where errors concentrate:** RMSE is 0.965 for the lightest fifth of users and about 0.90 for everyone else, and 0.982 for the least-rated fifth of movies.
- **The ranking problem:** precision@10 fell from .619 to .596 after tuning, and recall@10 from .282 to .261. Lower RMSE did not produce better lists. Cremonesi et al. (2010) found the same disconnect between rating accuracy and top-N quality across algorithms; ours shows it within one. The explanation belongs to Ethical Considerations, so give it one sentence here and point there.
- **Evaluation validity** (short subsection): random split 0.921; per-user chronological 0.972 (each user's latest 20% held out), which is the fair comparison, about 5% worse; single cutoff date 1.021. On the random split, 98% of test ratings are for movies whose later reception the model already saw in training. **Caveat:** the single-cutoff split is 85% brand-new users (RMSE 1.028 for them vs 0.979 for returning users), so it mostly measures cold start, not leakage.

**Watch out:** Do not swap in a different configuration because its test score looks better. An earlier run's configuration had test RMSE 0.917, but every choice was made on training folds, and picking by test score would make the test set part of the tuning.

**Source:** Cremonesi et al. (2010) is already in the References. Add Campos et al. (2014) for chronological evaluation protocols.

### Bharadwaj: Ethical Considerations (0.5 page)

**Cover:**

- **Demographic bias.** RMSE is 0.977 for women (95% CI [0.943, 1.015], 273 users) and 0.902 for men ([0.882, 0.921], 669 users). The gap, 0.075 with CI [0.036, 0.118], excludes zero. It holds within every activity quartile (+0.086, +0.125, +0.033, +0.040), so it isn't because women rated fewer movies. The likely cause is that women contribute about 26% of training ratings, so the learned taste factors fit the majority. Ekstrand et al. (2018) found demographic accuracy gaps like this in MovieLens data.
- **Thin-evidence exposure.** 56.5% of the tuned model's recommendations go to movies with fewer than 10 training ratings. Two films with 7 and 2 ratings reach 97% and 87% of users. Users can't see that a confident-looking recommendation rests on two ratings, which is a transparency problem as much as an accuracy one. Use the mechanism from Section 3 here.
- **Mitigation, with its limits.** A 10-rating minimum removes thin-evidence recommendations at no measurable cost (precision stays .596; recall drops .002) but leaves lists less personal than the untuned model's (overlap .318 vs .177). A 5-rating minimum backfires: *Pather Panchali* survives it and reaches 98% of lists. The threshold was justified by the EDA, not chosen by test performance.
- **Privacy.** The dataset holds age, gender, occupation, and zip code alongside full rating histories. Narayanan and Shmatikov (2008) showed a handful of ratings can re-identify people in an anonymized ratings dataset. Our repo does not redistribute the data.

**Watch out:**

- Don't call the thin-evidence problem "popularity bias." The most-rated 10% of movies get 22.3% of recommendations but 42.4% of training ratings, so the model actually *under*-exposes popular movies. The problem is weak evidence, not popularity.
- Don't claim occupation gaps; several groups have under 10 users. Note that the heaviest-activity quartile has only 10 women.

**Figures:** `fairness_rmse_by_group.png` or `mitigation_min_support.png`. Pick one.

**Sources:** Ekstrand et al. (2018) and Narayanan and Shmatikov (2008).

### Eric: Introduction, Team Collaboration, Conclusion, and Assembly

The three sections are drafted in the shared document. Assembly checklist:

- Number figures and tables in order.
- Check every figure and table note.
- Merge References alphabetically.
- Check every number in the draft against `KEY_NUMBERS.md`.
- Delete every highlighted placeholder.
- Confirm the collaboration roles and the Albert/Alberto spelling on the title page.

---

## 6. Mistakes that would cost points

- Comparing 5-fold and 3-fold CV numbers.
- Calling the thin-evidence problem popularity bias.
- Reporting a small RMSE difference as an improvement without comparing it to the fold noise.
- Stating the gender gap's cause or the SVD mechanism as proven rather than likely.
- Treating the single-cutoff-date RMSE as a pure leakage measure. It's mostly cold start.
- Citing a paper you haven't read.

---

## 7. Candidate sources

All of these are peer-reviewed. Each was confirmed to exist at the DOI or URL shown. Read the paper before you cite it, and confirm it supports your sentence.

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://jmlr.org/papers/v13/bergstra12a.html

Campos, P. G., Díez, F., & Cantador, I. (2014). Time-aware recommender systems: A comprehensive survey and analysis of existing evaluation protocols. *User Modeling and User-Adapted Interaction, 24*(1–2), 67–119. https://doi.org/10.1007/s11257-012-9136-x

Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of recommender algorithms on top-N recommendation tasks. In *Proceedings of the Fourth ACM Conference on Recommender Systems* (pp. 39–46). Association for Computing Machinery. https://doi.org/10.1145/1864708.1864721 *(already in References)*

Ekstrand, M. D., Tian, M., Azpiazu, I. M., Ekstrand, J. D., Anuyah, O., McNeill, D., & Pera, M. S. (2018). All the cool kids, how do they fit in?: Popularity and demographic biases in recommender evaluation and effectiveness. *Proceedings of Machine Learning Research, 81*, 172–186. https://proceedings.mlr.press/v81/ekstrand18b.html

Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix recommender system: Algorithms, business value, and innovation. *ACM Transactions on Management Information Systems, 6*(4), Article 13. https://doi.org/10.1145/2843948 *(check the year against the ACM page; the issue is dated both 2015 and January 2016 in different indexes)*

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), Article 19. https://doi.org/10.1145/2827872 *(already in References)*

Narayanan, A., & Shmatikov, V. (2008). Robust de-anonymization of large sparse datasets. In *2008 IEEE Symposium on Security and Privacy* (pp. 111–125). IEEE. https://doi.org/10.1109/SP.2008.33

Park, Y.-J., & Tuzhilin, A. (2008). The long tail of recommender systems and how to leverage it. In *Proceedings of the 2008 ACM Conference on Recommender Systems* (pp. 11–18). Association for Computing Machinery. https://doi.org/10.1145/1454008.1454012

Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. In *Proceedings of the 10th International Conference on World Wide Web* (pp. 285–295). Association for Computing Machinery. https://doi.org/10.1145/371920.372071
