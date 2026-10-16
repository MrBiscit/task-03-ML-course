# Model card: term-deposit call ranking

## Purpose
Rank the clients of a Portuguese bank's call list by how likely they are to say **yes** to a term deposit, so that the 10 % of clients the call centre can phone in one wave contain as many yes-clients as possible. The score is made before any call. Teaching dataset (UCI Bank Marketing, Moro, Rita & Cortez 2014, CC BY 4.0); not for conclusions about real banks or clients.

## Data
`train.csv`: 32 940 rows, 11.27 % yes. `test.csv`: 8 236 rows, no answer. `test.csv` lacks `duration` (call length, known only after the call, so using it would be leakage) and `y`. The model uses the 19 columns of test except `id`. Missing values are the text `unknown` (`default` 20.9 %, `education` 4.3 %, `housing`/`loan` 2.4 %, `job` 0.8 %, `marital` 0.2 %); `pdays = 999` means "never contacted" for 96.2 % of rows. Rows are in random order (no time information in the file).

## Pipeline (`model.joblib`, scikit-learn only)
`ColumnTransformer` -> `RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=SEED)`.
- Numeric columns: `StandardScaler`.
- `pdays`: 999 replaced by the median of the real values (+ 0/1 "never contacted" flag), then scaled.
- Categorical columns: one-hot, `unknown` kept as its own level, unseen levels ignored.
- Everything is fitted inside the pipeline on the training part of each fold only (checked: the scaler mean equals the fit-rows mean). The saved model accepts the raw test columns directly. Final fit on all of `train.csv`.

## Validation and results
Own 20 % stratified hold-out (touched once); 5-fold stratified CV on the rest, same folds for all models. SEED = 2666.

| Model (5-fold CV, mean ± std) | ROC-AUC | Avg. precision | Precision in top 10 % |
|---|---|---|---|
| Majority class | 0.500 ± 0.000 | 0.113 | 0.114 (accuracy 0.887) |
| Logistic regression | 0.793 ± 0.017 | 0.452 ± 0.024 | 0.498 ± 0.016 |
| Decision tree (tuned) | 0.792 ± 0.015 | 0.427 ± 0.026 | 0.490 ± 0.007 |
| Random forest (tuned) | 0.800 ± 0.016 | 0.465 ± 0.026 | 0.508 ± 0.025 |

Honest reading: logistic regression, the tree and the tuned logistic are indistinguishable. The forest is ahead on ROC-AUC by 0.007 in all 5 folds, but that is small next to the fold spread (0.016), and its gains in average precision and precision at 10 % are inside the noise. `class_weight="balanced"` made no difference beyond noise and was not used. Preprocessing variants (`pdays` handling, `unknown` as level vs most frequent) are also inside the noise; they were chosen for stated reasons.

**Hold-out (one evaluation):** ROC-AUC 0.800, average precision 0.483 (base rate 0.113). With the top 10 % called (659 of 6 588, cutoff score 0.324): precision 0.536 (353 yes), recall 0.476 (of 742 yes-clients), lift 4.8 over random calling. 306 calls reach clients who say no; 389 yes-clients are not called. Standard error of the precision about 0.019.

## Limits and ways it can fail
- **Optimistic split.** The split is random, the real use is forward in time. Holding out whole periods instead dropped CV ROC-AUC from about 0.80 to about 0.71-0.73 and precision at 10 % to about 0.36-0.40. Treat the hold-out figures as an upper bound for a new campaign.
- **Macro columns act as date stamps** (`euribor3m`, `nr.employed`, `emp.var.rate`). In a new economic situation their values may lie outside anything seen in training, and a forest cannot extrapolate; the ranking can degrade without warning.
- **Dependence on `poutcome` / previous contact.** If the share of previously contacted clients changes, the score distribution and the 10 % cutoff change.
- **Data from 2008-2010 and one bank**; client behaviour, channels and rates have changed. No client id: repeated clients cannot be excluded from the evaluation.
- Scores are rankings, not calibrated probabilities (forest); do not read 0.3 as "30 % chance".
- Fairness: age, job and marital status are inputs; check that the call list is not systematically skewed before real use.

## What to monitor after launch
Weekly yes-rate and precision in the called 10 % against the 0.5 expected (and its ±0.02 noise); score distribution and share of `unknown` / `pdays = 999` against training; the macro columns' ranges; lift measured on a small random control group that is called regardless of score; retrain on the newest data when precision stays below the noise band.
