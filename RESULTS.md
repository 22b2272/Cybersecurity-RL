# Evaluation results

- **Evaluation date:** 2026-10-25
- **Dataset:** NSL-KDD `KDDTrain+` / `KDDTest+`
- **Test records:** 22,544 (9,711 normal, 12,833 attack)
- **Label mapping:** `normal` = 0; all attack labels = 1
- **Runtime:** Python 3.13; scikit-learn 1.9.1; NumPy 2.5.4

The Random Forest attack threshold was selected on a stratified 20% validation split of the training file (seed 42). The offline contextual DDQN policy trained for 10,000 steps (seed 42) using immediate labeled rewards and no bootstrapping across independent replay records. KDDTest+ was also used for threshold sensitivity analysis in the flow section; all reported KDDTest+ results are descriptive and are not an untouched final estimate for future model selection.

## Held-out comparison

| Metric | Random Forest | DDQN policy |
| --- | ---: | ---: |
| Accuracy | 0.7995 | 0.7793 |
| Macro F1 | 0.7993 | 0.7792 |
| Attack precision | 0.9695 | 0.9222 |
| Attack recall | 0.6689 | 0.6687 |
| Attack F1 | 0.7916 | 0.7752 |
| False positives (normal predicted as attack) | 270 | 724 |
| False negatives (attack predicted as normal) | 4,249 | 4,252 |

On this run, DDQN has slightly fewer false detections but also slightly lower attack recall; its macro F1 trails the supervised baseline. This is a small operating-point tradeoff, not evidence that reinforcement learning is superior for the static labeled-record task.

### Confusion matrices

Rows are true labels and columns are predictions, ordered `[normal, attack]`.

**Random Forest** (validation-selected attack probability threshold: `0.41655`):

```text
[[9441,  270],
 [4249, 8584]]
```

For DDQN, `alert` and `block` count as detected threats:

```text
[[8987,  724],
 [4252, 8581]]
```

### DDQN response actions

| True class | Allow | Alert | Block | Total |
| --- | ---: | ---: | ---: | ---: |
| Normal | 8,987 | 386 | 338 | 9,711 |
| Attack | 4,252 | 243 | 8,338 | 12,833 |
| **Total** | **13,239** | **629** | **8,676** | **22,544** |

The policy blocked 338 normal test records. A live system must treat `block` as a potentially disruptive action and validate it against its own traffic and response controls before enforcement. Mean reward was `0.5633` under the reward table in the README; this value is specific to those chosen rewards and should not be compared across different reward schemes.

## Reproduce

Place the NSL-KDD files under `data/` and run from an environment where the project is installed:

```powershell
ids-train --train data/raw/KDDTrain+.txt --test data/raw/KDDTest+.txt
ids-ddqn --train data/raw/KDDTrain+.txt --test data/raw/KDDTest+.txt --steps 10000 --seed 42
```

Model artifacts and metrics are written under `artifacts/` and are intentionally ignored by Git. The trained artifacts used for this report were local evaluation outputs, not committed files. The supervised IDS metrics are also persisted as `ids_random_forest_metrics.json` for reproducible machine-readable reporting.

## Password classifier

The password pipeline uses 11 features covering character classes, character diversity, Shannon entropy per character, class transitions, repeated runs, adjacent sequences, and common patterns. A reproducible reservoir sample of up to 10,000 examples per class was drawn from PWLDS v1.2 (50,000 rows total). An 80/20 grouped holdout keeps identical feature vectors in one partition. Because large duplicate groups cannot be split, this seed produced 40,803 training and 9,197 held-out rows.

| Metric | Held-out result |
| --- | ---: |
| Accuracy | 0.9338 |
| Macro F1 | 0.9419 |
| Held-out rows | 9,197 |

Confusion matrix, true labels as rows and predicted labels as columns, classes `[0, 1, 2, 3, 4]`:

```text
[[ 894,    1,    0,    0,    0],
 [  14, 1826,   48,   97,    0],
 [   0,   18, 2092,  166,    0],
 [   0,   53,  212, 1821,    0],
 [   0,    0,    0,    0, 1955]]
```

This is a real held-out score on a **synthetic** dataset, not an estimate for real user-chosen passwords. PWLDS is licensed CC BY 4.0 and its documentation notes English-word and synthetic-generation biases. The score measures prediction of its labels from hand-engineered features, not password crackability. The feature export contains no raw password column.

To reproduce, download PWLDS from its [Hugging Face dataset page](https://huggingface.co/datasets/InfinitodeLTD/PWLDS) to `data/raw/pwlds_full.csv`, then run:

```powershell
password-eda data/raw/pwlds_full.csv --sample-per-class 10000 --output data/processed/pwlds_features.csv
password-train data/processed/pwlds_features.csv --model-out artifacts/pwlds_password_model.joblib --metrics-out artifacts/pwlds_password_metrics.json
```

The run produced `artifacts/pwlds_password_metrics.json` locally; artifacts and datasets are ignored by Git.

## Packet-observable IDS model

Packet mode now uses a dedicated supervised model with exactly the nine fields emitted by the packet-summary flow adapter. The model calibrates its alert threshold on a stratified holdout from KDDTrain+, then evaluates on KDDTest+. Two training-validation targets were measured to show the false-positive/recall tradeoff:

| Training validation FPR target | Threshold | KDDTest+ FPR | Attack recall | Accuracy | Macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0% | 0.247716 | 8.88% (862 false positives) | 84.39% | 87.29% | 0.8719 |
| 0.1% | 0.946467 | 2.18% (212 false positives) | 54.13% | 72.95% | 0.7259 |

The stricter target substantially reduced false alarms on KDDTest+, while missing more attacks. Neither validation target guarantees that same FPR on the test distribution; the gap shows a meaningful shift between the training holdout and KDDTest+. The stricter row is not a free improvement: it trades away attack recall.

Reproduce the stricter setting with:

```powershell
ids-flow-train --train 'data/raw/KDDTrain+.txt' --test 'data/raw/KDDTest+.txt' --max-fpr 0.001
```

The evaluation is on NSL-KDD rows projected onto packet-observable fields, not on packet captures. It measures the compact schema on benchmark labels and does not establish live network performance. The held-out benchmark has been used for threshold sensitivity analysis; future threshold tuning needs a new untouched test split or labeled traffic capture.

## Action-dependent simulator

The project now includes a separate synthetic threat-session environment where the response affects future threat persistence. One reproducible run trained for 300 episodes and evaluated on 300 separate simulated episodes (seed 42; evaluation seed 43):

| Policy | Mean reward per episode | Threat detection rate | False blocks per benign step |
| --- | ---: | ---: | ---: |
| DDQN | 31.0833 | 92.51% | 0.64% |
| Always allow | -21.7533 | 0.00% | 0.00% |
| Always alert | -7.1083 | 100.00% | 0.00% |
| Always block | -36.9113 | 100.00% | 100.00% |

Here, detection means choosing `alert` or `block` on a simulated malicious step. The reward reflects the simulator's chosen costs, so this score is only useful for comparing policies inside this exact setup. The result is synthetic and must not be read as an NSL-KDD or live-network metric. The simulator is kept separate from the KDD results above because it uses a different state schema and reward process.

## Interpretation and next work

- The supervised model is a strong comparison point for static labeled records; reinforcement learning is not automatically better for this task.
- The offline DDQN environment replays records and uses labels for immediate response rewards. Its discount is zero so it does not bootstrap across unrelated observations; only the separate synthetic simulator has action-dependent transitions. Realistic costs, calibrated traffic dynamics, and repeated seed runs are still needed before drawing operational conclusions.
- Do not tune further against `KDDTest+`; use training-only validation or a separate test set for future changes.
