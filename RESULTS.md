# Evaluation results

- **Evaluation date:** 2026-10-25
- **Dataset:** NSL-KDD `KDDTrain+` / `KDDTest+`
- **Test records:** 22,544 (9,711 normal, 12,833 attack)
- **Label mapping:** `normal` = 0; all attack labels = 1
- **Runtime:** Python 3.13; scikit-learn 1.9.1; NumPy 2.5.4

Both models used the official train/test files. The test set was reserved for final evaluation. The Random Forest attack threshold was selected on a stratified 20% validation split of the training file (seed 42). DDQN trained for 10,000 steps (seed 42) in the repository's offline replay environment.

## Held-out comparison

| Metric | Random Forest | DDQN policy |
| --- | ---: | ---: |
| Accuracy | 0.7995 | 0.7985 |
| Macro F1 | 0.7993 | 0.7986 |
| Attack precision | 0.9695 | 0.9218 |
| Attack recall | 0.6689 | 0.7060 |
| Attack F1 | 0.7916 | 0.7998 |
| False positives (normal predicted as attack) | 270 | 769 |
| False negatives (attack predicted as normal) | 4,249 | 3,771 |

The DDQN policy catches more attacks on this split, with lower attack precision and more false alerts. Its macro F1 is essentially tied with the supervised baseline. This is a recall/false-alarm tradeoff, not a general win for reinforcement learning.

### Confusion matrices

Rows are true labels and columns are predictions, ordered `[normal, attack]`.

**Random Forest** (validation-selected attack probability threshold: `0.41655`):

```text
[[9441,  270],
 [4249, 8584]]
```

For DDQN, `alert` and `block` count as detected threats:

```text
[[8942,  769],
 [3771, 9062]]
```

### DDQN response actions

| True class | Allow | Alert | Block | Total |
| --- | ---: | ---: | ---: | ---: |
| Normal | 8,942 | 454 | 315 | 9,711 |
| Attack | 3,771 | 927 | 8,135 | 12,833 |
| **Total** | **12,713** | **1,381** | **8,450** | **22,544** |

The policy blocked 315 normal test records. A live system must treat `block` as a potentially disruptive action and validate it against its own traffic and response controls before enforcement. Mean reward was `0.5815` under the reward table in the README; this value is specific to those chosen rewards and should not be compared across different reward schemes.

## Reproduce

Place the NSL-KDD files under `data/` and run from an environment where the project is installed:

```powershell
ids-train --train data/KDDTrain+.txt --test data/KDDTest+.txt
ids-ddqn --train data/KDDTrain+.txt --test data/KDDTest+.txt --steps 10000 --seed 42
```

Model artifacts are written under `artifacts/` and are intentionally ignored by Git. The trained artifacts used for this report were local evaluation outputs, not committed files.

## Password classifier

No labeled password dataset was found in the project or checked data folders, so the password classifier has not been evaluated on real data. `password-train` now writes held-out JSON metrics (accuracy, macro F1, confusion matrix, and per-class report) when a labeled feature CSV is supplied. Synthetic smoke checks only verify that the pipeline executes; they are not password-model performance results.

## Action-dependent simulator

The project now includes a separate synthetic threat-session environment where the response affects future threat persistence. One reproducible run trained for 300 episodes and evaluated on 300 separate simulated episodes (seed 42; evaluation seed 43):

| Policy | Mean reward per episode | Threat detection rate | False blocks per benign step |
| --- | ---: | ---: | ---: |
| DDQN | 31.1047 | 92.60% | 0.62% |
| Always allow | -21.7533 | 0.00% | 0.00% |
| Always alert | -7.1083 | 100.00% | 0.00% |
| Always block | -36.9113 | 100.00% | 100.00% |

Here, detection means choosing `alert` or `block` on a simulated malicious step. The reward reflects the simulator's chosen costs, so this score is only useful for comparing policies inside this exact setup. The result is synthetic and must not be read as an NSL-KDD or live-network metric. The simulator is kept separate from the KDD results above because it uses a different state schema and reward process.

## Interpretation and next work

- The supervised model is a strong comparison point for static labeled records; reinforcement learning is not automatically better for this task.
- The offline DDQN environment replays records and uses labels for immediate response rewards; its next observation is independent of the action. The synthetic simulator adds action-dependent transitions, but realistic costs, calibrated traffic dynamics, and repeated seed runs are still needed before drawing operational conclusions.
- Do not tune further against `KDDTest+`; use training-only validation or a separate test set for future changes.
