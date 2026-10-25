# Cybersecurity ML Lab

A research prototype for intrusion detection and password analysis. It includes a supervised IDS baseline, offline and synthetic DDQN policy experiments, a packet-summary flow adapter, and password utility scripts.

> **Scope:** The packet monitor uses a flow model trained on the same nine observable flow fields produced by the packet adapter. It emits allow/alert recommendations and does not modify firewall rules. Benchmark evaluation uses NSL-KDD records projected onto those fields; it is not a live-network capture evaluation.

[Evaluation results](RESULTS.md) | [Setup](#setup) | [Limitations](#limitations)

## Project structure

```text
src/cybersecurity_ml/
|-- ids/
|   |-- data.py           # NSL-KDD schema and preprocessing
|   |-- train.py          # Supervised Random Forest baseline
|   |-- ddqn.py           # NumPy Double DQN agent
|   |-- environment.py    # Labeled offline response replay
|   |-- lab_simulator.py  # Synthetic action-dependent threat sessions
|   |-- simulate.py       # Train/evaluate DDQN in the simulator
|   |-- flow_adapter.py   # Packet-summary aggregation to observable flow records
|   |-- flow_train.py    # Flow-schema training and FPR calibration
|   |-- responses.py      # Recommendation-only response adapter interface
|   |-- rl_train.py       # DDQN training and held-out evaluation
|   `-- monitor.py        # KDD or packet-summary JSONL recommendations
`-- passwords/
    |-- features.py       # Composition, diversity, repetition, and pattern features
    |-- eda.py            # Feature export without raw passwords
    |-- train_classifier.py # Held-out metrics JSON for labeled data
    `-- generate.py       # Cryptographically random passwords
```

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate` after creating the environment. Dependencies are declared in `pyproject.toml`.

Keep datasets in a local `data/` directory. Download the NSL-KDD `KDDTrain+` and `KDDTest+` files from the [dataset record](https://zenodo.org/records/17424143) into `data/raw/`. Dataset files, model artifacts, and password CSVs are excluded from Git. Check dataset terms before redistributing any data.

## Intrusion detection

### Supervised baseline

`ids-train` fits a class-balanced Random Forest. Preprocessing is fit on training records, unknown test categories are supported, and the attack threshold is selected on a validation split from the training file. The supplied test file is used for final evaluation.

```powershell
ids-train --train data/KDDTrain+.txt --test data/KDDTest+.txt
```

The saved artifact contains the fitted pipeline and its attack threshold.

### Packet-observable flow model

The packet adapter emits only fields it can measure or derive: duration, protocol, destination-port service, TCP-state summary, directional payload bytes, same-endpoint indicator, wrong-fragment count, and urgent-flag count. `ids-flow-train` trains and evaluates a separate model using precisely these fields. It selects a threshold on a split of KDDTrain+ to target a false-positive rate, then reports held-out KDDTest+ results.

```powershell
ids-flow-train --train 'data/raw/KDDTrain+.txt' --test 'data/raw/KDDTest+.txt' --max-fpr 0.001
```

This evaluates the reduced schema on NSL-KDD benchmark records. Threshold calibration targets a 0.1% false-positive rate on a training-only validation split; the held-out test rate can differ under distribution shift. See [RESULTS.md](RESULTS.md). NSL-KDD is not packet-capture data, so this is not live-traffic validation. Service and TCP-state fields are best-effort summaries from packet metadata.

### DDQN response policy

`ids-ddqn` trains a Double DQN agent with replay memory, an online Q-network, and a periodically synchronized target network. It selects among `allow`, `alert`, and `block`. Training uses these rewards:

| True class | Allow | Alert | Block |
| --- | ---: | ---: | ---: |
| Normal | 1.0 | -0.25 | -1.0 |
| Attack | -1.0 | 0.25 | 1.0 |

```powershell
ids-ddqn --train data/KDDTrain+.txt --test data/KDDTest+.txt --steps 10000 --seed 42
```

This is an **offline replay experiment**. NSL-KDD supplies labels for rewards, but it does not model how a response changes later traffic. The reported DDQN results therefore measure a learned per-record response policy, not long-term policy behavior or live-network effectiveness.

### Action-dependent simulator

`ids-simulate` trains a separate DDQN policy in a synthetic environment where allowing a threat can increase its persistence, alerting slows it, and blocking suppresses it. It evaluates separate simulated episodes and compares mean reward, attack detection rate, and false blocks with fixed allow, alert, and block policies.

```powershell
ids-simulate --episodes 300 --seed 42
```

This is a controlled research model, not evidence of real network performance. Its artifact uses a six-feature schema and cannot be loaded by the KDD monitor.

### Streaming decisions

`ids-monitor` in KDD mode reads newline-delimited JSON events containing all NSL-KDD feature fields except `outcome` and `difficulty` (41 fields total).

```powershell
Get-Content .\flow-events.jsonl -Wait | ids-monitor --model artifacts/ids_ddqn.joblib
```

The monitor emits JSON recommendations and never creates firewall rules. Packet mode requires an `ids-flow-train` artifact and accepts JSONL events with `timestamp` (Unix seconds or ISO-8601), `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol` (`tcp`, `udp`, `icmp`), and `packet_length`; optional fields are `payload_length`, `tcp_flags`, and `wrong_fragment`:

```powershell
Get-Content .\packet-events.jsonl -Wait | ids-monitor --model artifacts/ids_flow.joblib --input-format packets --idle-timeout 5
```

Flows are grouped by bidirectional five-tuple and emitted after the idle timeout or at end of input. Packet mode emits only `allow` and `alert`; `ResponseAdapter` separates inference from response, and the built-in `RecommendationAdapter` does not modify the host.

## Password tools

Generate passwords with the operating system's cryptographic random source:

```powershell
password-generate --length 24 --count 3
```

For a labeled password CSV, extract composition, repetition, diversity, and common-pattern features and train the baseline classifier. Both `password`/`strength` and `Password`/`Strength_Level` columns are accepted. For large files, `--sample-per-class` scans the input and retains a reproducible reservoir sample per label:

```powershell
password-eda data/raw/passwords.csv --sample-per-class 10000 --output data/processed/password_features.csv
password-train data/processed/password_features.csv
```

The exported feature table omits raw password strings. Do not commit source password datasets, generated secrets, or feature files derived from private data. For a reproducible public benchmark, PWLDS provides labels from five strength groups under CC BY 4.0; it is synthetically generated, so its results should not be generalized to real user passwords. See the [dataset card](https://github.com/Infinitode/PWLDS) and [download page](https://huggingface.co/datasets/InfinitodeLTD/PWLDS).

Training writes held-out accuracy, macro F1, confusion matrix, and class metrics to `artifacts/password_strength_metrics.json` by default; use `--metrics-out` to choose another path.

## Limitations

- The NSL-KDD results are a reproducible baseline, not evidence of current production-network performance.
- The offline DDQN replay environment has no action-dependent traffic transitions. The synthetic simulator models action effects only under its stated assumptions.
- Flow-model metrics measure NSL-KDD benchmark rows projected onto packet-observable fields; they do not establish performance on live captures or a specific network.
- Password features are heuristic inputs, not a complete measure of password security. Evaluation on synthetic labels cannot establish performance on real user-chosen passwords.
- Results depend on the dataset, label mapping, reward values, and evaluation split. See [RESULTS.md](RESULTS.md) for measured tradeoffs and reproducible evaluation commands.
