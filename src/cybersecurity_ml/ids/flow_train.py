"""Train and evaluate an IDS model on packet-observable NSL-KDD fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cybersecurity_ml.ids.data import load_kdd, split_features_and_labels
from cybersecurity_ml.ids.flow_adapter import FEATURE_COLUMNS

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL]


def build_flow_model() -> Pipeline:
    """Create the flow model using only features the packet adapter emits."""
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=42,
        )),
    ])


def _flow_data(path: str | Path):
    data = load_kdd(path)
    features, labels = split_features_and_labels(data)
    return features[FEATURE_COLUMNS], labels


def _calibrate_threshold(x, y, max_false_positive_rate: float) -> tuple[float, float]:
    x_fit, x_val, y_fit, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )
    model = build_flow_model().fit(x_fit, y_fit)
    attack_index = list(model.classes_).index(1)
    scores = model.predict_proba(x_val)[:, attack_index]
    thresholds = np.unique(scores)
    normal_scores = scores[np.asarray(y_val) == 0]
    candidates = np.unique(np.r_[0.0, thresholds, 1.0])
    valid = [t for t in candidates if np.mean(normal_scores >= t) <= max_false_positive_rate]
    threshold = float(min(valid)) if valid else 1.0
    validation_fpr = float(np.mean(normal_scores >= threshold))
    return threshold, validation_fpr


def train_and_evaluate(train_path: str | Path, test_path: str | Path,
                       model_path: str | Path, metrics_path: str | Path | None = None,
                       max_false_positive_rate: float = 0.001) -> dict:
    """Fit on KDDTrain+, calibrate on its internal holdout, evaluate KDDTest+."""
    if not 0 <= max_false_positive_rate <= 1:
        raise ValueError("max_false_positive_rate must be between 0 and 1")
    x_train, y_train = _flow_data(train_path)
    threshold, validation_fpr = _calibrate_threshold(x_train, y_train, max_false_positive_rate)
    model = build_flow_model().fit(x_train, y_train)
    x_test, y_test = _flow_data(test_path)
    scores = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    predictions = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(y_test, predictions, labels=[0, 1],
                                   target_names=["normal", "attack"],
                                   output_dict=True, zero_division=0)
    metrics = {
        "model_type": "packet_flow_random_forest_v1",
        "train_dataset": str(train_path), "test_dataset": str(test_path),
        "test_rows": int(len(y_test)), "features": FEATURE_COLUMNS,
        "threshold": threshold, "validation_max_fpr": max_false_positive_rate,
        "validation_false_positive_rate": validation_fpr,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "false_positive_rate": float(matrix[0, 1] / max(1, matrix[0].sum())),
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "classification_report": report,
        "calibration": "stratified 20% split from training file; test file not used",
    }
    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_type": "packet_flow_random_forest_v1", "pipeline": model,
                 "feature_columns": FEATURE_COLUMNS, "attack_threshold": threshold}, out)
    metrics_out = Path(metrics_path) if metrics_path else out.with_name("ids_flow_metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(classification_report(y_test, predictions, target_names=["normal", "attack"], zero_division=0))
    print(f"Threshold={threshold:.6f}; FPR={metrics['false_positive_rate']:.4%}; macro F1={metrics['macro_f1']:.4f}")
    print(f"Saved model: {out}\nSaved metrics: {metrics_out}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Headerless NSL-KDD KDDTrain+ file")
    parser.add_argument("--test", type=Path, required=True, help="Headerless NSL-KDD KDDTest+ file")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/ids_flow.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--max-fpr", type=float, default=0.001,
                        help="Training-validation false positive rate target (default: 0.001)")
    args = parser.parse_args()
    train_and_evaluate(args.train, args.test, args.model_out, args.metrics_out, args.max_fpr)


if __name__ == "__main__":
    main()
