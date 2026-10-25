"""Train and evaluate an NSL-KDD intrusion detection baseline."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from cybersecurity_ml.ids.data import build_preprocessor, load_kdd, split_features_and_labels


def build_model() -> Pipeline:
    """Create the fixed, reproducible Random Forest pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(train_path: str | Path) -> Pipeline:
    """Fit the model on the complete training file."""
    data = load_kdd(train_path)
    features, labels = split_features_and_labels(data)
    return build_model().fit(features, labels)


def calibrate_attack_threshold(train_path: str | Path, validation_size: float = 0.2) -> float:
    """Choose an attack threshold by maximizing attack F1 on a training-only holdout."""
    data = load_kdd(train_path)
    features, labels = split_features_and_labels(data)
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        features,
        labels,
        test_size=validation_size,
        random_state=42,
        stratify=labels,
    )
    validation_model = build_model().fit(x_fit, y_fit)
    classifier = validation_model.named_steps["classifier"]
    attack_index = list(classifier.classes_).index(1)
    probabilities = validation_model.predict_proba(x_validation)[:, attack_index]
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f1_scores))])


def evaluate_model(model: Pipeline, test_path: str | Path, threshold: float = 0.5,
                   metrics_path: str | Path | None = None) -> float:
    """Report held-out metrics at a fixed attack-probability threshold."""
    data = load_kdd(test_path)
    features, labels = split_features_and_labels(data)
    classifier = model.named_steps["classifier"]
    attack_index = list(classifier.classes_).index(1)
    probabilities = model.predict_proba(features)[:, attack_index]
    predictions = (probabilities >= threshold).astype("int8")
    report = classification_report(labels, predictions, labels=[0, 1],
                                   target_names=["normal", "attack"], output_dict=True,
                                   zero_division=0)
    print(classification_report(labels, predictions, target_names=["normal", "attack"], zero_division=0))
    print("Confusion matrix [normal, attack]:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    macro_f1 = float(f1_score(labels, predictions, average="macro"))
    if metrics_path is not None:
        metrics = {
            "model_type": "nsl_kdd_random_forest_v1", "dataset": str(test_path),
            "rows": int(len(labels)), "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, predictions)), "macro_f1": macro_f1,
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
            "classification_report": report,
            "threshold_calibration": "training-file stratified holdout; test labels not used for fitting",
        }
        output = Path(metrics_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Path to KDDTrain+.txt")
    parser.add_argument("--test", type=Path, required=True, help="Path to KDDTest+.txt")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/ids_random_forest.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/ids_random_forest_metrics.json"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Attack probability threshold; by default, select it on a holdout from training data.",
    )
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    threshold = args.threshold
    if threshold is None:
        threshold = calibrate_attack_threshold(args.train)
        print(f"Selected attack threshold from training validation split: {threshold:.4f}")
    model = train_model(args.train)
    macro_f1 = evaluate_model(model, args.test, threshold, args.metrics_out)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "attack_threshold": threshold}, args.model_out)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved model and threshold to {args.model_out}")
    print(f"Saved evaluation metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
