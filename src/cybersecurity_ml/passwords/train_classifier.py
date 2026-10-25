"""Train and evaluate a password-strength classifier from labeled features."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["length", "uppercase", "lowercase", "digits", "special"]


def train_classifier(
    data_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path | None = None,
) -> dict:
    """Fit on a stratified split, save the estimator, and write held-out metrics."""
    data = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + ["strength"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Feature CSV missing columns: {', '.join(sorted(missing))}")
    clean = data.dropna(subset=list(required))
    if clean["strength"].nunique() < 2:
        raise ValueError("At least two strength classes are needed for training")
    counts = clean["strength"].value_counts()
    if counts.min() < 2:
        raise ValueError("Every strength class needs at least two rows for a stratified holdout")
    x_train, x_test, y_train, y_test = train_test_split(
        clean[FEATURE_COLUMNS], clean["strength"], test_size=0.2,
        random_state=42, stratify=clean["strength"],
    )
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = list(model.classes_)
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0,
    )
    metrics = {
        "dataset": str(data_path),
        "rows_total": int(len(clean)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "features": FEATURE_COLUMNS,
        "classes": [str(label) for label in labels],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": report,
        "random_state": 42,
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    metrics_output = Path(metrics_path) if metrics_path else output.with_name("password_strength_metrics.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print(f"Accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved model to {output}")
    print(f"Saved held-out metrics to {metrics_output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="CSV created by password-eda")
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/password_strength.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="JSON metrics file (defaults beside the model)")
    args = parser.parse_args()
    train_classifier(args.features, args.model_out, args.metrics_out)


if __name__ == "__main__":
    main()
