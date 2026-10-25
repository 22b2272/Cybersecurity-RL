"""Predict password-strength labels from passwords read one per stdin line."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from cybersecurity_ml.passwords.features import extract_features
from cybersecurity_ml.passwords.train_classifier import FEATURE_COLUMNS


def predict_passwords(model_path: str | Path, passwords: list[str]) -> list[dict]:
    """Return predictions without writing or logging the supplied passwords."""
    model = joblib.load(model_path)
    features = pd.DataFrame([extract_features(password) for password in passwords],
                            columns=FEATURE_COLUMNS)
    if not passwords:
        return []
    predictions = model.predict(features)
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    results = []
    for index, label in enumerate(predictions):
        result = {"strength": label.item() if hasattr(label, "item") else label}
        if probabilities is not None:
            result["confidence"] = float(probabilities[index].max())
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/password_strength.joblib"))
    args = parser.parse_args()
    # Read stdin to avoid placing passwords in shell history or process arguments.
    passwords = [line.rstrip("\r\n") for line in sys.stdin]
    try:
        for result in predict_passwords(args.model, passwords):
            print(json.dumps(result), flush=True)
    except (OSError, ValueError, AttributeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
