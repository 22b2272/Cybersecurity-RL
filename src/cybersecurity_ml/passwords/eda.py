"""Create composition features from a labeled password CSV."""
import argparse
from pathlib import Path
import pandas as pd
from cybersecurity_ml.passwords.features import extract_features


def build_feature_table(input_path: str | Path) -> pd.DataFrame:
    """Return features and labels without carrying raw plaintext passwords forward."""
    data = pd.read_csv(input_path, on_bad_lines="skip")
    missing = {"password", "strength"}.difference(data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    data = data.dropna(subset=["password", "strength"])
    result = pd.DataFrame(data["password"].astype(str).map(extract_features).tolist())
    result["strength"] = data["strength"].to_numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with password and strength columns")
    parser.add_argument("--output", type=Path, default=Path("data/processed/password_features.csv"))
    args = parser.parse_args()
    result = build_feature_table(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}; raw passwords were excluded.")


if __name__ == "__main__":
    main()
