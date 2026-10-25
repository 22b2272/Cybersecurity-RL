"""Create composition features from a labeled password CSV."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from cybersecurity_ml.passwords.features import extract_features


def _column_map(columns) -> dict[str, str]:
    """Resolve common password/label column names, including PWLDS headers."""
    normalized = {str(column).strip().lower().replace("_", ""): column for column in columns}
    password = next((normalized[key] for key in ("password", "passwd") if key in normalized), None)
    strength = next((normalized[key] for key in ("strength", "strengthlevel", "label", "level")
                     if key in normalized), None)
    if password is None or strength is None:
        raise ValueError("CSV needs password/Password and strength/Strength_Level columns")
    return {"password": password, "strength": strength}


def build_feature_table(input_path: str | Path, sample_per_class: int | None = None,
                        seed: int = 42, chunksize: int = 100_000) -> pd.DataFrame:
    """Return features and labels; optionally reservoir-sample each class.

    The raw passwords are held only while selecting/extracting features and are
    never written to the resulting feature table.
    """
    if sample_per_class is not None and sample_per_class < 1:
        raise ValueError("sample_per_class must be positive")
    header = pd.read_csv(input_path, nrows=0)
    columns = _column_map(header.columns)
    usecols = list(columns.values())
    if sample_per_class is None:
        data = pd.read_csv(input_path, usecols=usecols, on_bad_lines="skip")
        data = data.rename(columns={columns["password"]: "password", columns["strength"]: "strength"})
    else:
        rng = np.random.default_rng(seed)
        reservoirs: dict[object, list[tuple[float, str]]] = {}
        for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize, on_bad_lines="skip"):
            chunk = chunk.rename(columns={columns["password"]: "password", columns["strength"]: "strength"})
            chunk = chunk.dropna(subset=["password", "strength"])
            priorities = rng.random(len(chunk))
            chunk = chunk.assign(_priority=priorities)
            for label, group in chunk.groupby("strength", sort=False):
                current = reservoirs.setdefault(label, [])
                candidates = current + list(zip(group["_priority"].to_numpy(), group["password"].astype(str)))
                candidates.sort(key=lambda item: item[0])
                reservoirs[label] = candidates[:sample_per_class]
        data = pd.DataFrame([
            {"password": password, "strength": label}
            for label, records in reservoirs.items() for _, password in records
        ])
    data = data.dropna(subset=["password", "strength"])
    result = pd.DataFrame(data["password"].astype(str).map(extract_features).tolist())
    result["strength"] = data["strength"].to_numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with password and strength columns")
    parser.add_argument("--output", type=Path, default=Path("data/processed/password_features.csv"))
    parser.add_argument("--sample-per-class", type=int, default=None,
                        help="Reservoir sample at most this many rows per label (scans the full CSV)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build_feature_table(args.input, args.sample_per_class, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}; raw passwords were excluded.")


if __name__ == "__main__":
    main()
