from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "outcome", "difficulty",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def load_kdd(path: str | Path) -> pd.DataFrame:
    """Load a headerless NSL-KDD text file."""
    data = pd.read_csv(path, header=None, names=KDD_COLUMNS, skipinitialspace=True)
    if data.empty:
        raise ValueError(f"Dataset is empty: {path}")
    data["outcome"] = data["outcome"].astype(str).str.strip().str.rstrip(".")
    return data


def split_features_and_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map normal traffic to 0 and every attack class to 1."""
    labels = (data["outcome"].str.lower() != "normal").astype("int8")
    return data.drop(columns=["outcome", "difficulty"], errors="ignore"), labels


def build_preprocessor() -> ColumnTransformer:
    """Fit numeric scaling and categorical encoding within the model pipeline."""
    numeric = [c for c in KDD_COLUMNS if c not in CATEGORICAL_COLUMNS + ["outcome", "difficulty"]]
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])
