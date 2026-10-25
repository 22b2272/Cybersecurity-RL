from __future__ import annotations

"""Stream KDD rows or packet-summary JSONL through a trained DDQN policy."""
import argparse
import json
import queue
import sys
import threading
from pathlib import Path

import joblib
import pandas as pd

from cybersecurity_ml.ids.ddqn import ACTION_NAMES
from cybersecurity_ml.ids.flow_adapter import FEATURE_COLUMNS, PacketFlowFeatureAdapter
from cybersecurity_ml.ids.data import KDD_COLUMNS
from cybersecurity_ml.ids.responses import RecommendationAdapter
from cybersecurity_ml.ids.rl_train import _dense_float32


def decide(bundle: dict, event: dict) -> str:
    """Choose an action for one KDD-shaped feature record without exploration."""
    kdd_features = [name for name in KDD_COLUMNS if name not in {"outcome", "difficulty"}]
    missing = set(kdd_features).difference(event)
    if missing:
        raise ValueError(f"Event missing KDD features: {', '.join(sorted(missing))}")
    row = pd.DataFrame([{column: event[column] for column in kdd_features}])
    state = _dense_float32(bundle["preprocessor"].transform(row))[0]
    action = bundle["agent"].select_action(state, explore=False)
    return ACTION_NAMES[action]


def decide_flow(bundle: dict, event: dict) -> str:
    """Classify an observed flow with the matching supervised flow artifact."""
    if bundle.get("model_type") != "packet_flow_random_forest_v1":
        raise ValueError("Packet mode requires an ids-flow-train artifact")
    columns = bundle["feature_columns"]
    missing = set(columns).difference(event)
    if missing:
        raise ValueError(f"Flow event missing features: {', '.join(sorted(missing))}")
    row = pd.DataFrame([{column: event[column] for column in columns}])
    model = bundle["pipeline"]
    attack_index = list(model.classes_).index(1)
    score = model.predict_proba(row)[0, attack_index]
    return "alert" if score >= bundle["attack_threshold"] else "allow"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="DDQN artifact for KDD mode; ids-flow-train artifact for packet mode")
    parser.add_argument("--input-format", choices=("kdd", "packets"), default="kdd",
                        help="KDD feature JSONL rows or packet-summary JSONL to aggregate")
    parser.add_argument("--idle-timeout", type=float, default=5.0,
                        help="Seconds before a packet flow is emitted (packets mode)")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    response_adapter = RecommendationAdapter()
    flow_adapter = PacketFlowFeatureAdapter(args.idle_timeout) if args.input_format == "packets" else None

    def emit(event: dict) -> None:
        action = decide_flow(bundle, event) if flow_adapter is not None else decide(bundle, event)
        result = response_adapter.dispatch(event.get("event_id"), action)
        print(json.dumps(result), flush=True)

    lines: queue.Queue[str | None] = queue.Queue()

    def read_lines() -> None:
        for input_line in sys.stdin:
            lines.put(input_line)
        lines.put(None)

    threading.Thread(target=read_lines, daemon=True).start()
    line_number = 0
    input_closed = False
    while not input_closed:
        try:
            line = lines.get(timeout=0.25 if flow_adapter is not None else None)
        except queue.Empty:
            for flow_event in flow_adapter.expire_idle():
                try:
                    emit(flow_event)
                except (TypeError, ValueError) as error:
                    print(json.dumps({"error": str(error)}), file=sys.stderr, flush=True)
            continue
        if line is None:
            input_closed = True
            break
        line_number += 1
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Each JSONL line must contain an object")
            if flow_adapter is None:
                emit(event)
            else:
                for flow_event in flow_adapter.feed(event):
                    emit(flow_event)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            print(json.dumps({"line": line_number, "error": str(error)}), file=sys.stderr, flush=True)
    if flow_adapter is not None:
        for flow_event in flow_adapter.flush():
            emit(flow_event)


if __name__ == "__main__":
    main()
