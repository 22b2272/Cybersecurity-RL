import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cybersecurity_ml.ids.data import KDD_COLUMNS, load_kdd, split_features_and_labels
from cybersecurity_ml.ids.ddqn import DDQNAgent, QNetwork
from cybersecurity_ml.ids.flow_adapter import PacketFlowFeatureAdapter
from cybersecurity_ml.passwords.eda import build_feature_table


class KddDataTests(unittest.TestCase):
    def test_rejects_unknown_labels(self):
        row = ["0"] * len(KDD_COLUMNS)
        row[KDD_COLUMNS.index("protocol_type")] = "tcp"
        row[KDD_COLUMNS.index("service")] = "http"
        row[KDD_COLUMNS.index("flag")] = "SF"
        row[KDD_COLUMNS.index("outcome")] = "not_a_label"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.txt"
            path.write_text(",".join(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown NSL-KDD"):
                load_kdd(path)

    def test_known_attack_maps_to_one(self):
        features, labels = split_features_and_labels(
            pd.DataFrame({"outcome": pd.Series(["normal", "neptune."])})
        )
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertNotIn("outcome", features.columns)


class DdqnTests(unittest.TestCase):
    def test_empty_prediction_and_input_validation(self):
        agent = DDQNAgent(state_size=2, batch_size=2, memory_size=4)
        self.assertEqual(agent.predict_actions(np.empty((0, 2))).shape, (0,))
        with self.assertRaises(ValueError):
            agent.remember(np.zeros(2), 3, 0.0, np.zeros(2), False)

    def test_network_update_changes_parameters(self):
        network = QNetwork(input_size=2, action_size=3, seed=7)
        before = network.parameters["w2"].copy()
        loss = network.train(np.ones((4, 2)), np.ones((4, 3)))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, network.parameters["w2"]))


class FlowAdapterTests(unittest.TestCase):
    def test_service_detected_when_first_packet_is_server_response(self):
        adapter = PacketFlowFeatureAdapter()
        adapter.feed({"timestamp": 1, "src_ip": "server", "dst_ip": "client",
                      "src_port": 80, "dst_port": 50000, "protocol": "tcp",
                      "packet_length": 100, "payload_length": 60})
        flow = adapter.flush()[0]
        self.assertEqual(flow["service"], "http")
        self.assertEqual(flow["src_bytes"], 60)

    def test_rejects_impossible_payload_length(self):
        adapter = PacketFlowFeatureAdapter()
        with self.assertRaisesRegex(ValueError, "lengths must satisfy"):
            adapter.feed({"timestamp": 1, "src_ip": "a", "dst_ip": "b",
                          "src_port": 1, "dst_port": 2, "protocol": "udp",
                          "packet_length": 10, "payload_length": 11})


class PasswordDataTests(unittest.TestCase):
    def test_csv_password_is_read_as_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "passwords.csv"
            pd.DataFrame({"Password": ["00123", "abc"],
                          "Strength_Level": [0, 1]}).to_csv(path, index=False)
            result = build_feature_table(path)
        self.assertEqual(result.loc[0, "length"], 5)
        self.assertEqual(result["strength"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
