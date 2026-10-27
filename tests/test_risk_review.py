import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from risk_review import (
    build_review,
    inspect_application_model,
    load_applicant_snapshot,
    load_graph_snapshot,
)


class RiskReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.application_csv = self.root / "application.csv"
        self.graph_csv = self.root / "graph.csv"
        self.model_json = self.root / "model.json"

        pd.DataFrame([{
            "applicant_id": "APP100000",
            "identity_coherence_score": 0.8,
            "identity_risk_from_ics": 0.2,
            "xgb_proba": 0.4,
            "final_identity_risk_score": 0.34,
            "risk_category": "Medium",
        }]).to_csv(self.application_csv, index=False)
        pd.DataFrame([{
            "node_id": "NODE_00003",
            "out_degree": 2,
            "in_degree": 3,
            "total_degree": 5,
            "ics": 0.7,
            "ics_risk": 0.3,
            "fraud_probability": 0.6,
            "predicted_fraud": 1,
        }]).to_csv(self.graph_csv, index=False)
        self.model_json.write_text(json.dumps({
            "learner": {"feature_names": ["b", "a"]}
        }), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_saved_applicant_snapshot(self):
        record = load_applicant_snapshot(self.application_csv, "app100000")
        self.assertEqual(record["applicant_id"], "APP100000")
        self.assertAlmostEqual(record["xgb_proba"], 0.4)
        self.assertIn("not live inference", record["score_origin"])

    def test_missing_applicant_is_clear(self):
        with self.assertRaises(KeyError):
            load_applicant_snapshot(self.application_csv, "APP999999")

    def test_rejects_out_of_range_saved_scores(self):
        frame = pd.read_csv(self.application_csv)
        frame.loc[0, "xgb_proba"] = 1.2
        frame.to_csv(self.application_csv, index=False)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            load_applicant_snapshot(self.application_csv, "APP100000")

    def test_loads_graph_node_without_inventing_applicant_mapping(self):
        record = load_graph_snapshot(self.graph_csv, "node_00003")
        self.assertEqual(record["node_id"], "NODE_00003")
        self.assertIn("no applicant-to-node mapping", record["score_origin"])

    def test_model_features_can_be_selected_in_declared_order(self):
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(self.application_csv, index=False)
        status = inspect_application_model(self.model_json, self.application_csv)
        self.assertEqual(status["feature_names"], ["b", "a"])
        self.assertTrue(status["feature_order_can_be_selected"])
        self.assertFalse(status["missing_csv_features"])

    def test_review_never_creates_combined_decision(self):
        review = build_review(
            {"applicant_id": "APP100000"},
            {"node_id": "NODE_00003"},
        )
        self.assertIsNone(review["combined_decision"])
        self.assertIn("no validated applicant-to-node mapping", review["decision_status"])


if __name__ == "__main__":
    unittest.main()
