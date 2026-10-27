"""Synthetic-only tests for disk-backed Module C node aggregates."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module_c_disk_features import (
    IMPLEMENTED_NODE_COLUMNS,
    IMPLEMENTED_FEATURE_COLUMNS,
    TemporaryStorageLimitExceeded,
    populate_node_features,
)
from module_c_disk_prototype import open_bounded_disk_prototype
from module_c_graph_builder import FEATURE_COLUMNS, build_graph


def tiny_frame():
    rows = [
        (1, "A", 2, "X", 10.0, 9.0, 1),
        (2, "B", 1, "A", 3.0, 3.0, 0),
        (1, "A", 2, "X", 10.0, 9.0, 1),
        (1, "A", 4, "D", 5.0, 4.0, 0),
        (4, "D", 4, "D", 2.0, 2.0, 0),
    ]
    columns = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering",
    ]
    records = []
    for row_index, (from_bank, account, to_bank, account_to, paid, received, label) in enumerate(rows):
        records.append(
            {
                "Timestamp": f"2022/09/01 00:{row_index:02d}",
                "From Bank": from_bank,
                "Account": account,
                "To Bank": to_bank,
                "Account.1": account_to,
                "Amount Received": received,
                "Receiving Currency": "USD",
                "Amount Paid": paid,
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": label,
            }
        )
    return pd.DataFrame(records, columns=columns)


def independent_expected(frame):
    sender_keys = [
        f"{row['From Bank']}_{row['Account']}"
        for row in frame.to_dict(orient="records")
    ]
    receiver_keys = [
        f"{row['To Bank']}_{row['Account.1']}"
        for row in frame.to_dict(orient="records")
    ]
    ordered = list(dict.fromkeys(sender_keys))
    seen = set(ordered)
    for receiver in receiver_keys:
        if receiver not in seen:
            seen.add(receiver)
            ordered.append(receiver)
    mapping = {key: index for index, key in enumerate(ordered)}

    stats = {
        index: {
            "total_sent": 0.0,
            "total_received": 0.0,
            "num_sent": 0,
            "num_received": 0,
            "fraud_sent": 0,
            "fraud_received": 0,
            "receivers": set(),
            "senders": set(),
            "in_degree": 0,
            "out_degree": 0,
        }
        for index in range(len(ordered))
    }
    topology = set()
    for row, sender_key, receiver_key in zip(
        frame.to_dict(orient="records"), sender_keys, receiver_keys
    ):
        source = mapping[sender_key]
        target = mapping[receiver_key]
        stats[source]["total_sent"] += float(row["Amount Paid"])
        stats[source]["num_sent"] += 1
        stats[source]["fraud_sent"] += int(row["Is Laundering"])
        stats[source]["receivers"].add(target)
        stats[target]["total_received"] += float(row["Amount Received"])
        stats[target]["num_received"] += 1
        stats[target]["fraud_received"] += int(row["Is Laundering"])
        stats[target]["senders"].add(source)
        topology.add((source, target))

    for source, target in topology:
        stats[source]["out_degree"] += 1
        stats[target]["in_degree"] += 1

    expected = {}
    for key, index in mapping.items():
        item = stats[index]
        total_fraud = item["fraud_sent"] + item["fraud_received"]
        expected[key] = {
            "label": int(total_fraud > 0),
            "total_sent": item["total_sent"],
            "total_received": item["total_received"],
            "num_sent": item["num_sent"],
            "num_received": item["num_received"],
            "avg_sent": item["total_sent"] / item["num_sent"] if item["num_sent"] else 0,
            "avg_received": item["total_received"] / item["num_received"] if item["num_received"] else 0,
            "unique_receivers": len(item["receivers"]),
            "unique_senders": len(item["senders"]),
            "net_flow": item["total_received"] - item["total_sent"],
            "in_degree": item["in_degree"],
            "out_degree": item["out_degree"],
        }
    return expected


class DiskFeatureTests(unittest.TestCase):
    def setUp(self):
        self.root_context = tempfile.TemporaryDirectory(prefix="module-c-feature-test-")
        self.addCleanup(self.root_context.cleanup)
        self.root = Path(self.root_context.name)
        self.csv_path = self.root / "transactions.csv"
        self.frame = tiny_frame()
        self.frame.to_csv(self.csv_path, index=False)

    def test_exact_aggregates_labels_degrees_and_duplicate_boundary_match_reference(self):
        expected = independent_expected(self.frame)
        helper = build_graph(self.frame, seed=151)
        helper_features = helper["features"].set_index("account")
        self.assertEqual(FEATURE_COLUMNS[:11], IMPLEMENTED_FEATURE_COLUMNS)
        self.assertEqual(IMPLEMENTED_NODE_COLUMNS, ["label", *FEATURE_COLUMNS[:11]])

        with open_bounded_disk_prototype(
            self.csv_path,
            max_rows=len(self.frame),
            chunk_rows=2,
            temp_root=self.root,
        ) as prototype_store:
            features = populate_node_features(
                self.csv_path,
                prototype_store,
                max_rows=len(self.frame),
                batch_rows=2,
                max_temp_bytes=8 * 1024 * 1024,
            )
            rows = list(features.iter_rows(batch_rows=2))
            self.assertEqual(features.row_count, len(self.frame))

        self.assertEqual(len(rows), len(expected))
        self.assertEqual([row["account"] for row in rows], list(expected))
        self.assertEqual(rows[0]["num_sent"], 3)
        self.assertEqual(rows[0]["unique_receivers"], 2)
        self.assertEqual(
            next(row for row in rows if row["account"] == "2_X")["unique_senders"],
            1,
        )
        self_loop = next(row for row in rows if row["account"] == "4_D")
        self.assertEqual(self_loop["in_degree"], 2)
        self.assertEqual(self_loop["out_degree"], 1)
        for row in rows:
            reference = expected[row["account"]]
            for column in IMPLEMENTED_NODE_COLUMNS:
                self.assertAlmostEqual(float(row[column]), float(reference[column]), places=12)
            helper_row = helper_features.loc[row["account"]]
            for column in IMPLEMENTED_FEATURE_COLUMNS:
                self.assertAlmostEqual(
                    float(row[column]), float(helper_row[column]), places=12
                )

    def test_row_limit_and_batch_limit_are_enforced(self):
        with open_bounded_disk_prototype(
            self.csv_path,
            max_rows=4,
            chunk_rows=2,
            temp_root=self.root,
        ) as store:
            features = populate_node_features(
                self.csv_path,
                store,
                max_rows=4,
                batch_rows=2,
                max_temp_bytes=8 * 1024 * 1024,
            )
            rows = list(features.iter_rows(batch_rows=2))
            self.assertEqual(store.rows_read, 4)
            self.assertEqual(features.row_count, 4)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["num_sent"], 3)
            with self.assertRaisesRegex(ValueError, "batch_rows"):
                list(features.iter_rows(batch_rows=0))

    def test_temporary_storage_limit_is_explicit(self):
        with open_bounded_disk_prototype(
            self.csv_path,
            max_rows=len(self.frame),
            chunk_rows=2,
            temp_root=self.root,
        ) as store:
            with self.assertRaises(TemporaryStorageLimitExceeded):
                populate_node_features(
                    self.csv_path,
                    store,
                    max_rows=len(self.frame),
                    batch_rows=2,
                    max_temp_bytes=1,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)