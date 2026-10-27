"""Synthetic-only tests for the bounded disk-backed Module C prototype."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import module_c_disk_prototype as disk_prototype
from module_c_graph_builder import build_graph


def synthetic_rows():
    columns = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering",
    ]
    rows = [
        (1, "A", 2, "X", 10.0, 9.0, 1),
        (2, "B", 1, "A", 3.0, 3.0, 0),
        (1, "A", 2, "X", 10.0, 9.0, 1),
        (3, "C", 5, "Z", 4.0, 4.0, 0),
        (4, "D", 4, "D", 2.0, 2.0, 0),
        (1, "A", 5, "Z", 6.0, 6.0, 0),
    ]
    records = []
    for row_number, (from_bank, account, to_bank, account_to, paid, received, label) in enumerate(rows):
        records.append(
            {
                "Timestamp": f"2022/09/01 00:{row_number:02d}",
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


def independent_expected(frame, max_rows):
    bounded = frame.iloc[:max_rows].reset_index(drop=True).copy()
    bounded["sender_key"] = (
        bounded["From Bank"].astype(str) + "_" + bounded["Account"].astype(str)
    )
    bounded["receiver_key"] = (
        bounded["To Bank"].astype(str) + "_" + bounded["Account.1"].astype(str)
    )

    ordered_accounts = list(dict.fromkeys(bounded["sender_key"].tolist()))
    sender_keys = set(ordered_accounts)
    for receiver in bounded["receiver_key"]:
        if receiver not in sender_keys:
            sender_keys.add(receiver)
            ordered_accounts.append(receiver)
    mapping = {key: index for index, key in enumerate(ordered_accounts)}

    tx_edges = []
    topology_edges = set()
    sender_counterparts = {}
    receiver_counterparts = {}
    for row_index, row in bounded.iterrows():
        source = mapping[row["sender_key"]]
        target = mapping[row["receiver_key"]]
        tx_edges.append(
            (row_index, source, target, float(row["Amount Paid"]), int(row["Is Laundering"]))
        )
        topology_edges.add((source, target))
        sender_counterparts.setdefault(source, set()).add(target)
        receiver_counterparts.setdefault(target, set()).add(source)

    return {
        "accounts": ordered_accounts,
        "account_to_idx": mapping,
        "transaction_edges": tx_edges,
        "topology_edges": sorted(topology_edges),
        "sender_counts": {
            index: len(counterparts)
            for index, counterparts in sender_counterparts.items()
        },
        "receiver_counts": {
            index: len(counterparts)
            for index, counterparts in receiver_counterparts.items()
        },
    }


class DiskPrototypeTests(unittest.TestCase):
    def test_two_pass_order_edges_counts_and_topology_across_chunks(self):
        frame = synthetic_rows()
        expected = independent_expected(frame, max_rows=6)
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-temp-root-") as temp_root:
            csv_path = Path(temp_root) / "fixture.csv"
            frame.to_csv(csv_path, index=False)
            result = disk_prototype.build_bounded_disk_prototype(
                csv_path,
                max_rows=6,
                chunk_rows=2,
                temp_root=temp_root,
            )

        self.assertEqual(result["accounts"], expected["accounts"])
        self.assertEqual(result["account_to_idx"], expected["account_to_idx"])
        self.assertEqual(result["transaction_edges"], expected["transaction_edges"])
        self.assertEqual(result["topology_edges"], expected["topology_edges"])
        self.assertEqual(
            result["sender_distinct_receiver_counts"], expected["sender_counts"]
        )
        self.assertEqual(
            result["receiver_distinct_sender_counts"], expected["receiver_counts"]
        )
        self.assertEqual(result["rows_read"], 6)
        self.assertEqual(result["max_rows"], 6)

        mapping = result["account_to_idx"]
        self.assertEqual(result["accounts"], ["1_A", "2_B", "3_C", "4_D", "2_X", "5_Z"])
        self.assertEqual(len(result["transaction_edges"]), 6)
        self.assertEqual(
            result["transaction_edges"][0][1:],
            result["transaction_edges"][2][1:],
        )
        self.assertEqual(mapping["4_D"], result["transaction_edges"][4][1])
        self.assertEqual(result["transaction_edges"][4][1], result["transaction_edges"][4][2])
        self.assertEqual(len(result["topology_edges"]), 5)
        self.assertEqual(result["sender_distinct_receiver_counts"][mapping["1_A"]], 2)
        self.assertEqual(result["receiver_distinct_sender_counts"][mapping["2_X"]], 1)

        # Existing helper confirms the same capped transaction semantics.
        helper_result = build_graph(frame.iloc[:6].reset_index(drop=True), seed=137)
        helper_mapping = helper_result["account_to_idx"]
        helper_edges = [
            (row, int(source), int(target), float(amount), int(label))
            for row, (source, target, amount, label) in enumerate(
                zip(
                    helper_result["data"].edge_index[0].tolist(),
                    helper_result["data"].edge_index[1].tolist(),
                    helper_result["data"].edge_attr[:, 0].tolist(),
                    frame["Is Laundering"].tolist(),
                )
            )
        ]
        self.assertEqual(result["account_to_idx"], helper_mapping)
        self.assertEqual(result["transaction_edges"], helper_edges)
        self.assertEqual(
            result["sender_distinct_receiver_counts"],
            {
                helper_mapping[key]: int(value)
                for key, value in helper_result["features"].set_index("account")[
                    "unique_receivers"
                ].items()
                if value > 0
            },
        )
        self.assertEqual(
            result["receiver_distinct_sender_counts"],
            {
                helper_mapping[key]: int(value)
                for key, value in helper_result["features"].set_index("account")[
                    "unique_senders"
                ].items()
                if value > 0
            },
        )

    def test_row_limit_keeps_only_input_prefix(self):
        frame = synthetic_rows()
        expected = independent_expected(frame, max_rows=4)
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-temp-root-") as temp_root:
            csv_path = Path(temp_root) / "fixture.csv"
            frame.to_csv(csv_path, index=False)
            result = disk_prototype.build_bounded_disk_prototype(
                csv_path,
                max_rows=4,
                chunk_rows=3,
                temp_root=temp_root,
            )
        self.assertEqual(result["rows_read"], 4)
        self.assertEqual(result["transaction_edges"], expected["transaction_edges"])
        self.assertEqual(result["accounts"], expected["accounts"])

    def test_temporary_workspace_is_removed_on_failure(self):
        frame = synthetic_rows()
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-temp-root-") as temp_root:
            csv_path = Path(temp_root) / "fixture.csv"
            frame.to_csv(csv_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "synthetic pilot failure"):
                disk_prototype.build_bounded_disk_prototype(
                    csv_path,
                    max_rows=6,
                    chunk_rows=2,
                    temp_root=temp_root,
                    fail_after_rows=3,
                )
            remaining = list(Path(temp_root).glob("module-c-disk-prototype-*"))
            self.assertEqual(remaining, [])

    def test_invalid_limits_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-temp-root-") as temp_root:
            csv_path = Path(temp_root) / "fixture.csv"
            synthetic_rows().to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "max_rows"):
                disk_prototype.build_bounded_disk_prototype(
                    csv_path, max_rows=0, chunk_rows=2, temp_root=temp_root
                )
            with self.assertRaisesRegex(ValueError, "chunk_rows"):
                disk_prototype.build_bounded_disk_prototype(
                    csv_path, max_rows=2, chunk_rows=0, temp_root=temp_root
                )

    def test_outputs_directory_is_rejected_as_temp_root(self):
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-temp-root-") as temp_root:
            csv_path = Path(temp_root) / "fixture.csv"
            synthetic_rows().to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "outside outputs"):
                disk_prototype.build_bounded_disk_prototype(
                    csv_path,
                    max_rows=2,
                    chunk_rows=1,
                    temp_root=PROJECT_ROOT / "outputs",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)