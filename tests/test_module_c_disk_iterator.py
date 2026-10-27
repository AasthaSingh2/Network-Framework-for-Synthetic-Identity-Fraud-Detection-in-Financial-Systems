"""Synthetic tests for bounded iteration over Module C SQLite storage."""

import sys
import tempfile
import unittest
from itertools import islice
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from module_c_disk_iterator import open_bounded_disk_prototype
from test_module_c_disk_prototype import independent_expected, synthetic_rows


def collect_in_batches(iterator, batch_size):
    batches = []
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        if not batch:
            break
        batches.append(batch)
    return batches


class DiskIteratorTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = tempfile.TemporaryDirectory(prefix="module-c-iterator-test-")
        self.addCleanup(self.temp_root.cleanup)
        self.root = Path(self.temp_root.name)
        self.csv_path = self.root / "synthetic.csv"
        self.frame = synthetic_rows()
        self.frame.to_csv(self.csv_path, index=False)

    def test_multi_batch_reads_preserve_order_duplicates_topology_and_mapping(self):
        expected = independent_expected(self.frame, max_rows=len(self.frame))
        with open_bounded_disk_prototype(
            self.csv_path,
            max_rows=len(self.frame),
            chunk_rows=2,
            temp_root=self.root,
        ) as store:
            workspace = Path(store.temporary_directory)
            self.assertTrue(workspace.exists())
            self.assertEqual(store.rows_read, len(self.frame))

            transaction_batches = collect_in_batches(
                store.iter_transaction_edges(batch_size=2), 2
            )
            topology_batches = collect_in_batches(
                store.iter_topology_edges(batch_size=2), 2
            )
            account_batches = collect_in_batches(store.iter_accounts(batch_size=2), 2)

            transaction_edges = [row for batch in transaction_batches for row in batch]
            topology_edges = [row for batch in topology_batches for row in batch]
            accounts = [row for batch in account_batches for row in batch]

            self.assertGreater(len(transaction_batches), 1)
            self.assertGreater(len(topology_batches), 1)
            self.assertGreater(len(account_batches), 1)
            self.assertEqual(transaction_edges, expected["transaction_edges"])
            self.assertEqual(topology_edges, expected["topology_edges"])
            self.assertEqual(
                [(key, index) for key, index in accounts],
                list(expected["account_to_idx"].items()),
            )
            self.assertEqual(store.account_index("1_A"), expected["account_to_idx"]["1_A"])
            self.assertIsNone(store.account_index("not-present"))
            self.assertEqual(
                dict(store.iter_sender_counts(batch_size=2)), expected["sender_counts"]
            )
            self.assertEqual(
                dict(store.iter_receiver_counts(batch_size=2)), expected["receiver_counts"]
            )

            # Duplicated A->X input rows remain separate, while topology is unique.
            self.assertEqual(transaction_edges[0][1:3], transaction_edges[2][1:3])
            self.assertEqual(len(transaction_edges), len(self.frame))
            self.assertEqual(len(topology_edges), 5)
            self.assertIn((3, 3), topology_edges)  # D->D self-loop retained.
            self.assertLess(
                len(topology_edges),
                len(transaction_edges),
            )

        self.assertFalse(workspace.exists())

    def test_temporary_workspace_is_removed_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failure"):
            with open_bounded_disk_prototype(
                self.csv_path,
                max_rows=len(self.frame),
                chunk_rows=2,
                temp_root=self.root,
            ) as store:
                workspace = Path(store.temporary_directory)
                first_batch = list(
                    islice(store.iter_transaction_edges(batch_size=2), 2)
                )
                self.assertEqual(len(first_batch), 2)
                self.assertTrue(workspace.exists())
                raise RuntimeError("consumer failure")
        self.assertFalse(workspace.exists())

    def test_iterator_rejects_nonpositive_batch_size(self):
        with open_bounded_disk_prototype(
            self.csv_path,
            max_rows=len(self.frame),
            chunk_rows=2,
            temp_root=self.root,
        ) as store:
            with self.assertRaisesRegex(ValueError, "batch_size"):
                next(store.iter_transaction_edges(batch_size=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)