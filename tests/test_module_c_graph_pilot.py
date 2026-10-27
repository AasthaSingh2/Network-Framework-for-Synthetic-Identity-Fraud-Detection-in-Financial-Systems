"""Synthetic-only tests for the hard-capped Module C CSV pilot."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TESTS_DIR))

import module_c_graph_pilot as pilot
from test_module_c_graph_builder import notebook_reference, tiny_frame


class BoundedGraphPilotTests(unittest.TestCase):
    def _write_synthetic_csv(self, directory):
        frame = tiny_frame()
        path = Path(directory) / "synthetic_transactions.csv"
        frame.to_csv(path, index=False)
        return frame, path

    def _assert_graphs_equal(self, actual, expected):
        self.assertEqual(actual["accounts"].tolist(), expected["accounts"].tolist())
        self.assertEqual(actual["account_to_idx"], expected["account_to_idx"])
        pd.testing.assert_frame_equal(
            actual["features"].reset_index(drop=True),
            expected["features"].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )
        pd.testing.assert_frame_equal(
            actual["labels"].reset_index(drop=True),
            expected["labels"].reset_index(drop=True),
            check_dtype=False,
        )
        for field in (
            "x", "edge_index", "edge_attr", "y",
            "train_mask", "val_mask", "test_mask",
        ):
            torch.testing.assert_close(
                getattr(actual["data"], field),
                getattr(expected["data"], field),
                rtol=1e-6 if getattr(actual["data"], field).is_floating_point() else 0,
                atol=1e-7 if getattr(actual["data"], field).is_floating_point() else 0,
            )

    def test_multiple_chunk_boundaries_preserve_order_duplicates_and_parity(self):
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-input-") as source_dir:
            frame, csv_path = self._write_synthetic_csv(source_dir)
            result = pilot.run_bounded_csv_pilot(
                csv_path,
                max_rows=len(frame),
                chunk_rows=4,
                seed=127,
            )

        self.assertEqual(result["rows_read"], 9)
        self.assertEqual(result["chunk_sizes"], (4, 4, 1))
        self.assertTrue(result["temporary_directory_removed"])
        self.assertFalse(Path(result["temporary_directory"]).exists())
        self.assertNotEqual(
            Path(result["temporary_directory"]).parent,
            PROJECT_ROOT / "outputs",
        )

        expected = notebook_reference(frame, seed=127)
        self._assert_graphs_equal(result["graph_result"], expected)
        self.assertEqual(result["graph_result"]["data"].edge_index.shape[1], len(frame))
        self.assertEqual(result["graph_result"]["features"].loc[0, "num_sent"], 2)

    def test_hard_row_limit_stops_inside_chunk_and_matches_prefix_reference(self):
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-input-") as source_dir:
            frame, csv_path = self._write_synthetic_csv(source_dir)
            max_rows = 7
            with patch.object(pilot.pd, "read_csv", wraps=pd.read_csv) as read_csv:
                result = pilot.run_bounded_csv_pilot(
                    csv_path,
                    max_rows=max_rows,
                    chunk_rows=3,
                    seed=131,
                )

        self.assertEqual(result["rows_read"], max_rows)
        self.assertEqual(result["chunk_sizes"], (3, 3, 1))
        self.assertLessEqual(result["rows_read"], result["max_rows"])
        self.assertTrue(result["temporary_directory_removed"])
        read_csv.assert_called_once()
        self.assertEqual(read_csv.call_args.kwargs["nrows"], max_rows)
        self.assertEqual(read_csv.call_args.kwargs["chunksize"], 3)

        expected = notebook_reference(frame.iloc[:max_rows].reset_index(drop=True), seed=131)
        self._assert_graphs_equal(result["graph_result"], expected)
        self.assertEqual(result["graph_result"]["data"].edge_index.shape[1], max_rows)

    def test_invalid_limits_fail_before_reading_csv(self):
        with tempfile.TemporaryDirectory(prefix="module-c-pilot-input-") as source_dir:
            _, csv_path = self._write_synthetic_csv(source_dir)
            with patch.object(pilot.pd, "read_csv", side_effect=AssertionError("should not read")):
                with self.assertRaisesRegex(ValueError, "max_rows"):
                    pilot.run_bounded_csv_pilot(csv_path, max_rows=0, chunk_rows=2)
                with self.assertRaisesRegex(ValueError, "chunk_rows"):
                    pilot.run_bounded_csv_pilot(csv_path, max_rows=2, chunk_rows=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)