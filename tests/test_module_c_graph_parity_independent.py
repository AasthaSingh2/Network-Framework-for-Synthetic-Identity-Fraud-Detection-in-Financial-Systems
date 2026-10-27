"""Independent tiny-fixture assertions for Module C graph parity."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import test_module_c_graph_parity as parity


FEATURE_ORDER = [
    "total_sent",
    "total_received",
    "num_sent",
    "num_received",
    "avg_sent",
    "avg_received",
    "unique_receivers",
    "unique_senders",
    "net_flow",
    "in_degree",
    "out_degree",
    "pagerank",
    "clustering",
]

COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def make_frame(edges):
    rows = []
    for row_index, (source, target, amount_paid, amount_received, label) in enumerate(edges):
        source_bank, source_account = source
        target_bank, target_account = target
        rows.append(
            {
                "Timestamp": f"2022/09/01 00:{row_index:02d}",
                "From Bank": source_bank,
                "Account": source_account,
                "To Bank": target_bank,
                "Account.1": target_account,
                "Amount Received": amount_received,
                "Receiving Currency": "USD",
                "Amount Paid": amount_paid,
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": label,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def feature_fixture():
    # A -> B twice, B -> C, C -> A. Account order is A, B, C.
    return make_frame(
        [
            ((1, "A"), (2, "B"), 10.0, 9.0, 1),
            ((1, "A"), (2, "B"), 2.0, 2.0, 0),
            ((2, "B"), (3, "C"), 4.0, 5.0, 0),
            ((3, "C"), (1, "A"), 3.0, 3.0, 0),
        ]
    )


def self_loop_fixture(node_count):
    return make_frame(
        [((index, f"N{index}"), (index, f"N{index}"), 1.0, 1.0, index % 2)
         for index in range(1, node_count + 1)]
    )


def notebook_style_masks(node_count, seed):
    """Independent implementation of the notebook's permuted 70/15/15 masks."""
    torch.manual_seed(seed)
    permutation = torch.randperm(node_count)
    train_end = int(0.70 * node_count)
    validation_end = int(0.85 * node_count)
    train = torch.zeros(node_count, dtype=torch.bool)
    validation = torch.zeros(node_count, dtype=torch.bool)
    test = torch.zeros(node_count, dtype=torch.bool)
    train[permutation[:train_end]] = True
    validation[permutation[train_end:validation_end]] = True
    test[permutation[validation_end:]] = True
    return train, validation, test


class IndependentParityTests(unittest.TestCase):
    def test_successful_candidate_pagerank_matches_networkx(self):
        frame = make_frame(
            [
                ((1, "A"), (2, "B"), 1.0, 1.0, 0),
                ((2, "B"), (3, "C"), 1.0, 1.0, 0),
                ((3, "C"), (1, "A"), 1.0, 1.0, 0),
                ((3, "C"), (4, "D"), 1.0, 1.0, 0),
            ]
        )
        result = parity.sparse_candidate(frame, seed=69)
        reference_graph = parity.nx.DiGraph()
        reference_graph.add_edges_from(
            zip(result["transactions"]["sender_id"],
                result["transactions"]["receiver_id"])
        )
        expected = parity.nx.pagerank(reference_graph, max_iter=200, tol=1e-4)
        actual = result["features"].set_index("account")["pagerank"]
        np.testing.assert_allclose(
            actual.loc[list(expected)].to_numpy(dtype=np.float64),
            np.asarray([expected[node] for node in expected]),
            rtol=1e-9,
            atol=1e-10,
        )

    def test_exact_feature_order_and_hand_calculated_raw_values(self):
        result = parity.sparse_candidate(feature_fixture(), seed=61)
        self.assertEqual(
            parity.FEATURE_COLUMNS,
            FEATURE_ORDER,
            "The candidate tensor's selected feature order must remain exact.",
        )

        expected_accounts = ["1_A", "2_B", "3_C"]
        expected_raw = np.array(
            [
                [12.0, 3.0, 2.0, 1.0, 6.0, 3.0, 1.0, 1.0, -9.0,
                 1.0, 1.0, 1.0 / 3.0, 1.0],
                [4.0, 11.0, 1.0, 2.0, 4.0, 5.5, 1.0, 1.0, 7.0,
                 1.0, 1.0, 1.0 / 3.0, 1.0],
                [3.0, 5.0, 1.0, 1.0, 3.0, 5.0, 1.0, 1.0, 2.0,
                 1.0, 1.0, 1.0 / 3.0, 1.0],
            ],
            dtype=np.float64,
        )
        actual_accounts = result["accounts"].tolist()
        self.assertEqual(actual_accounts, expected_accounts)
        actual_raw = result["features"].set_index("account").loc[
            expected_accounts, FEATURE_ORDER
        ].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(actual_raw, expected_raw, rtol=0, atol=1e-12)

        # Fit expected scaling only from the hand-calculated values, not from
        # the candidate's feature frame or scaler instance. The notebook first
        # converts the selected matrix to torch.float (float32).
        independent_scaler = StandardScaler()
        expected_tensor_input = expected_raw.astype(np.float32)
        independently_scaled = independent_scaler.fit_transform(expected_tensor_input)
        np.testing.assert_allclose(
            result["scaler"].mean_, independent_scaler.mean_, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            result["scaler"].scale_, independent_scaler.scale_, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            result["data"].x.numpy(),
            independently_scaled.astype(np.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(result["data"].y.tolist(), [1, 1, 0])

    def test_masks_match_separately_implemented_notebook_algorithm(self):
        seed = 67
        result = parity.sparse_candidate(self_loop_fixture(20), seed=seed)
        expected = notebook_style_masks(result["data"].num_nodes, seed)
        actual = (
            result["data"].train_mask,
            result["data"].val_mask,
            result["data"].test_mask,
        )
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        self.assertEqual([int(mask.sum()) for mask in actual], [14, 3, 3])
        self.assertFalse(torch.any(actual[0] & actual[1]).item())
        self.assertFalse(torch.any(actual[0] & actual[2]).item())
        self.assertFalse(torch.any(actual[1] & actual[2]).item())
        self.assertTrue(torch.all(actual[0] | actual[1] | actual[2]).item())

    def test_actual_candidate_nonconvergence_uses_exact_in_degree_fallback(self):
        frame = make_frame(
            [
                ((1, "A"), (2, "B"), 1.0, 1.0, 0),
                ((2, "B"), (3, "C"), 1.0, 1.0, 0),
                ((3, "C"), (4, "D"), 1.0, 1.0, 0),
            ]
        )

        # Force the candidate's actual convergence criterion to remain above
        # tolerance on every iteration. The graph and PageRank loop still run.
        with patch.object(
            parity.np,
            "abs",
            side_effect=lambda values: np.ones_like(values),
        ) as forced_nonconvergence:
            result = parity.sparse_candidate(frame, seed=71)

        self.assertEqual(forced_nonconvergence.call_count, 200)
        ranks = result["features"].set_index("account")["pagerank"]
        expected_fallback = np.array([0.0, 0.25, 0.25, 0.25])
        candidate_order = ["1_A", "2_B", "3_C", "4_D"]
        np.testing.assert_array_equal(
            ranks.loc[candidate_order].to_numpy(dtype=np.float64),
            expected_fallback,
        )

    def test_empty_input_failure_difference_is_explicit(self):
        empty = pd.DataFrame(columns=COLUMNS)
        with self.assertRaises(TypeError):
            parity.notebook_reference(empty, seed=73)
        with self.assertRaises(ZeroDivisionError):
            parity.sparse_candidate(empty, seed=73)


if __name__ == "__main__":
    unittest.main(verbosity=2)