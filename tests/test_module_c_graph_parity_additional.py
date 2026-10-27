"""Additional isolated parity tests using only tiny in-memory fixtures."""

import io
import pickle
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from test_module_c_graph_parity import notebook_reference, sparse_candidate


ORDERED_FEATURES = [
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

CSV_COLUMNS = [
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


def rows_for_edges(edges):
    """Build tiny IBM-shaped records; edge tuples are (src, dst, label, amount)."""
    rows = []
    for index, (source, target, label, amount) in enumerate(edges):
        source_bank, source_account = source
        target_bank, target_account = target
        rows.append(
            {
                "Timestamp": f"2022/09/01 00:{index:02d}",
                "From Bank": source_bank,
                "Account": source_account,
                "To Bank": target_bank,
                "Account.1": target_account,
                "Amount Received": float(amount),
                "Receiving Currency": "USD",
                "Amount Paid": float(amount),
                "Payment Currency": "USD",
                "Payment Format": "Wire",
                "Is Laundering": int(label),
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def topology_fixture():
    # Triangle A-B-C, wedge C-D, bridge E-F with reciprocal edges,
    # a disconnected self-loop G, and H -> I where I is a dangling node.
    return rows_for_edges(
        [
            ((1, "A"), (2, "B"), 1, 3.0),
            ((2, "B"), (3, "C"), 0, 5.0),
            ((3, "C"), (1, "A"), 0, 7.0),
            ((3, "C"), (4, "D"), 0, 11.0),
            ((5, "E"), (6, "F"), 0, 13.0),
            ((6, "F"), (5, "E"), 0, 17.0),
            ((7, "G"), (7, "G"), 0, 19.0),
            ((8, "H"), (9, "I"), 0, 23.0),
        ]
    )


def one_node_fixture(label):
    return rows_for_edges([((1, "only"), (1, "only"), label, 1.0)])


def notebook_pagerank_with_fallback(graph):
    """Mirror the notebook's try/except around NetworkX PageRank."""
    try:
        return nx.pagerank(graph, max_iter=200, tol=1e-4)
    except Exception:
        node_count = max(graph.number_of_nodes(), 1)
        return {
            node: graph.in_degree(node) / node_count
            for node in graph.nodes()
        }


class AdditionalGraphParityTests(unittest.TestCase):
    def test_duplicate_row_candidate_parity_with_and_without_duplicate(self):
        duplicated = rows_for_edges(
            [
                ((1, "S"), (2, "R"), 1, 10.0),
                ((1, "S"), (2, "R"), 1, 10.0),
                ((1, "S"), (2, "R"), 0, 5.0),
                ((3, "T"), (1, "S"), 0, 7.0),
            ]
        )
        unique_rows = duplicated.drop(index=1).reset_index(drop=True)

        for frame in (duplicated, unique_rows):
            with self.subTest(transaction_rows=len(frame)):
                reference = notebook_reference(frame, seed=17)
                candidate = sparse_candidate(frame, seed=17)
                self.assertEqual(candidate["account_to_idx"], reference["account_to_idx"])
                self.assertEqual(candidate["data"].edge_index.shape[1], len(frame))
                np.testing.assert_array_equal(
                    candidate["data"].edge_index.numpy(),
                    reference["data"].edge_index.numpy(),
                )
                np.testing.assert_array_equal(
                    candidate["data"].edge_attr.numpy(),
                    reference["data"].edge_attr.numpy(),
                )
                np.testing.assert_allclose(
                    candidate["features"][ORDERED_FEATURES].to_numpy(dtype=float),
                    reference["features"][ORDERED_FEATURES].to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-10,
                )
                np.testing.assert_array_equal(
                    candidate["labels"]["total_fraud"].to_numpy(),
                    reference["labels"]["total_fraud"].to_numpy(),
                )
                torch.testing.assert_close(
                    candidate["data"].y, reference["data"].y, rtol=0, atol=0
                )

        duplicated_result = sparse_candidate(duplicated, seed=17)
        unique_result = sparse_candidate(unique_rows, seed=17)
        dup_features = duplicated_result["features"].set_index("account")
        unique_features = unique_result["features"].set_index("account")
        self.assertEqual(dup_features.loc["1_S", "num_sent"], 3)
        self.assertEqual(unique_features.loc["1_S", "num_sent"], 2)
        self.assertEqual(dup_features.loc["1_S", "total_sent"], 25.0)
        self.assertEqual(unique_features.loc["1_S", "total_sent"], 15.0)
        self.assertEqual(
            duplicated_result["data"].edge_index.shape[1]
            - unique_result["data"].edge_index.shape[1],
            1,
        )

    def test_triangle_wedge_bridge_reciprocal_and_self_loop_clustering(self):
        frame = topology_fixture()
        reference = notebook_reference(frame, seed=23)
        candidate = sparse_candidate(frame, seed=23)
        reference_features = reference["features"].set_index("account")
        candidate_features = candidate["features"].set_index("account")

        np.testing.assert_allclose(
            candidate_features["clustering"].to_numpy(),
            reference_features["clustering"].to_numpy(),
            rtol=0,
            atol=1e-15,
        )
        expected = {
            "1_A": 1.0,
            "2_B": 1.0,
            "3_C": 1.0 / 3.0,
            "4_D": 0.0,
            "5_E": 0.0,
            "6_F": 0.0,
            "7_G": 0.0,
            "8_H": 0.0,
            "9_I": 0.0,
        }
        for account, value in expected.items():
            self.assertAlmostEqual(
                candidate_features.loc[account, "clustering"], value, places=14
            )

    def test_pagerank_across_components_and_dangling_nodes(self):
        reference = notebook_reference(topology_fixture(), seed=29)
        candidate = sparse_candidate(topology_fixture(), seed=29)
        reference_features = reference["features"].set_index("account")
        candidate_features = candidate["features"].set_index("account")
        expected_graph = nx.DiGraph()
        expected_graph.add_edges_from(
            zip(reference["transactions"]["sender_id"],
                reference["transactions"]["receiver_id"])
        )
        expected_rank = nx.pagerank(expected_graph, max_iter=200, tol=1e-4)
        actual = candidate_features["pagerank"]

        self.assertAlmostEqual(float(actual.sum()), 1.0, places=12)
        for account, value in expected_rank.items():
            self.assertAlmostEqual(actual.loc[account], value, delta=1e-10)
            self.assertAlmostEqual(
                reference_features.loc[account, "pagerank"], value, delta=1e-10
            )
        self.assertGreater(actual.loc["9_I"], 0.0)
        self.assertNotEqual(actual.loc["1_A"], actual.loc["5_E"])

    def test_notebook_pagerank_nonconvergence_fallback_definition(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        failure = nx.PowerIterationFailedConvergence(200)
        with patch("networkx.pagerank", side_effect=failure):
            fallback = notebook_pagerank_with_fallback(graph)
        self.assertEqual(fallback, {"a": 0.0, "b": 0.25, "c": 0.25, "d": 0.25})

    def test_empty_input_fails_instead_of_producing_empty_graph(self):
        empty = pd.DataFrame(columns=CSV_COLUMNS)
        with self.assertRaises(TypeError):
            notebook_reference(empty, seed=37)
        with self.assertRaises(ZeroDivisionError):
            sparse_candidate(empty, seed=37)

    def test_one_node_all_negative_and_all_positive_are_graph_valid_but_split_degenerate(self):
        for label in (0, 1):
            with self.subTest(label=label):
                result = sparse_candidate(one_node_fixture(label), seed=41)
                data = result["data"]
                self.assertEqual(data.num_nodes, 1)
                self.assertEqual(data.y.tolist(), [label])
                self.assertEqual(int(data.train_mask.sum()), 0)
                self.assertEqual(int(data.val_mask.sum()), 0)
                self.assertEqual(int(data.test_mask.sum()), 1)
                self.assertTrue(torch.isfinite(data.x).all().item())

    def test_isolated_nodes_cannot_be_emitted_by_endpoint_derived_node_set(self):
        result = sparse_candidate(topology_fixture(), seed=43)
        data = result["data"]
        incident = torch.zeros(data.num_nodes, dtype=torch.bool)
        incident[data.edge_index.reshape(-1)] = True
        self.assertTrue(incident.all().item())
        self.assertEqual(data.num_nodes, len(result["account_to_idx"]))

    def test_exact_feature_order_and_independent_scaler_expectation(self):
        result = sparse_candidate(topology_fixture(), seed=47)
        feature_frame = result["features"]
        self.assertEqual(
            ORDERED_FEATURES,
            [
                "total_sent", "total_received", "num_sent", "num_received",
                "avg_sent", "avg_received", "unique_receivers", "unique_senders",
                "net_flow", "in_degree", "out_degree", "pagerank", "clustering",
            ],
        )
        expected_raw = feature_frame[ORDERED_FEATURES].to_numpy(dtype=np.float32)
        independent_scaler = StandardScaler()
        expected_scaled = independent_scaler.fit_transform(expected_raw)
        np.testing.assert_allclose(
            result["scaler"].mean_, independent_scaler.mean_, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            result["scaler"].scale_, independent_scaler.scale_, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            result["data"].x.numpy(), expected_scaled.astype(np.float32),
            rtol=1e-6, atol=1e-7,
        )

    def test_masks_are_disjoint_exhaustive_and_have_expected_counts(self):
        edges = [((i, f"N{i}"), (i, f"N{i}"), i % 2, float(i + 1))
                 for i in range(1, 21)]
        result = sparse_candidate(rows_for_edges(edges), seed=53)
        data = result["data"]
        masks = [data.train_mask, data.val_mask, data.test_mask]
        self.assertEqual([int(mask.sum()) for mask in masks], [14, 3, 3])
        self.assertFalse(torch.any(masks[0] & masks[1]).item())
        self.assertFalse(torch.any(masks[0] & masks[2]).item())
        self.assertFalse(torch.any(masks[1] & masks[2]).item())
        self.assertTrue(torch.all(masks[0] | masks[1] | masks[2]).item())

    def test_in_memory_serialization_contracts(self):
        result = sparse_candidate(topology_fixture(), seed=59)

        graph_buffer = io.BytesIO()
        torch.save(result["data"], graph_buffer)
        graph_buffer.seek(0)
        loaded_graph = torch.load(graph_buffer, map_location="cpu", weights_only=False)
        self.assertIsInstance(loaded_graph, Data)
        for field in (
            "x", "edge_index", "edge_attr", "y",
            "train_mask", "val_mask", "test_mask",
        ):
            torch.testing.assert_close(
                getattr(loaded_graph, field), getattr(result["data"], field)
            )

        mapping_buffer = io.BytesIO(pickle.dumps(result["account_to_idx"]))
        loaded_mapping = pickle.load(mapping_buffer)
        self.assertEqual(loaded_mapping, result["account_to_idx"])

        scaler_buffer = io.BytesIO()
        joblib.dump(result["scaler"], scaler_buffer)
        scaler_buffer.seek(0)
        loaded_scaler = joblib.load(scaler_buffer)
        np.testing.assert_array_equal(loaded_scaler.mean_, result["scaler"].mean_)
        np.testing.assert_array_equal(loaded_scaler.scale_, result["scaler"].scale_)


if __name__ == "__main__":
    unittest.main(verbosity=2)