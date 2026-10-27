"""Tiny in-memory parity tests for the isolated Module C graph builder."""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import module_c_graph_builder as candidate_builder


FEATURE_ORDER = [
    "total_sent", "total_received", "num_sent", "num_received",
    "avg_sent", "avg_received", "unique_receivers", "unique_senders",
    "net_flow", "in_degree", "out_degree", "pagerank", "clustering",
]

INPUT_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]


def tiny_frame():
    # Includes repeated transactions, a laundering edge, triangle, wedge,
    # reciprocal pair, disconnected self-loop, and a dangling receiver.
    rows = [
        (1, "A", 2, "B", 10.0, 9.0, 1),
        (1, "A", 2, "B", 10.0, 9.0, 1),
        (2, "B", 3, "C", 4.0, 5.0, 0),
        (3, "C", 1, "A", 3.0, 3.0, 0),
        (3, "C", 4, "D", 7.0, 6.0, 0),
        (5, "E", 6, "F", 2.0, 2.0, 0),
        (6, "F", 5, "E", 8.0, 8.0, 0),
        (7, "G", 7, "G", 1.0, 1.0, 0),
        (8, "H", 9, "I", 5.0, 5.0, 0),
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
    return pd.DataFrame(records, columns=INPUT_COLUMNS)


def notebook_reference(frame, seed):
    """Independent tiny reference following the notebook's pandas/NetworkX path."""
    transactions = frame.copy()
    transactions["sender_id"] = (
        transactions["From Bank"].astype(str)
        + "_"
        + transactions["Account"].astype(str)
    )
    transactions["receiver_id"] = (
        transactions["To Bank"].astype(str)
        + "_"
        + transactions["Account.1"].astype(str)
    )
    accounts = pd.concat(
        [transactions["sender_id"], transactions["receiver_id"]]
    ).unique()
    account_to_idx = {account: index for index, account in enumerate(accounts)}

    sent = transactions.groupby("sender_id").agg(
        total_sent=("Amount Paid", "sum"),
        num_sent=("Amount Paid", "count"),
        unique_receivers=("receiver_id", "nunique"),
    ).reset_index().rename(columns={"sender_id": "account"})
    received = transactions.groupby("receiver_id").agg(
        total_received=("Amount Received", "sum"),
        num_received=("Amount Received", "count"),
        unique_senders=("sender_id", "nunique"),
    ).reset_index().rename(columns={"receiver_id": "account"})
    features = pd.DataFrame({"account": accounts})
    features = features.merge(sent, on="account", how="left")
    features = features.merge(received, on="account", how="left")
    features = features.fillna(0)
    features["avg_sent"] = features["total_sent"] / features["num_sent"].replace(0, 1)
    features["avg_received"] = (
        features["total_received"] / features["num_received"].replace(0, 1)
    )
    features["net_flow"] = features["total_received"] - features["total_sent"]

    graph = nx.DiGraph()
    graph.add_edges_from(zip(transactions["sender_id"], transactions["receiver_id"]))
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    try:
        pagerank = nx.pagerank(graph, max_iter=200, tol=1e-4)
    except Exception:
        pagerank = {
            node: graph.in_degree(node) / max(graph.number_of_nodes(), 1)
            for node in graph.nodes()
        }
    clustering = nx.clustering(graph.to_undirected())
    features["in_degree"] = features["account"].map(in_degree).fillna(0)
    features["out_degree"] = features["account"].map(out_degree).fillna(0)
    features["pagerank"] = features["account"].map(pagerank).fillna(0)
    features["clustering"] = features["account"].map(clustering).fillna(0)

    sent_fraud = transactions.groupby("sender_id")["Is Laundering"].sum().reset_index()
    sent_fraud = sent_fraud.rename(
        columns={"sender_id": "account", "Is Laundering": "fraud_sent"}
    )
    received_fraud = (
        transactions.groupby("receiver_id")["Is Laundering"].sum().reset_index()
    )
    received_fraud = received_fraud.rename(
        columns={"receiver_id": "account", "Is Laundering": "fraud_received"}
    )
    labels = pd.DataFrame({"account": accounts})
    labels = labels.merge(sent_fraud, on="account", how="left")
    labels = labels.merge(received_fraud, on="account", how="left")
    labels = labels.fillna(0)
    labels["total_fraud"] = labels["fraud_sent"] + labels["fraud_received"]
    y = torch.tensor((labels["total_fraud"] > 0).astype(int).values, dtype=torch.long)

    x = torch.tensor(features[FEATURE_ORDER].values, dtype=torch.float)
    scaler = StandardScaler()
    x_scaled = torch.tensor(scaler.fit_transform(x.numpy()), dtype=torch.float)
    source = transactions["sender_id"].map(account_to_idx).to_numpy(dtype=np.int64)
    target = transactions["receiver_id"].map(account_to_idx).to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.stack([source, target]), dtype=torch.long)
    edge_attr = torch.tensor(
        transactions["Amount Paid"].fillna(0).to_numpy(), dtype=torch.float
    ).unsqueeze(1)

    torch.manual_seed(seed)
    permutation = torch.randperm(len(accounts))
    train_end = int(0.70 * len(accounts))
    validation_end = int(0.85 * len(accounts))
    train_mask = torch.zeros(len(accounts), dtype=torch.bool)
    validation_mask = torch.zeros(len(accounts), dtype=torch.bool)
    test_mask = torch.zeros(len(accounts), dtype=torch.bool)
    train_mask[permutation[:train_end]] = True
    validation_mask[permutation[train_end:validation_end]] = True
    test_mask[permutation[validation_end:]] = True
    data = Data(
        x=x_scaled,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=validation_mask,
        test_mask=test_mask,
    )
    return {
        "data": data,
        "accounts": accounts,
        "account_to_idx": account_to_idx,
        "features": features,
        "labels": labels,
        "scaler": scaler,
    }


class ModuleCGraphBuilderTests(unittest.TestCase):
    def test_candidate_matches_notebook_reference_on_tiny_fixture(self):
        frame = tiny_frame()
        seed = 101
        expected = notebook_reference(frame, seed)
        actual = candidate_builder.build_graph(frame, seed=seed)

        self.assertEqual(actual["accounts"].tolist(), expected["accounts"].tolist())
        self.assertEqual(actual["account_to_idx"], expected["account_to_idx"])
        self.assertEqual(candidate_builder.FEATURE_COLUMNS, FEATURE_ORDER)
        np.testing.assert_allclose(
            actual["features"][FEATURE_ORDER].to_numpy(dtype=np.float64),
            expected["features"][FEATURE_ORDER].to_numpy(dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual["scaler"].mean_, expected["scaler"].mean_, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            actual["scaler"].scale_, expected["scaler"].scale_, rtol=1e-12, atol=1e-12
        )
        self.assertIsInstance(actual["data"], Data)
        for field in ("x", "edge_index", "edge_attr", "y", "train_mask", "val_mask", "test_mask"):
            torch.testing.assert_close(
                getattr(actual["data"], field), getattr(expected["data"], field),
                rtol=1e-6 if getattr(actual["data"], field).is_floating_point() else 0,
                atol=1e-7 if getattr(actual["data"], field).is_floating_point() else 0,
            )

    def test_duplicate_rows_are_preserved_in_aggregates_labels_and_edges(self):
        frame = tiny_frame()
        result = candidate_builder.build_graph(frame, seed=103)
        features = result["features"].set_index("account")
        labels = result["labels"].set_index("account")
        sender = "1_A"
        receiver = "2_B"
        self.assertEqual(features.loc[sender, "num_sent"], 2)
        self.assertEqual(features.loc[sender, "total_sent"], 20.0)
        self.assertEqual(labels.loc[sender, "total_fraud"], 2)
        self.assertEqual(labels.loc[receiver, "total_fraud"], 2)
        self.assertEqual(result["data"].edge_index.shape[1], len(frame))
        self.assertEqual(result["data"].edge_attr.shape, (len(frame), 1))

    def test_clustering_triangle_wedge_bridge_reciprocal_and_self_loop(self):
        result = candidate_builder.build_graph(tiny_frame(), seed=107)
        features = result["features"].set_index("account")
        self.assertAlmostEqual(features.loc["1_A", "clustering"], 1.0)
        self.assertAlmostEqual(features.loc["2_B", "clustering"], 1.0)
        self.assertAlmostEqual(features.loc["3_C", "clustering"], 1.0 / 3.0)
        self.assertEqual(features.loc["4_D", "clustering"], 0.0)
        self.assertEqual(features.loc["7_G", "clustering"], 0.0)
        self.assertEqual(features.loc["7_G", "in_degree"], 1)
        self.assertEqual(features.loc["7_G", "out_degree"], 1)

    def test_page_rank_nonconvergence_uses_notebook_in_degree_fallback(self):
        frame = pd.DataFrame(
            [
                {"From Bank": 1, "Account": "A", "To Bank": 2, "Account.1": "B",
                 "Amount Paid": 1.0, "Amount Received": 1.0, "Is Laundering": 0},
                {"From Bank": 2, "Account": "B", "To Bank": 3, "Account.1": "C",
                 "Amount Paid": 1.0, "Amount Received": 1.0, "Is Laundering": 0},
                {"From Bank": 3, "Account": "C", "To Bank": 4, "Account.1": "D",
                 "Amount Paid": 1.0, "Amount Received": 1.0, "Is Laundering": 0},
            ]
        )
        with patch.object(candidate_builder, "_pagerank_has_converged", return_value=False) as convergence:
            result = candidate_builder.build_graph(frame, seed=109)
        self.assertEqual(convergence.call_count, 200)
        ranks = result["features"].set_index("account")["pagerank"]
        np.testing.assert_array_equal(
            ranks.loc[["1_A", "2_B", "3_C", "4_D"]].to_numpy(dtype=np.float64),
            np.array([0.0, 0.25, 0.25, 0.25]),
        )

    def test_page_rank_calculation_exception_uses_notebook_in_degree_fallback(self):
        frame = pd.DataFrame(
            [
                {"From Bank": 1, "Account": "A", "To Bank": 2, "Account.1": "B",
                 "Amount Paid": 1.0, "Amount Received": 1.0, "Is Laundering": 0},
                {"From Bank": 2, "Account": "B", "To Bank": 3, "Account.1": "C",
                 "Amount Paid": 1.0, "Amount Received": 1.0, "Is Laundering": 0},
                {"From Bank": 3, "Account": "C", "To Bank": 4, "Account.1": "D",
                 "Amount Paid": 1.0, "Amount Received": 1.0, "Is Laundering": 0},
            ]
        )
        with patch.object(
            candidate_builder,
            "_pagerank_has_converged",
            side_effect=RuntimeError("synthetic PageRank calculation failure"),
        ):
            result = candidate_builder.build_graph(frame, seed=111)
        ranks = result["features"].set_index("account")["pagerank"]
        np.testing.assert_array_equal(
            ranks.loc[["1_A", "2_B", "3_C", "4_D"]].to_numpy(dtype=np.float64),
            np.array([0.0, 0.25, 0.25, 0.25]),
        )

    def test_empty_pagerank_graph_behavior_is_explicit(self):
        empty_topology = candidate_builder.coo_matrix((0, 0)).tocsr()
        with self.assertRaisesRegex(ValueError, "at least one node"):
            candidate_builder._sparse_pagerank(
                empty_topology, np.array([], dtype=np.int64)
            )
        self.assertEqual(nx.pagerank(nx.DiGraph()), {})

    def test_nonconvergence_fallback_respects_deduplicated_topology(self):
        frame = pd.DataFrame(
            [
                {"From Bank": 1, "Account": "A", "To Bank": 2, "Account.1": "B",
                 "Amount Paid": 2.0, "Amount Received": 2.0, "Is Laundering": 0},
                {"From Bank": 1, "Account": "A", "To Bank": 2, "Account.1": "B",
                 "Amount Paid": 3.0, "Amount Received": 3.0, "Is Laundering": 0},
            ]
        )
        with patch.object(candidate_builder, "_pagerank_has_converged", return_value=False):
            result = candidate_builder.build_graph(frame, seed=113)
        ranks = result["features"].set_index("account")["pagerank"]
        self.assertEqual(ranks.loc["1_A"], 0.0)
        self.assertEqual(ranks.loc["2_B"], 0.5)
        self.assertEqual(result["data"].edge_index.shape[1], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)