"""Tiny-fixture parity checks for a sparse Module C graph builder.

This file is test-only. It reads no project datasets and writes no artifacts.
The reference path mirrors the graph-construction notebook's transformations.
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, diags
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


FEATURE_COLUMNS = [
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


def tiny_transactions():
    rows = [
        {
            "Timestamp": "2022/09/01 00:00",
            "From Bank": 1,
            "Account": "S",
            "To Bank": 2,
            "Account.1": "R",
            "Amount Received": 10.0,
            "Receiving Currency": "USD",
            "Amount Paid": 10.0,
            "Payment Currency": "USD",
            "Payment Format": "Wire",
            "Is Laundering": 1,
        },
        {
            "Timestamp": "2022/09/01 00:00",
            "From Bank": 1,
            "Account": "S",
            "To Bank": 2,
            "Account.1": "R",
            "Amount Received": 10.0,
            "Receiving Currency": "USD",
            "Amount Paid": 10.0,
            "Payment Currency": "USD",
            "Payment Format": "Wire",
            "Is Laundering": 1,
        },
        {
            "Timestamp": "2022/09/01 00:01",
            "From Bank": 1,
            "Account": "S",
            "To Bank": 2,
            "Account.1": "R",
            "Amount Received": 5.0,
            "Receiving Currency": "USD",
            "Amount Paid": 5.0,
            "Payment Currency": "USD",
            "Payment Format": "ACH",
            "Is Laundering": 0,
        },
        {
            "Timestamp": "2022/09/01 00:02",
            "From Bank": 3,
            "Account": "T",
            "To Bank": 1,
            "Account.1": "S",
            "Amount Received": 7.0,
            "Receiving Currency": "USD",
            "Amount Paid": 7.0,
            "Payment Currency": "USD",
            "Payment Format": "Card",
            "Is Laundering": 0,
        },
        {
            "Timestamp": "2022/09/01 00:03",
            "From Bank": 4,
            "Account": "L",
            "To Bank": 4,
            "Account.1": "L",
            "Amount Received": 2.0,
            "Receiving Currency": "USD",
            "Amount Paid": 2.0,
            "Payment Currency": "USD",
            "Payment Format": "Cash",
            "Is Laundering": 0,
        },
        {
            "Timestamp": "2022/09/01 00:04",
            "From Bank": 2,
            "Account": "R",
            "To Bank": 5,
            "Account.1": "Z",
            "Amount Received": 4.0,
            "Receiving Currency": "USD",
            "Amount Paid": 4.0,
            "Payment Currency": "USD",
            "Payment Format": "ACH",
            "Is Laundering": 0,
        },
    ]
    return pd.DataFrame(rows)


def _assemble_result(transactions, accounts, account_to_idx, features, labels,
                     edge_index, edge_attr, scaler, seed):
    x = torch.tensor(features[FEATURE_COLUMNS].values, dtype=torch.float)
    x_scaled = torch.tensor(scaler.fit_transform(x.numpy()), dtype=torch.float)
    y = torch.tensor((labels["total_fraud"] > 0).astype(int).values,
                     dtype=torch.long)

    num_nodes = x_scaled.shape[0]
    torch.manual_seed(seed)
    indices = torch.randperm(num_nodes)
    train_end = int(0.70 * num_nodes)
    val_end = int(0.85 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data = Data(
        x=x_scaled,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return {
        "transactions": transactions,
        "accounts": accounts,
        "account_to_idx": account_to_idx,
        "features": features,
        "labels": labels,
        "scaler": scaler,
        "data": data,
    }


def notebook_reference(frame, seed=42):
    """Mirror the notebook path, including NetworkX's simple DiGraph."""
    ibm = frame.copy()
    ibm["sender_id"] = (
        ibm["From Bank"].astype(str) + "_" + ibm["Account"].astype(str)
    )
    ibm["receiver_id"] = (
        ibm["To Bank"].astype(str) + "_" + ibm["Account.1"].astype(str)
    )
    accounts = pd.concat([ibm["sender_id"], ibm["receiver_id"]]).unique()
    account_to_idx = {account: idx for idx, account in enumerate(accounts)}

    sent_stats = ibm.groupby("sender_id").agg(
        total_sent=("Amount Paid", "sum"),
        num_sent=("Amount Paid", "count"),
        unique_receivers=("receiver_id", "nunique"),
    ).reset_index().rename(columns={"sender_id": "account"})
    recv_stats = ibm.groupby("receiver_id").agg(
        total_received=("Amount Received", "sum"),
        num_received=("Amount Received", "count"),
        unique_senders=("sender_id", "nunique"),
    ).reset_index().rename(columns={"receiver_id": "account"})

    features = pd.DataFrame({"account": accounts})
    features = features.merge(sent_stats, on="account", how="left")
    features = features.merge(recv_stats, on="account", how="left")
    features = features.fillna(0)
    features["avg_sent"] = features["total_sent"] / features["num_sent"].replace(0, 1)
    features["avg_received"] = (
        features["total_received"] / features["num_received"].replace(0, 1)
    )
    features["net_flow"] = features["total_received"] - features["total_sent"]

    graph = nx.DiGraph()
    graph.add_edges_from(zip(ibm["sender_id"], ibm["receiver_id"]))
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, max_iter=200, tol=1e-4)
    clustering = nx.clustering(graph.to_undirected())
    features["in_degree"] = features["account"].map(in_degree).fillna(0)
    features["out_degree"] = features["account"].map(out_degree).fillna(0)
    features["pagerank"] = features["account"].map(pagerank).fillna(0)
    features["clustering"] = features["account"].map(clustering).fillna(0)

    fraud_sent = ibm.groupby("sender_id")["Is Laundering"].sum().reset_index()
    fraud_sent = fraud_sent.rename(
        columns={"sender_id": "account", "Is Laundering": "fraud_sent"}
    )
    fraud_received = ibm.groupby("receiver_id")["Is Laundering"].sum().reset_index()
    fraud_received = fraud_received.rename(
        columns={"receiver_id": "account", "Is Laundering": "fraud_received"}
    )
    labels = pd.DataFrame({"account": accounts})
    labels = labels.merge(fraud_sent, on="account", how="left")
    labels = labels.merge(fraud_received, on="account", how="left")
    labels = labels.fillna(0)
    labels["total_fraud"] = labels["fraud_sent"] + labels["fraud_received"]

    src_nodes = ibm["sender_id"].map(account_to_idx).values.tolist()
    dst_nodes = ibm["receiver_id"].map(account_to_idx).values.tolist()
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(
        ibm["Amount Paid"].fillna(0).values, dtype=torch.float
    ).unsqueeze(1)

    return _assemble_result(
        ibm, accounts, account_to_idx, features, labels, edge_index,
        edge_attr, StandardScaler(), seed
    )


def sparse_candidate(frame, seed=42):
    """Use sparse adjacency for topology while retaining transaction edges."""
    ibm = frame.copy()
    ibm["sender_id"] = (
        ibm["From Bank"].astype(str) + "_" + ibm["Account"].astype(str)
    )
    ibm["receiver_id"] = (
        ibm["To Bank"].astype(str) + "_" + ibm["Account.1"].astype(str)
    )
    accounts = pd.concat([ibm["sender_id"], ibm["receiver_id"]]).unique()
    account_to_idx = {account: idx for idx, account in enumerate(accounts)}

    sent_stats = ibm.groupby("sender_id").agg(
        total_sent=("Amount Paid", "sum"),
        num_sent=("Amount Paid", "count"),
        unique_receivers=("receiver_id", "nunique"),
    ).reset_index().rename(columns={"sender_id": "account"})
    recv_stats = ibm.groupby("receiver_id").agg(
        total_received=("Amount Received", "sum"),
        num_received=("Amount Received", "count"),
        unique_senders=("sender_id", "nunique"),
    ).reset_index().rename(columns={"receiver_id": "account"})
    features = pd.DataFrame({"account": accounts})
    features = features.merge(sent_stats, on="account", how="left")
    features = features.merge(recv_stats, on="account", how="left")
    features = features.fillna(0)
    features["avg_sent"] = features["total_sent"] / features["num_sent"].replace(0, 1)
    features["avg_received"] = (
        features["total_received"] / features["num_received"].replace(0, 1)
    )
    features["net_flow"] = features["total_received"] - features["total_sent"]

    src = ibm["sender_id"].map(account_to_idx).to_numpy(dtype=np.int64)
    dst = ibm["receiver_id"].map(account_to_idx).to_numpy(dtype=np.int64)
    node_count = len(accounts)
    topology = coo_matrix(
        (np.ones(len(src), dtype=np.float64), (src, dst)),
        shape=(node_count, node_count),
    ).tocsr()
    topology.data[:] = 1.0
    topology.eliminate_zeros()

    out_degree = np.diff(topology.indptr)
    in_degree = np.diff(topology.tocsc().indptr)
    features["in_degree"] = in_degree
    features["out_degree"] = out_degree

    row_sums = np.asarray(topology.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sums)
    non_dangling = row_sums != 0
    inverse[non_dangling] = 1.0 / row_sums[non_dangling]
    transition = diags(inverse) @ topology
    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    uniform = rank.copy()
    dangling = ~non_dangling
    converged = False
    for _ in range(200):
        previous = rank
        dangling_mass = previous[dangling].sum()
        rank = (
            0.85 * (transition.T @ previous + dangling_mass * uniform)
            + 0.15 * uniform
        )
        if np.abs(rank - previous).sum() < node_count * 1e-4:
            converged = True
            break
    if not converged:
        rank = in_degree / max(node_count, 1)
    features["pagerank"] = rank

    undirected = topology.maximum(topology.T).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    undirected.data[:] = 1.0
    degrees = np.diff(undirected.indptr)
    triangle_walks = np.asarray(
        undirected.multiply(undirected @ undirected).sum(axis=1)
    ).reshape(-1)
    triangles = triangle_walks / 2.0
    denominator = degrees * (degrees - 1)
    clustering = np.zeros(node_count, dtype=np.float64)
    eligible = denominator > 0
    clustering[eligible] = 2.0 * triangles[eligible] / denominator[eligible]
    features["clustering"] = clustering

    fraud_sent = ibm.groupby("sender_id")["Is Laundering"].sum().reset_index()
    fraud_sent = fraud_sent.rename(
        columns={"sender_id": "account", "Is Laundering": "fraud_sent"}
    )
    fraud_received = ibm.groupby("receiver_id")["Is Laundering"].sum().reset_index()
    fraud_received = fraud_received.rename(
        columns={"receiver_id": "account", "Is Laundering": "fraud_received"}
    )
    labels = pd.DataFrame({"account": accounts})
    labels = labels.merge(fraud_sent, on="account", how="left")
    labels = labels.merge(fraud_received, on="account", how="left")
    labels = labels.fillna(0)
    labels["total_fraud"] = labels["fraud_sent"] + labels["fraud_received"]

    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    edge_attr = torch.tensor(
        ibm["Amount Paid"].fillna(0).to_numpy(), dtype=torch.float
    ).unsqueeze(1)
    return _assemble_result(
        ibm, accounts, account_to_idx, features, labels, edge_index,
        edge_attr, StandardScaler(), seed
    )


class GraphParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tiny_transactions()
        cls.reference = notebook_reference(cls.fixture)
        cls.candidate = sparse_candidate(cls.fixture)

    def test_sender_first_then_receiver_only_node_order(self):
        expected = ["1_S", "3_T", "4_L", "2_R", "5_Z"]
        self.assertEqual(list(self.reference["account_to_idx"]), expected)
        self.assertEqual(
            self.candidate["account_to_idx"], self.reference["account_to_idx"]
        )

    def test_repeated_transactions_preserve_rows_but_simple_topology_collapses(self):
        data = self.reference["data"]
        self.assertEqual(data.edge_index.shape[1], len(self.fixture))
        self.assertEqual(data.edge_attr.shape, (len(self.fixture), 1))
        self.assertEqual(self.reference["account_to_idx"]["1_S"], 0)
        self.assertEqual(self.reference["account_to_idx"]["2_R"], 3)
        graph = nx.DiGraph()
        graph.add_edges_from(zip(
            self.reference["transactions"]["sender_id"],
            self.reference["transactions"]["receiver_id"],
        ))
        self.assertEqual(graph.number_of_edges(), 4)
        np.testing.assert_array_equal(
            self.candidate["data"].edge_index.numpy(), data.edge_index.numpy()
        )

    def test_duplicate_rows_affect_aggregates_and_fraud_totals(self):
        duplicate = self.fixture.iloc[[0]]
        without_duplicate = self.fixture.drop(index=1).reset_index(drop=True)
        with_duplicate = self.reference
        without = notebook_reference(without_duplicate)
        sender = "1_S"
        receiver = "2_R"
        with_features = with_duplicate["features"].set_index("account")
        without_features = without["features"].set_index("account")
        with_labels = with_duplicate["labels"].set_index("account")
        without_labels = without["labels"].set_index("account")
        self.assertEqual(with_features.loc[sender, "num_sent"], 3)
        self.assertEqual(without_features.loc[sender, "num_sent"], 2)
        self.assertEqual(with_features.loc[sender, "total_sent"], 25.0)
        self.assertEqual(without_features.loc[sender, "total_sent"], 15.0)
        self.assertEqual(with_labels.loc[sender, "total_fraud"], 2)
        self.assertEqual(without_labels.loc[sender, "total_fraud"], 1)
        self.assertEqual(with_labels.loc[receiver, "total_fraud"], 2)
        self.assertEqual(without_labels.loc[receiver, "total_fraud"], 1)
        self.assertEqual(with_duplicate["data"].y.tolist(), without["data"].y.tolist())
        self.assertEqual(int(duplicate.iloc[0]["Is Laundering"]), 1)

    def test_self_loop_is_allowed_and_has_directed_degree_one(self):
        graph = nx.DiGraph()
        graph.add_edges_from(zip(
            self.reference["transactions"]["sender_id"],
            self.reference["transactions"]["receiver_id"],
        ))
        loop_node = self.reference["account_to_idx"]["4_L"]
        self.assertTrue(graph.has_edge("4_L", "4_L"))
        self.assertEqual(self.reference["features"].loc[loop_node, "in_degree"], 1)
        self.assertEqual(self.reference["features"].loc[loop_node, "out_degree"], 1)

    def test_laundering_transaction_labels_both_endpoints(self):
        data = self.reference["data"]
        sender = self.reference["account_to_idx"]["1_S"]
        receiver = self.reference["account_to_idx"]["2_R"]
        self.assertEqual(data.y[sender].item(), 1)
        self.assertEqual(data.y[receiver].item(), 1)
        self.assertEqual(
            self.candidate["data"].y.tolist(), data.y.tolist()
        )

    def test_directed_degrees_and_pagerank_match_reference(self):
        reference_features = self.reference["features"].set_index("account")
        candidate_features = self.candidate["features"].set_index("account")
        np.testing.assert_array_equal(
            candidate_features["in_degree"].to_numpy(),
            reference_features["in_degree"].to_numpy(),
        )
        np.testing.assert_array_equal(
            candidate_features["out_degree"].to_numpy(),
            reference_features["out_degree"].to_numpy(),
        )
        np.testing.assert_allclose(
            candidate_features["pagerank"].to_numpy(),
            reference_features["pagerank"].to_numpy(),
            rtol=1e-9,
            atol=1e-10,
        )

    def test_undirected_clustering_matches_networkx_with_self_loop(self):
        reference = self.reference["features"].set_index("account")["clustering"]
        candidate = self.candidate["features"].set_index("account")["clustering"]
        np.testing.assert_allclose(candidate.to_numpy(), reference.to_numpy(), atol=0)

    def test_features_scaler_edges_and_pyg_data_contract(self):
        expected_fields = {
            "x", "edge_index", "edge_attr", "y",
            "train_mask", "val_mask", "test_mask",
        }
        reference_data = self.reference["data"]
        candidate_data = self.candidate["data"]
        self.assertIsInstance(candidate_data, Data)
        self.assertEqual(set(candidate_data.keys()), expected_fields)
        self.assertEqual(len(FEATURE_COLUMNS), 13)
        self.assertTrue(
            set(FEATURE_COLUMNS).issubset(self.reference["features"].columns)
        )
        self.assertTrue(
            set(FEATURE_COLUMNS).issubset(self.candidate["features"].columns)
        )
        self.assertEqual(candidate_data.x.shape, (5, 13))
        self.assertEqual(candidate_data.edge_index.dtype, torch.int64)
        self.assertEqual(candidate_data.edge_attr.dtype, torch.float32)
        self.assertEqual(candidate_data.y.dtype, torch.int64)
        self.assertEqual(candidate_data.train_mask.dtype, torch.bool)
        for field in expected_fields:
            left = getattr(candidate_data, field)
            right = getattr(reference_data, field)
            if left.is_floating_point():
                torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        np.testing.assert_allclose(
            self.candidate["scaler"].mean_,
            self.reference["scaler"].mean_,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.candidate["scaler"].scale_,
            self.reference["scaler"].scale_,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(torch.all(
            candidate_data.train_mask
            | candidate_data.val_mask
            | candidate_data.test_mask
        ).item())
        self.assertEqual(
            int(candidate_data.train_mask.sum()
                + candidate_data.val_mask.sum()
                + candidate_data.test_mask.sum()),
            candidate_data.num_nodes,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)