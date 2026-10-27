"""Isolated, sparse graph-construction candidate for Module C parity work.

This module creates an in-memory PyTorch Geometric object and does not save
files. It is intended for tiny-fixture validation; it is not yet a full-data
streaming builder.
"""

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


def _pagerank_has_converged(rank, previous, node_count, tolerance):
    return np.abs(rank - previous).sum() < node_count * tolerance


def _sparse_pagerank(topology, in_degree, max_iter=200, tolerance=1e-4):
    node_count = topology.shape[0]
    if node_count == 0:
        raise ValueError("PageRank requires at least one node")

    try:
        row_sums = np.asarray(topology.sum(axis=1)).reshape(-1)
        inverse = np.zeros_like(row_sums, dtype=np.float64)
        non_dangling = row_sums != 0
        inverse[non_dangling] = 1.0 / row_sums[non_dangling]
        transition = diags(inverse) @ topology

        rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
        uniform = rank.copy()
        dangling = ~non_dangling
        converged = False
        for _ in range(max_iter):
            previous = rank
            dangling_mass = previous[dangling].sum()
            rank = (
                0.85 * (transition.T @ previous + dangling_mass * uniform)
                + 0.15 * uniform
            )
            if _pagerank_has_converged(rank, previous, node_count, tolerance):
                converged = True
                break

        if converged:
            return np.asarray(rank, dtype=np.float64)
    except Exception:
        pass

    rank = in_degree / max(node_count, 1)
    return np.asarray(rank, dtype=np.float64)


def _sparse_undirected_clustering(topology):
    undirected = topology.maximum(topology.T).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    undirected.data[:] = 1.0
    node_count = undirected.shape[0]
    degrees = np.diff(undirected.indptr)

    # Orient each undirected edge by (degree, node index). Every triangle is
    # then counted once through forward-neighbor intersections.
    forward_rows = []
    forward_cols = []
    for source in range(node_count):
        neighbors = undirected.indices[
            undirected.indptr[source]:undirected.indptr[source + 1]
        ]
        for target in neighbors:
            if (degrees[source], source) < (degrees[target], int(target)):
                forward_rows.append(source)
                forward_cols.append(int(target))

    forward = coo_matrix(
        (np.ones(len(forward_rows), dtype=np.uint8),
         (forward_rows, forward_cols)),
        shape=(node_count, node_count),
    ).tocsr()
    forward.sort_indices()
    triangle_counts = np.zeros(node_count, dtype=np.int64)
    for source in range(node_count):
        source_neighbors = forward.indices[
            forward.indptr[source]:forward.indptr[source + 1]
        ]
        for target in source_neighbors:
            target = int(target)
            target_neighbors = forward.indices[
                forward.indptr[target]:forward.indptr[target + 1]
            ]
            common = np.intersect1d(
                source_neighbors, target_neighbors, assume_unique=True
            )
            if common.size:
                triangle_counts[source] += common.size
                triangle_counts[target] += common.size
                triangle_counts[common] += 1

    denominator = degrees * (degrees - 1)
    clustering = np.zeros(node_count, dtype=np.float64)
    eligible = denominator > 0
    clustering[eligible] = 2.0 * triangle_counts[eligible] / denominator[eligible]
    return clustering


def build_graph(frame, seed=None):
    """Build a PyG graph from a small IBM-shaped DataFrame without saving it.

    ``seed=None`` preserves the notebook's use of the current global Torch RNG.
    Passing a seed is intended for reproducible parity tests.
    """
    required = {
        "From Bank", "Account", "To Bank", "Account.1", "Amount Paid",
        "Amount Received", "Is Laundering",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Graph construction requires at least one transaction")

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

    sent_stats = transactions.groupby("sender_id").agg(
        total_sent=("Amount Paid", "sum"),
        num_sent=("Amount Paid", "count"),
        unique_receivers=("receiver_id", "nunique"),
    ).reset_index().rename(columns={"sender_id": "account"})
    received_stats = transactions.groupby("receiver_id").agg(
        total_received=("Amount Received", "sum"),
        num_received=("Amount Received", "count"),
        unique_senders=("sender_id", "nunique"),
    ).reset_index().rename(columns={"receiver_id": "account"})

    features = pd.DataFrame({"account": accounts})
    features = features.merge(sent_stats, on="account", how="left")
    features = features.merge(received_stats, on="account", how="left")
    features = features.fillna(0)
    features["avg_sent"] = (
        features["total_sent"] / features["num_sent"].replace(0, 1)
    )
    features["avg_received"] = (
        features["total_received"] / features["num_received"].replace(0, 1)
    )
    features["net_flow"] = features["total_received"] - features["total_sent"]

    source = transactions["sender_id"].map(account_to_idx).to_numpy(dtype=np.int64)
    target = transactions["receiver_id"].map(account_to_idx).to_numpy(dtype=np.int64)
    node_count = len(accounts)
    topology = coo_matrix(
        (np.ones(len(source), dtype=np.float64), (source, target)),
        shape=(node_count, node_count),
    ).tocsr()
    topology.sum_duplicates()
    topology.data[:] = 1.0
    topology.eliminate_zeros()

    out_degree = np.diff(topology.indptr)
    in_degree = np.diff(topology.tocsc().indptr)
    features["in_degree"] = in_degree
    features["out_degree"] = out_degree
    features["pagerank"] = _sparse_pagerank(topology, in_degree)
    features["clustering"] = _sparse_undirected_clustering(topology)

    fraud_sent = transactions.groupby("sender_id")["Is Laundering"].sum().reset_index()
    fraud_sent = fraud_sent.rename(
        columns={"sender_id": "account", "Is Laundering": "fraud_sent"}
    )
    fraud_received = (
        transactions.groupby("receiver_id")["Is Laundering"].sum().reset_index()
    )
    fraud_received = fraud_received.rename(
        columns={"receiver_id": "account", "Is Laundering": "fraud_received"}
    )
    labels = pd.DataFrame({"account": accounts})
    labels = labels.merge(fraud_sent, on="account", how="left")
    labels = labels.merge(fraud_received, on="account", how="left")
    labels = labels.fillna(0)
    labels["total_fraud"] = labels["fraud_sent"] + labels["fraud_received"]
    y = torch.tensor(
        (labels["total_fraud"] > 0).astype(int).values, dtype=torch.long
    )

    x = torch.tensor(features[FEATURE_COLUMNS].values, dtype=torch.float)
    scaler = StandardScaler()
    x_scaled = torch.tensor(scaler.fit_transform(x.numpy()), dtype=torch.float)

    edge_index = torch.tensor(np.stack([source, target]), dtype=torch.long)
    edge_attr = torch.tensor(
        transactions["Amount Paid"].fillna(0).to_numpy(), dtype=torch.float
    ).unsqueeze(1)

    if seed is not None:
        torch.manual_seed(seed)
    indices = torch.randperm(node_count)
    train_end = int(0.70 * node_count)
    validation_end = int(0.85 * node_count)
    train_mask = torch.zeros(node_count, dtype=torch.bool)
    validation_mask = torch.zeros(node_count, dtype=torch.bool)
    test_mask = torch.zeros(node_count, dtype=torch.bool)
    train_mask[indices[:train_end]] = True
    validation_mask[indices[train_end:validation_end]] = True
    test_mask[indices[validation_end:]] = True

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