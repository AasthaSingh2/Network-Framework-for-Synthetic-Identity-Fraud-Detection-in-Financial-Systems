"""Bounded-batch PageRank and clustering over a live prototype SQLite store.

Directed and undirected adjacency remain disk-backed. Working vectors and
triangle counters require O(number of nodes) RAM; no NetworkX graph or full
edge table is materialized. This module does not build final graph artifacts.
"""

from dataclasses import dataclass
import sqlite3
import uuid

import numpy as np


def _check_batch_size(batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")


def _insert_unique_undirected(connection, table_name, edge_batch):
    values = []
    for source, target in edge_batch:
        if source == target:
            continue
        values.append((min(source, target), max(source, target)))
    if values:
        connection.executemany(
            f"INSERT OR IGNORE INTO {table_name}(u, v) VALUES (?, ?)", values
        )


def _iter_sql_batches(connection, query, parameters, batch_size):
    _check_batch_size(batch_size)
    cursor = connection.execute(query, parameters)
    try:
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            yield batch
    finally:
        cursor.close()


def _advance_pagerank(rank, out_degree, store, *, batch_size, damping):
    node_count = len(rank)
    incoming = np.zeros(node_count, dtype=np.float64)
    for batch in store.iter_topology_edges(batch_size=batch_size):
        for source, target in batch:
            degree = out_degree[source]
            if degree:
                incoming[target] += rank[source] / degree

    dangling_mass = rank[out_degree == 0].sum()
    uniform = 1.0 / node_count
    return damping * (incoming + dangling_mass * uniform) + (1.0 - damping) * uniform


@dataclass
class DiskTopologyFeatureStore:
    """Bounded reader for computed topology features in the prototype DB."""

    connection: sqlite3.Connection
    table_name: str
    node_count: int

    def iter_rows(self, *, batch_size=1024):
        yield from _iter_sql_batches(
            self.connection,
            f"SELECT account_index, in_degree, out_degree, pagerank, clustering "
            f"FROM {self.table_name} ORDER BY account_index",
            (),
            batch_size,
        )


def compute_disk_topology_features(
    store,
    *,
    batch_size=4096,
    max_iter=200,
    tolerance=1e-4,
    damping=0.85,
):
    """Compute directed degree/PageRank and undirected clustering on disk.

    PageRank follows NetworkX defaults used by the notebook: uniform initial
    rank/personalization/dangling redistribution, unit edge weights,
    ``max_iter=200``, and L1 error threshold ``node_count * tolerance``. Any
    ordinary exception during PageRank or exhaustion without convergence uses
    the notebook's directed in-degree divided by node count fallback.
    """
    _check_batch_size(batch_size)
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero")
    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")
    if not 0.0 < damping < 1.0:
        raise ValueError("damping must be between zero and one")

    connection = store._connection
    node_count = connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    if node_count == 0:
        raise ValueError("topology feature calculation requires at least one node")

    suffix = uuid.uuid4().hex
    undirected_table = f"disk_undirected_{suffix}"
    feature_table = f"disk_topology_features_{suffix}"
    connection.execute(
        f"CREATE TABLE {undirected_table} ("
        "u INTEGER NOT NULL, v INTEGER NOT NULL, PRIMARY KEY (u, v))"
    )

    for batch in store.iter_topology_edges(batch_size=batch_size):
        _insert_unique_undirected(connection, undirected_table, batch)
        connection.commit()

    in_degree = np.zeros(node_count, dtype=np.int64)
    out_degree = np.zeros(node_count, dtype=np.int64)
    for batch in store.iter_topology_edges(batch_size=batch_size):
        for source, target in batch:
            out_degree[source] += 1
            in_degree[target] += 1

    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    converged = False
    try:
        for _ in range(max_iter):
            next_rank = _advance_pagerank(
                rank,
                out_degree,
                store,
                batch_size=batch_size,
                damping=damping,
            )
            if np.abs(next_rank - rank).sum() < node_count * tolerance:
                rank = next_rank
                converged = True
                break
            rank = next_rank
    except Exception:
        converged = False

    if not converged:
        rank = in_degree.astype(np.float64) / max(node_count, 1)

    undirected_degree = np.zeros(node_count, dtype=np.int64)
    for batch in _iter_sql_batches(
        connection,
        f"SELECT u, v FROM {undirected_table} ORDER BY u, v",
        (),
        batch_size,
    ):
        for left, right in batch:
            undirected_degree[left] += 1
            undirected_degree[right] += 1

    triangle_count = np.zeros(node_count, dtype=np.int64)
    triangle_query = (
        f"SELECT a.u, a.v, b.v FROM {undirected_table} AS a "
        f"JOIN {undirected_table} AS b ON b.u = a.u AND a.v < b.v "
        f"JOIN {undirected_table} AS c ON c.u = a.v AND c.v = b.v "
        f"ORDER BY a.u, a.v, b.v"
    )
    for batch in _iter_sql_batches(connection, triangle_query, (), batch_size):
        for first, second, third in batch:
            triangle_count[first] += 1
            triangle_count[second] += 1
            triangle_count[third] += 1

    denominator = undirected_degree * (undirected_degree - 1)
    clustering = np.zeros(node_count, dtype=np.float64)
    valid = denominator > 0
    clustering[valid] = 2.0 * triangle_count[valid] / denominator[valid]

    connection.execute(
        f"CREATE TABLE {feature_table} ("
        "account_index INTEGER PRIMARY KEY, "
        "in_degree INTEGER NOT NULL, "
        "out_degree INTEGER NOT NULL, "
        "pagerank REAL NOT NULL, "
        "clustering REAL NOT NULL)"
    )
    for start in range(0, node_count, batch_size):
        stop = min(start + batch_size, node_count)
        connection.executemany(
            f"INSERT INTO {feature_table} VALUES (?, ?, ?, ?, ?)",
            [
                (
                    index,
                    int(in_degree[index]),
                    int(out_degree[index]),
                    float(rank[index]),
                    float(clustering[index]),
                )
                for index in range(start, stop)
            ],
        )
        connection.commit()

    connection.execute(f"DROP TABLE {undirected_table}")
    connection.commit()
    return DiskTopologyFeatureStore(connection, feature_table, node_count)