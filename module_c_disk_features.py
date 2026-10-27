"""Bounded disk-backed Module C node aggregates and degree features.

This stage deliberately stops before PageRank, clustering, scaling, masks, or
PyG artifact creation. It requires a live ``DiskPrototypeStore`` context.
"""

from pathlib import Path
import sqlite3

import pandas as pd

from module_c_disk_prototype import _receiver_key, _sender_key


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
IMPLEMENTED_FEATURE_COLUMNS = FEATURE_COLUMNS[:11]
IMPLEMENTED_NODE_COLUMNS = ["label", *IMPLEMENTED_FEATURE_COLUMNS]
REQUIRED_FEATURE_COLUMNS = {
    "From Bank", "Account", "To Bank", "Account.1", "Amount Paid",
    "Amount Received", "Is Laundering",
}


class TemporaryStorageLimitExceeded(RuntimeError):
    """Raised when prototype database and journal exceed the configured cap."""


def _validate_limits(batch_rows, max_temp_bytes):
    if batch_rows <= 0:
        raise ValueError("batch_rows must be greater than zero")
    if max_temp_bytes <= 0:
        raise ValueError("max_temp_bytes must be greater than zero")


def _size_of_sqlite_workspace(store):
    database = Path(store.temporary_directory) / "prototype.sqlite3"
    journal = database.with_name(database.name + "-journal")
    return database.stat().st_size + (journal.stat().st_size if journal.exists() else 0)


def _sql_number(value):
    if pd.isna(value):
        return None
    return float(value)


def _create_feature_tables(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS node_aggregates (
            account_index INTEGER PRIMARY KEY,
            total_sent REAL NOT NULL DEFAULT 0,
            num_sent INTEGER NOT NULL DEFAULT 0,
            total_received REAL NOT NULL DEFAULT 0,
            num_received INTEGER NOT NULL DEFAULT 0,
            fraud_sent REAL NOT NULL DEFAULT 0,
            fraud_received REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS node_degrees (
            account_index INTEGER PRIMARY KEY,
            in_degree INTEGER NOT NULL DEFAULT 0,
            out_degree INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO node_degrees(account_index) "
        "SELECT account_index FROM accounts"
    )
    connection.execute(
        "INSERT INTO node_degrees(account_index, out_degree) "
        "SELECT source_index, COUNT(*) FROM topology GROUP BY source_index "
        "ON CONFLICT(account_index) DO UPDATE SET out_degree=excluded.out_degree"
    )
    connection.execute(
        "INSERT INTO node_degrees(account_index, in_degree) "
        "SELECT target_index, COUNT(*) FROM topology GROUP BY target_index "
        "ON CONFLICT(account_index) DO UPDATE SET in_degree=excluded.in_degree"
    )


class DiskNodeFeatureStore:
    """Bounded-batch ordered view of the implemented node features."""

    def __init__(self, connection, *, row_count):
        self._connection = connection
        self.row_count = row_count

    def iter_rows(self, *, batch_rows=1024):
        if batch_rows <= 0:
            raise ValueError("batch_rows must be greater than zero")
        query = """
            SELECT a.account_key,
                   CASE WHEN COALESCE(g.fraud_sent, 0) +
                                  COALESCE(g.fraud_received, 0) > 0 THEN 1 ELSE 0 END,
                   COALESCE(g.total_sent, 0),
                   COALESCE(g.total_received, 0),
                   COALESCE(g.num_sent, 0),
                   COALESCE(g.num_received, 0),
                   CASE WHEN COALESCE(g.num_sent, 0) = 0 THEN 0
                        ELSE g.total_sent / g.num_sent END,
                   CASE WHEN COALESCE(g.num_received, 0) = 0 THEN 0
                        ELSE g.total_received / g.num_received END,
                   COALESCE(sc.distinct_receivers, 0),
                   COALESCE(rc.distinct_senders, 0),
                   COALESCE(g.total_received, 0) - COALESCE(g.total_sent, 0),
                   d.in_degree,
                   d.out_degree
            FROM accounts AS a
            LEFT JOIN node_aggregates AS g ON g.account_index = a.account_index
            LEFT JOIN sender_counts AS sc ON sc.account_index = a.account_index
            LEFT JOIN receiver_counts AS rc ON rc.account_index = a.account_index
            LEFT JOIN node_degrees AS d ON d.account_index = a.account_index
            ORDER BY a.account_index
        """
        cursor = self._connection.execute(query)
        try:
            while True:
                rows = cursor.fetchmany(batch_rows)
                if not rows:
                    break
                for row in rows:
                    yield dict(zip(("account", *IMPLEMENTED_NODE_COLUMNS), row))
        finally:
            cursor.close()


def populate_node_features(
    csv_path,
    store,
    *,
    max_rows,
    batch_rows,
    max_temp_bytes,
):
    """Aggregate exact transaction features into the live prototype database.

    The CSV is reread up to the prototype's row limit and paired with its
    ordered transaction-edge iterator. The SQLite database remains owned by
    the caller's ``open_bounded_disk_prototype`` context.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    _validate_limits(batch_rows, max_temp_bytes)
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero")
    if max_rows != store.max_rows:
        raise ValueError("max_rows must match the prototype store row limit")
    missing = REQUIRED_FEATURE_COLUMNS.difference(
        pd.read_csv(csv_path, nrows=0).columns
    )
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    connection = store._connection
    _create_feature_tables(connection)
    edge_rows = iter(store.iter_transaction_edges(batch_size=batch_rows))
    row_index = 0

    reader = pd.read_csv(csv_path, chunksize=batch_rows, nrows=store.rows_read)
    try:
        for chunk in reader:
            records = chunk.to_dict(orient="records")
            aggregate_updates = []
            for record in records:
                try:
                    stored = next(edge_rows)
                except StopIteration as error:
                    raise RuntimeError("CSV has more rows than the stored edge table") from error

                stored_row, source_index, target_index, stored_amount, stored_label = stored
                if stored_row != row_index:
                    raise RuntimeError("Stored transaction row order is inconsistent")
                source_key = _sender_key(record)
                target_key = _receiver_key(record)
                if (
                    store.account_index(source_key) != source_index
                    or store.account_index(target_key) != target_index
                ):
                    raise RuntimeError("CSV endpoints do not match stored edge endpoints")

                amount_paid = _sql_number(record["Amount Paid"])
                amount_received = _sql_number(record["Amount Received"])
                label = _sql_number(record["Is Laundering"])
                if (
                    stored_amount is not None
                    and amount_paid is not None
                    and float(stored_amount) != float(amount_paid)
                ):
                    raise RuntimeError("CSV amount does not match stored transaction edge")
                if label is None:
                    label = 0.0
                if stored_label is not None and int(stored_label) != int(label):
                    raise RuntimeError("CSV laundering label does not match stored edge")

                aggregate_updates.append(
                    (
                        source_index,
                        amount_paid,
                        int(amount_paid is not None),
                        0.0,
                        0,
                        label,
                        0.0,
                    )
                )
                aggregate_updates.append(
                    (
                        target_index,
                        0.0,
                        0,
                        amount_received,
                        int(amount_received is not None),
                        0.0,
                        label,
                    )
                )
                row_index += 1

            connection.executemany(
                """
                INSERT INTO node_aggregates(
                    account_index, total_sent, num_sent, total_received,
                    num_received, fraud_sent, fraud_received
                ) VALUES (?, COALESCE(?, 0), ?, COALESCE(?, 0), ?, ?, ?)
                ON CONFLICT(account_index) DO UPDATE SET
                    total_sent = node_aggregates.total_sent + excluded.total_sent,
                    num_sent = node_aggregates.num_sent + excluded.num_sent,
                    total_received = node_aggregates.total_received + excluded.total_received,
                    num_received = node_aggregates.num_received + excluded.num_received,
                    fraud_sent = node_aggregates.fraud_sent + excluded.fraud_sent,
                    fraud_received = node_aggregates.fraud_received + excluded.fraud_received
                """,
                aggregate_updates,
            )
            connection.commit()
            if _size_of_sqlite_workspace(store) > max_temp_bytes:
                raise TemporaryStorageLimitExceeded(
                    f"SQLite workspace exceeds {max_temp_bytes} bytes"
                )
    finally:
        reader.close()

    try:
        next(edge_rows)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Stored edge table has more rows than the CSV prefix")
    if row_index != store.rows_read:
        raise RuntimeError(
            f"Feature pass read {row_index} rows; prototype stored {store.rows_read}"
        )

    connection.commit()
    if _size_of_sqlite_workspace(store) > max_temp_bytes:
        raise TemporaryStorageLimitExceeded(
            f"SQLite workspace exceeds {max_temp_bytes} bytes"
        )
    return DiskNodeFeatureStore(connection, row_count=row_index)