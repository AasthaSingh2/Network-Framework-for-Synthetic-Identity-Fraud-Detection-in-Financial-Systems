"""Tiny-input disk-backed prototype for Module C endpoint processing.

This module intentionally stops before graph features, scaling, masks, PyG
objects, or production artifacts. It is suitable only for bounded pilots.
"""

from contextlib import contextmanager
import csv
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Paid",
    "Is Laundering",
}


def _sender_key(row):
    return f"{str(row['From Bank'])}_{str(row['Account'])}"


def _receiver_key(row):
    return f"{str(row['To Bank'])}_{str(row['Account.1'])}"


def _csv_chunks(csv_path, *, chunk_rows, max_rows):
    """Yield at most max_rows while preserving CSV row order."""
    remaining = max_rows
    reader = pd.read_csv(csv_path, chunksize=chunk_rows, nrows=max_rows)
    try:
        for chunk in reader:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)
            yield chunk
    finally:
        reader.close()


@contextmanager
def _disk_workspace(temp_root=None):
    outputs_directory = (Path(__file__).resolve().parent / "outputs").resolve()
    if temp_root is not None:
        resolved_root = Path(temp_root).resolve()
        try:
            resolved_root.relative_to(outputs_directory)
        except ValueError:
            pass
        else:
            raise ValueError("temporary workspace must be outside outputs/")
    with tempfile.TemporaryDirectory(
        prefix="module-c-disk-prototype-", dir=temp_root
    ) as directory:
        yield Path(directory).resolve()


def _initialize_database(connection):
    connection.executescript(
        """
        CREATE TABLE accounts (
            account_key TEXT PRIMARY KEY,
            account_index INTEGER NOT NULL UNIQUE
        );
        CREATE TABLE transaction_edges (
            row_index INTEGER PRIMARY KEY,
            source_index INTEGER NOT NULL,
            target_index INTEGER NOT NULL,
            amount_paid REAL,
            is_laundering INTEGER
        );
        CREATE TABLE sender_counterpart_pairs (
            sender_index INTEGER NOT NULL,
            receiver_index INTEGER NOT NULL,
            PRIMARY KEY (sender_index, receiver_index)
        );
        CREATE TABLE receiver_counterpart_pairs (
            receiver_index INTEGER NOT NULL,
            sender_index INTEGER NOT NULL,
            PRIMARY KEY (receiver_index, sender_index)
        );
        CREATE TABLE topology (
            source_index INTEGER NOT NULL,
            target_index INTEGER NOT NULL,
            PRIMARY KEY (source_index, target_index)
        );
        CREATE TABLE sender_counts (
            account_index INTEGER PRIMARY KEY,
            distinct_receivers INTEGER NOT NULL
        );
        CREATE TABLE receiver_counts (
            account_index INTEGER PRIMARY KEY,
            distinct_senders INTEGER NOT NULL
        );
        """
    )


def _assign_account(connection, account_key, next_index):
    existing = connection.execute(
        "SELECT account_index FROM accounts WHERE account_key = ?",
        (account_key,),
    ).fetchone()
    if existing is not None:
        return existing[0], next_index
    connection.execute(
        "INSERT INTO accounts(account_key, account_index) VALUES (?, ?)",
        (account_key, next_index),
    )
    return next_index, next_index + 1


def _load_index_map(connection):
    return {
        key: index
        for key, index in connection.execute(
            "SELECT account_key, account_index FROM accounts ORDER BY account_index"
        )
    }


def _validate_header(csv_path):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = REQUIRED_COLUMNS.difference(header)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


class DiskPrototypeStore:
    """Bounded-batch, disk-backed view of a live prototype SQLite database."""

    def __init__(self, connection, *, rows_read, max_rows, workspace):
        self._connection = connection
        self.rows_read = rows_read
        self.max_rows = max_rows
        self.temporary_directory = str(workspace)

    @staticmethod
    def _iter_query(connection, query, parameters, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        cursor = connection.execute(query, parameters)
        try:
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                yield from batch
        finally:
            cursor.close()

    def iter_accounts(self, *, batch_size=1024):
        """Yield ``(account_key, account_index)`` ordered by assigned index."""
        yield from self._iter_query(
            self._connection,
            "SELECT account_key, account_index FROM accounts "
            "ORDER BY account_index",
            (),
            batch_size,
        )

    def account_index(self, account_key):
        """Look up one account index without materializing the full mapping."""
        row = self._connection.execute(
            "SELECT account_index FROM accounts WHERE account_key = ?",
            (account_key,),
        ).fetchone()
        return None if row is None else row[0]

    def iter_transaction_edges(self, *, batch_size=4096):
        """Yield every input edge row in original CSV order, duplicates included."""
        yield from self._iter_query(
            self._connection,
            "SELECT row_index, source_index, target_index, amount_paid, "
            "is_laundering FROM transaction_edges ORDER BY row_index",
            (),
            batch_size,
        )

    def iter_topology_edges(self, *, batch_size=4096):
        """Yield unique directed topology edges in deterministic index order."""
        yield from self._iter_query(
            self._connection,
            "SELECT source_index, target_index FROM topology "
            "ORDER BY source_index, target_index",
            (),
            batch_size,
        )

    def iter_sender_counts(self, *, batch_size=1024):
        yield from self._iter_query(
            self._connection,
            "SELECT account_index, distinct_receivers FROM sender_counts "
            "ORDER BY account_index",
            (),
            batch_size,
        )

    def iter_receiver_counts(self, *, batch_size=1024):
        yield from self._iter_query(
            self._connection,
            "SELECT account_index, distinct_senders FROM receiver_counts "
            "ORDER BY account_index",
            (),
            batch_size,
        )


@contextmanager
def open_bounded_disk_prototype(
    csv_path,
    *,
    max_rows,
    chunk_rows,
    temp_root=None,
    fail_after_rows=None,
):
    """Open a live, temporary disk store for a bounded CSV prefix.

    Exactly two endpoint-index passes are performed: sender-first assignment,
    then receiver-only assignment. The third pass persists every capped input
    row in order and forms separate exact distinct-pair and topology tables.
    The yielded store remains usable only inside the ``with`` block. All
    temporary database files are deleted on normal exit or exception.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be greater than zero")
    if fail_after_rows is not None and fail_after_rows <= 0:
        raise ValueError("fail_after_rows must be greater than zero")
    _validate_header(csv_path)

    with _disk_workspace(temp_root) as workspace:
        database_path = workspace / "prototype.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _initialize_database(connection)

            # Pass 1: first-seen sender keys in source row order.
            next_index = 0
            for chunk in _csv_chunks(
                csv_path, chunk_rows=chunk_rows, max_rows=max_rows
            ):
                for row in chunk.to_dict(orient="records"):
                    _, next_index = _assign_account(
                        connection, _sender_key(row), next_index
                    )
                connection.commit()

            # Pass 2: append receiver-only keys in source row order.
            for chunk in _csv_chunks(
                csv_path, chunk_rows=chunk_rows, max_rows=max_rows
            ):
                for row in chunk.to_dict(orient="records"):
                    _, next_index = _assign_account(
                        connection, _receiver_key(row), next_index
                    )
                connection.commit()

            account_to_idx = _load_index_map(connection)

            # Pass 3: preserve every transaction row and build exact pair sets.
            row_index = 0
            for chunk in _csv_chunks(
                csv_path, chunk_rows=chunk_rows, max_rows=max_rows
            ):
                transaction_rows = []
                sender_pairs = []
                receiver_pairs = []
                topology_pairs = []
                for row in chunk.to_dict(orient="records"):
                    source_key = _sender_key(row)
                    target_key = _receiver_key(row)
                    source_index = account_to_idx[source_key]
                    target_index = account_to_idx[target_key]
                    amount = row["Amount Paid"]
                    laundering = row["Is Laundering"]
                    transaction_rows.append(
                        (row_index, source_index, target_index, amount, laundering)
                    )
                    sender_pairs.append((source_index, target_index))
                    receiver_pairs.append((target_index, source_index))
                    topology_pairs.append((source_index, target_index))
                    row_index += 1
                    if fail_after_rows is not None and row_index >= fail_after_rows:
                        raise RuntimeError("synthetic pilot failure requested")

                connection.executemany(
                    "INSERT INTO transaction_edges VALUES (?, ?, ?, ?, ?)",
                    transaction_rows,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO sender_counterpart_pairs VALUES (?, ?)",
                    sender_pairs,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO receiver_counterpart_pairs VALUES (?, ?)",
                    receiver_pairs,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO topology VALUES (?, ?)", topology_pairs
                )
                connection.commit()

            if row_index > max_rows:
                raise RuntimeError("row limit was exceeded")

            connection.execute(
                "INSERT INTO sender_counts "
                "SELECT sender_index, COUNT(*) FROM sender_counterpart_pairs "
                "GROUP BY sender_index"
            )
            connection.execute(
                "INSERT INTO receiver_counts "
                "SELECT receiver_index, COUNT(*) FROM receiver_counterpart_pairs "
                "GROUP BY receiver_index"
            )
            connection.commit()

            yield DiskPrototypeStore(
                connection,
                rows_read=row_index,
                max_rows=max_rows,
                workspace=workspace,
            )
        finally:
            connection.close()


def build_bounded_disk_prototype(
    csv_path,
    *,
    max_rows,
    chunk_rows,
    temp_root=None,
    fail_after_rows=None,
):
    """Compatibility wrapper that explicitly materializes a bounded result.

    Prefer ``open_bounded_disk_prototype`` when consumers can process batches.
    This legacy-shaped wrapper retains the prior return contract and therefore
    materializes the bounded tables in memory by explicit caller choice.
    """
    with open_bounded_disk_prototype(
        csv_path,
        max_rows=max_rows,
        chunk_rows=chunk_rows,
        temp_root=temp_root,
        fail_after_rows=fail_after_rows,
    ) as store:
        accounts = list(store.iter_accounts(batch_size=1024))
        account_to_idx = {key: index for key, index in accounts}
        return {
            "accounts": [key for key, _ in accounts],
            "account_to_idx": account_to_idx,
            "transaction_edges": list(store.iter_transaction_edges(batch_size=4096)),
            "sender_distinct_receiver_counts": dict(
                store.iter_sender_counts(batch_size=1024)
            ),
            "receiver_distinct_sender_counts": dict(
                store.iter_receiver_counts(batch_size=1024)
            ),
            "topology_edges": list(store.iter_topology_edges(batch_size=4096)),
            "rows_read": store.rows_read,
            "max_rows": store.max_rows,
            "temporary_directory": store.temporary_directory,
        }