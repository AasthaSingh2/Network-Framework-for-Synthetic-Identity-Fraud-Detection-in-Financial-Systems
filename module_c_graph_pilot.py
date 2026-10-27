"""Strictly bounded CSV chunk-reader pilot for Module C.

This is not a full-data graph builder. It stages no more than ``max_rows``
from a CSV into a unique system temporary directory, reconstructs that bounded
frame, and delegates graph construction to the existing in-memory helper.
All staging files are removed before the function returns.
"""

from pathlib import Path
import tempfile

import pandas as pd

from module_c_graph_builder import build_graph


def run_bounded_csv_pilot(csv_path, *, max_rows, chunk_rows, seed=None):
    """Read and validate at most ``max_rows`` rows from ``csv_path``.

    CSV chunks are staged as temporary pickle files to preserve pandas dtypes
    across chunk boundaries. The bounded chunks are then concatenated in source
    order and passed to ``build_graph``. No output is written under the project
    or its ``outputs`` directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be greater than zero")

    chunk_sizes = []
    staged_parts = []
    rows_read = 0
    graph_result = None
    temp_path = None

    with tempfile.TemporaryDirectory(prefix="module-c-bounded-pilot-") as temp_dir:
        temp_path = Path(temp_dir).resolve()
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_rows,
            nrows=max_rows,
        )
        try:
            for chunk_number, chunk in enumerate(reader):
                if rows_read + len(chunk) > max_rows:
                    raise RuntimeError("CSV reader exceeded the configured row limit")
                if chunk.empty:
                    continue
                staged_file = temp_path / f"chunk-{chunk_number:06d}.pkl"
                chunk.to_pickle(staged_file)
                staged_parts.append(staged_file)
                chunk_sizes.append(len(chunk))
                rows_read += len(chunk)
        finally:
            reader.close()

        if not staged_parts:
            raise ValueError("No transaction rows were read within the configured limit")

        bounded_frame = pd.concat(
            [pd.read_pickle(part) for part in staged_parts],
            ignore_index=True,
        )
        if len(bounded_frame) != rows_read or rows_read > max_rows:
            raise RuntimeError("Staged row count does not match the bounded read count")

        graph_result = build_graph(bounded_frame, seed=seed)

    return {
        "graph_result": graph_result,
        "rows_read": rows_read,
        "chunk_sizes": tuple(chunk_sizes),
        "max_rows": max_rows,
        "chunk_rows": chunk_rows,
        "temporary_directory": str(temp_path),
        "temporary_directory_removed": not temp_path.exists(),
    }