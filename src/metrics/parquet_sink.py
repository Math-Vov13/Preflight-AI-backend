"""Append-only Parquet writer for per-call cost records."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from metrics.cost import CallRecord


def write_call_records(records: list[CallRecord], path: Path) -> None:
    if not records:
        return
    rows = [asdict(r) for r in records]
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
