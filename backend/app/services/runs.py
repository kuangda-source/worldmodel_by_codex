from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..schemas import Provenance, RunComparisonResponse, RunComparisonRow, RunExportResponse, RunRecord
from ..storage import get_run as storage_get_run
from ..storage import list_runs as storage_list_runs
from ..storage import put_run, utc_now


def record_run(
    *,
    run_id: str,
    kind: str,
    name: str,
    provenance: Provenance,
    sequence_id: str | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    status: str = "completed",
) -> RunRecord:
    record = RunRecord(
        run_id=run_id,
        kind=kind,
        name=name,
        status=status,  # type: ignore[arg-type]
        sequence_id=sequence_id,
        source=provenance.source,
        provenance=provenance,
        metrics=metrics or {},
        artifacts=artifacts or {},
        config=config or {},
        created_at=utc_now(),
    )
    return put_run(record)


def list_run_records(source: str | None = None, kind: str | None = None, limit: int = 50) -> list[RunRecord]:
    return storage_list_runs(source=source, kind=kind, limit=limit)


def get_run_record(run_id: str) -> RunRecord:
    record = storage_get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return record


def export_run_record(run_id: str) -> RunExportResponse:
    record = get_run_record(run_id)
    bundle = {
        "schema_version": "orwm-run-bundle-v0.1",
        "run_id": record.run_id,
        "kind": record.kind,
        "source": record.source,
        "created_at": record.created_at,
        "provenance": record.provenance.model_dump(),
        "config": record.config,
        "metrics": record.metrics,
        "artifacts": record.artifacts,
        "warnings": record.provenance.notes,
    }
    return RunExportResponse(run=record, bundle=bundle)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | str | int | None]:
    scalars: dict[str, float | str | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            scalars[key] = int(value)
        elif isinstance(value, (int, float, str)) or value is None:
            scalars[key] = value
    return scalars


def compare_run_records(source: str | None = None, kind: str | None = None, limit: int = 30) -> RunComparisonResponse:
    records = list_run_records(source=source if source != "all" else None, kind=kind if kind != "all" else None, limit=limit)
    rows: list[RunComparisonRow] = []
    metric_keys: set[str] = set()
    for record in records:
        metrics = _scalar_metrics(record.metrics)
        metric_keys.update(metrics.keys())
        rows.append(
            RunComparisonRow(
                run_id=record.run_id,
                name=record.name,
                kind=record.kind,
                source=record.source,
                created_at=record.created_at,
                metrics=metrics,
            )
        )
    return RunComparisonResponse(kind=kind, source=source, metric_keys=sorted(metric_keys), rows=rows)
