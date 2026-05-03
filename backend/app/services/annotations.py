from __future__ import annotations

import json

from ..schemas import AnnotationRecord, AnnotationRequest
from ..storage import connect, init_db, utc_now


def save_annotation(payload: AnnotationRequest) -> AnnotationRecord:
    init_db()
    created_at = utc_now()
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO annotations (
                sequence_id, frame_id, terrain, weather, task, labels_json, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.sequence_id,
                payload.frame_id,
                payload.terrain,
                payload.weather,
                payload.task,
                json.dumps(payload.labels, ensure_ascii=False),
                payload.note,
                created_at,
            ),
        )
        annotation_id = int(cursor.lastrowid)
    return AnnotationRecord(id=annotation_id, created_at=created_at, **payload.model_dump())
