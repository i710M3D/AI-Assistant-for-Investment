from __future__ import annotations

from pathlib import Path
from typing import Any

from .rag import build_index
from .scoring import compute_scores
from .monitoring import generate_alerts
from .notes import generate_company_note


def get_system(dataset_dir: str | Path = "data/dataset") -> dict[str, Any]:
    index = build_index(dataset_dir)
    scores = compute_scores(dataset_dir)
    alerts = generate_alerts(scores, index)
    notes = {row["company_name"]: generate_company_note(row, alerts, index) for row in scores}
    return {"index": index, "scores": scores, "alerts": alerts, "notes": notes}
