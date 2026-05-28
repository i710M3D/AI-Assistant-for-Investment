from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .rag import LocalVectorIndex, answer_question, build_index
from .scoring import compute_scores
from .monitoring import generate_alerts


ACTION_BY_FLAG = {
    "PRIORITY": "ADVANCE TO DD",
    "WATCH": "MONITOR 90 DAYS",
    "LOW": "DEPRIORITISE",
}


def get_system(dataset_dir: str | Path = "data/dataset") -> dict[str, Any]:
    index = build_index(dataset_dir)
    scores = compute_scores(dataset_dir)
    alerts = generate_alerts(scores, index)
    notes = {row["company_name"]: generate_company_note(row, alerts, index) for row in scores}
    return {"index": index, "scores": scores, "alerts": alerts, "notes": notes}


def _signals(company: str, index: LocalVectorIndex, positive: bool = True) -> list[dict[str, str]]:
    query = (
        f"{company} partnerships differentiation growth ESG certification positive signals"
        if positive
        else f"{company} key risks runway regulatory governance customer concentration technology risk"
    )
    results = index.search(query, top_k=5, company=company)
    rows = []
    for result in results:
        text = re.sub(r"\s+", " ", result["text"]).strip()
        rows.append(
            {
                "source": result["metadata"].get("source", "unknown"),
                "document_type": result["metadata"].get("document_type", "unknown"),
                "excerpt": text[:260],
            }
        )
    return rows[:3]


def recommended_action(score_flag: str) -> str:
    return ACTION_BY_FLAG[score_flag]


def fallback_thesis(row: dict[str, Any]) -> str:
    company = row["company_name"]
    return (
        f"{company} screens as {row['score_flag']} with a composite score of {row['total_score']}/100. "
        f"The strongest measurable factors are revenue CAGR of {row['revenue_cagr_2023_2025_pct']}%, "
        f"{row['gross_margin_pct']}% gross margin, and {row['runway_months']} months of runway. "
        f"The score also reflects technology, market, and ESG evidence from the company factsheet."
    )


def investment_thesis(company: str, row: dict[str, Any], index: LocalVectorIndex) -> str:
    answer = answer_question(f"Give an investment thesis for {company}", index, top_k=4, use_llm=False)["answer"]
    return fallback_thesis(row) if answer.startswith("I could not") else answer


def generate_company_note(row: dict[str, Any], alerts: list[dict[str, Any]], index: LocalVectorIndex) -> str:
    company = row["company_name"]
    company_alerts = [a for a in alerts if a["company_name"] == company]
    positives = _signals(company, index, positive=True)
    risks = _signals(company, index, positive=False)
    thesis = investment_thesis(company, row, index)

    lines = [
        f"# {company}",
        "",
        f"**Country:** {row['country']}  ",
        f"**Sub-sector:** {row['sub_sector']}  ",
        f"**Composite score:** {row['total_score']}/100  ",
        f"**Priority flag:** {row['score_flag']}",
        "",
        "## Investment thesis",
        thesis,
        "",
        "## Top 3 positive signals",
    ]
    for item in positives:
        lines.append(f"- {item['excerpt']} ({item['source']}, {item['document_type']})")
    lines.append("")
    lines.append("## Top 3 risks")
    for item in risks:
        lines.append(f"- {item['excerpt']} ({item['source']}, {item['document_type']})")
    lines.append("")
    lines.append("## Active alerts")
    if company_alerts:
        for alert in company_alerts:
            lines.append(f"- **{alert['alert_type']}**: {alert['trigger_value']} - {alert['recommended_action']}")
    else:
        lines.append("- No active monitoring alerts.")
    lines.extend(["", "## Recommended action", recommended_action(row["score_flag"])])
    return "\n".join(lines)
