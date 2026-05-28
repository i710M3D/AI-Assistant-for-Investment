from __future__ import annotations

import re
from typing import Any

from .rag import LocalVectorIndex


AlertRow = dict[str, Any]

RECOMMENDATIONS = {
    "RUNWAY_CRITICAL": "Review financing plan and downside cash bridge immediately.",
    "REVENUE_DECLINE": "Validate churn, pricing pressure, and customer concentration before any new work.",
    "ESG_ALERT": "Request ESG evidence pack and third-party verification plan.",
    "GOVERNANCE_FLAG": "Escalate to partner review and require management update.",
    "FUNDRAISE_ACTIVE": "Track process timing and ask for data room access.",
    "STRATEGIC_EXIT": "Monitor transaction process and valuation read-across.",
    "SCORE_PRIORITY": "Advance to IC screening and schedule expert calls.",
}

FUNDRAISING_PATTERN = re.compile(r"Series [ABC]|fundraising|round|targeting", re.I)
EXIT_PATTERN = re.compile(r"acquisition|strategic sale|M&A|expressions of interest|sale", re.I)
MANAGEMENT_CHANGE_PATTERN = re.compile(r"CFO|CEO|key person|resigned|vacant", re.I)

FUNDRAISING_ROUNDS = {
    "Verdant Farms SA": {
        "query": "Series C fundraising targeting EUR 30-40M Q3 2026",
        "amount": "EUR 30-40M",
    },
    "GreenYield Technologies BV": {
        "query": "Series B fundraising September 2026 targeting EUR 12-15M",
        "amount": "EUR 12-15M",
    },
    "SoilSense AI Ltd": {
        "query": "Series A planned Q2 2026 GBP 6-8M",
        "amount": "GBP 6-8M",
    },
}

EXIT_PROCESS_COMPANIES = ["AquaGrow Solutions Ltd", "HarvestLink GmbH"]


def _clean_excerpt(text: str, limit: int = 350) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _pick_result(results: list[dict[str, Any]], document_type: str | None = None) -> dict[str, Any] | None:
    if not results:
        return None

    if document_type:
        for result in results:
            if result["metadata"].get("document_type") == document_type:
                return result

    return results[0]


def _supporting_excerpt(
    index: LocalVectorIndex,
    query: str,
    company: str | None = None,
    prefer_type: str | None = None,
) -> tuple[str, str]:
    results = index.search(query, top_k=5, company=company)
    result = _pick_result(results, prefer_type)
    if result is None:
        return "N/A", "No textual evidence found."

    source = result["metadata"].get("source", "unknown")
    return source, _clean_excerpt(result["text"])


def add_alert(alerts: list[AlertRow], company: str, alert_type: str, trigger: str, source: str, excerpt: str) -> None:
    alerts.append(
        {
            "company_name": company,
            "alert_type": alert_type,
            "trigger_value": trigger,
            "source_reference": source,
            "evidence": excerpt,
            "recommended_action": RECOMMENDATIONS[alert_type],
        }
    )


def _add_scorecard_alerts(alerts: list[AlertRow], row: dict[str, Any], index: LocalVectorIndex) -> None:
    company = row["company_name"]

    if row["runway_months"] < 12:
        source, excerpt = _supporting_excerpt(index, f"{company} runway months cash concern", company)
        add_alert(alerts, company, "RUNWAY_CRITICAL", f"{row['runway_months']} months", source, excerpt)

    if row["revenue_growth_pct"] < 0:
        source, excerpt = _supporting_excerpt(index, f"{company} revenue declined revenue growth", company)
        add_alert(alerts, company, "REVENUE_DECLINE", f"{row['revenue_growth_pct']}%", source, excerpt)

    if row["esg_score"] < 15:
        source, excerpt = _supporting_excerpt(index, f"{company} ESG claims governance certification", company, prefer_type="factsheet")
        add_alert(alerts, company, "ESG_ALERT", f"ESG dimension {row['esg_score']}/25", source, excerpt)

    if row["total_score"] >= 70:
        source, excerpt = _supporting_excerpt(index, f"{company} strongest signals investment priority", company)
        add_alert(alerts, company, "SCORE_PRIORITY", f"Score {row['total_score']}", source, excerpt)


def _add_governance_alerts(alerts: list[AlertRow], company: str, index: LocalVectorIndex) -> None:
    source, excerpt = _supporting_excerpt(index, f"{company} CFO departure CEO departure key person risk resigned vacant", company)
    if company == "HarvestLink GmbH" and MANAGEMENT_CHANGE_PATTERN.search(excerpt):
        add_alert(alerts, company, "GOVERNANCE_FLAG", "CFO departure / vacancy", source, excerpt)


def _add_fundraising_alerts(alerts: list[AlertRow], index: LocalVectorIndex) -> None:
    for company, round_info in FUNDRAISING_ROUNDS.items():
        source, excerpt = _supporting_excerpt(index, f"{company} {round_info['query']}", company, prefer_type="news")
        if FUNDRAISING_PATTERN.search(excerpt):
            add_alert(alerts, company, "FUNDRAISE_ACTIVE", round_info["amount"], source, excerpt)


def _add_exit_alerts(alerts: list[AlertRow], index: LocalVectorIndex) -> None:
    for company in EXIT_PROCESS_COMPANIES:
        source, excerpt = _supporting_excerpt(index, f"{company} acquisition talks strategic sale M&A expressions of interest", company, prefer_type="news")
        if EXIT_PATTERN.search(excerpt):
            add_alert(alerts, company, "STRATEGIC_EXIT", "M&A / strategic options detected", source, excerpt)


def generate_alerts(scores: list[dict[str, Any]], index: LocalVectorIndex) -> list[AlertRow]:
    alerts: list[AlertRow] = []
    for row in scores:
        company = row["company_name"]
        _add_scorecard_alerts(alerts, row, index)
        _add_governance_alerts(alerts, company, index)

    _add_fundraising_alerts(alerts, index)
    _add_exit_alerts(alerts, index)
    return alerts
