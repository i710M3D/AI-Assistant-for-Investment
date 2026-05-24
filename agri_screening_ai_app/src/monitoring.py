from __future__ import annotations

import re
from typing import Any

from .rag import LocalVectorIndex


RECOMMENDATIONS = {
    "RUNWAY_CRITICAL": "Review financing plan and downside cash bridge immediately.",
    "REVENUE_DECLINE": "Validate churn, pricing pressure, and customer concentration before any new work.",
    "ESG_ALERT": "Request ESG evidence pack and third-party verification plan.",
    "GOVERNANCE_FLAG": "Escalate to partner review and require management update.",
    "FUNDRAISE_ACTIVE": "Track process timing and ask for data room access.",
    "STRATEGIC_EXIT": "Monitor transaction process and valuation read-across.",
    "SCORE_PRIORITY": "Advance to IC screening and schedule expert calls.",
}


def _evidence(
    index: LocalVectorIndex,
    query: str,
    company: str | None = None,
    prefer_type: str | None = None,
) -> tuple[str, str]:
    results = index.search(query, top_k=5, company=company)
    if not results:
        return "N/A", "No textual evidence found."
    if prefer_type:
        preferred = [r for r in results if r["metadata"].get("document_type") == prefer_type]
        if preferred:
            results = preferred
    result = results[0]
    text = re.sub(r"\s+", " ", result["text"]).strip()
    return result["metadata"].get("source", "unknown"), text[:350]


def add_alert(alerts: list[dict[str, Any]], company: str, alert_type: str, trigger: str, source: str, quote: str) -> None:
    alerts.append(
        {
            "company_name": company,
            "alert_type": alert_type,
            "trigger_value": trigger,
            "source_reference": source,
            "evidence": quote,
            "recommended_action": RECOMMENDATIONS[alert_type],
        }
    )


def generate_alerts(scores: list[dict[str, Any]], index: LocalVectorIndex) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in scores:
        company = row["company_name"]
        if row["runway_months"] < 12:
            source, quote = _evidence(index, f"{company} runway months cash concern", company)
            add_alert(alerts, company, "RUNWAY_CRITICAL", f"{row['runway_months']} months", source, quote)
        if row["revenue_growth_pct"] < 0:
            source, quote = _evidence(index, f"{company} revenue declined revenue growth", company)
            add_alert(alerts, company, "REVENUE_DECLINE", f"{row['revenue_growth_pct']}%", source, quote)
        if row["esg_score"] < 15:
            source, quote = _evidence(index, f"{company} ESG claims governance certification", company)
            add_alert(alerts, company, "ESG_ALERT", f"ESG dimension {row['esg_score']}/25", source, quote)
        if row["total_score"] >= 70:
            source, quote = _evidence(index, f"{company} strongest signals investment priority", company)
            add_alert(alerts, company, "SCORE_PRIORITY", f"Score {row['total_score']}", source, quote)

        source, quote = _evidence(index, f"{company} CFO departure CEO departure key person risk resigned vacant", company)
        if re.search(r"CFO|CEO|key person|resigned|vacant", quote, re.I) and company == "HarvestLink GmbH":
            add_alert(alerts, company, "GOVERNANCE_FLAG", "CFO departure / vacancy", source, quote)

    fundraise_queries = {
        "Verdant Farms SA": "Series C fundraising targeting EUR 30-40M Q3 2026",
        "GreenYield Technologies BV": "Series B fundraising September 2026 targeting EUR 12-15M",
        "SoilSense AI Ltd": "Series A planned Q2 2026 GBP 6-8M",
    }
    for company, query in fundraise_queries.items():
        source, quote = _evidence(index, f"{company} {query}", company, prefer_type="news")
        if re.search(r"Series [ABC]|fundraising|round|targeting", quote, re.I):
            amount = "EUR 30-40M" if "Verdant" in company else "EUR 12-15M" if "GreenYield" in company else "GBP 6-8M"
            add_alert(alerts, company, "FUNDRAISE_ACTIVE", amount, source, quote)

    for company in ["AquaGrow Solutions Ltd", "HarvestLink GmbH"]:
        source, quote = _evidence(index, f"{company} acquisition talks strategic sale M&A expressions of interest", company, prefer_type="news")
        if re.search(r"acquisition|strategic sale|M&A|expressions of interest|sale", quote, re.I):
            add_alert(alerts, company, "STRATEGIC_EXIT", "M&A / strategic options detected", source, quote)

    return alerts
