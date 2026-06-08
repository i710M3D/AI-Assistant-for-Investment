from __future__ import annotations

import re
from typing import Any, Callable

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

NO_DIRECT_EVIDENCE = "No direct source evidence found in corpus."

ALERT_CONFIG = {
    "RUNWAY_CRITICAL": {
        "keywords": ["runway", "cash runway", "cash", "burn", "months", "financial snapshot"],
        "document_types": ["financials", "factsheet"],
    },
    "REVENUE_DECLINE": {
        "keywords": ["revenue growth", "decline", "negative growth", "revenue"],
        "document_types": ["financials", "factsheet"],
    },
    "ESG_ALERT": {
        "keywords": ["ESG", "environmental", "social", "governance", "unverified", "certification"],
        "document_types": ["factsheet", "report", "news"],
    },
    "GOVERNANCE_FLAG": {
        "keywords": ["CFO", "CEO", "resigned", "vacant", "key person", "governance"],
        "document_types": ["factsheet", "news"],
    },
    "FUNDRAISE_ACTIVE": {
        "keywords": ["fundraising", "Series A", "Series B", "Series C", "round", "targeting"],
        "document_types": ["news", "funding", "factsheet"],
    },
    "STRATEGIC_EXIT": {
        "keywords": ["acquisition", "M&A", "strategic sale", "buyer", "exit", "sale"],
        "document_types": ["news", "factsheet"],
    },
    "SCORE_PRIORITY": {
        "keywords": ["score", "growth", "market", "ESG", "technology"],
        "document_types": ["factsheet", "financials"],
    },
}

FUNDRAISING_PATTERN = re.compile(r"\b(?:Series [ABC]|fundraising|round|targeting)\b", re.I)
EXIT_PATTERN = re.compile(r"\b(?:acquisition|strategic sale|expressions of interest|sale)\b|M&A", re.I)
MANAGEMENT_CHANGE_PATTERN = re.compile(r"\b(?:(?:CFO|CEO) departure|key person risk|resigned|vacant)\b", re.I)

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

EVENT_DOCUMENT_TYPES = {"factsheet", "funding", "news"}
ACTIVE_FUNDRAISING_PATTERN = re.compile(
    r"\b(?:targeting|planned|preparing|launch|mandate|exploring|rumou?red|process expected)\b|\bfundraising (?:round|process)\b",
    re.I,
)
INACTIVE_FUNDRAISING_PATTERN = re.compile(r"\b(?:bridge round|flat valuation|investor hesitation|fundraising strategy)\b", re.I)
ACTIVE_EXIT_PATTERN = re.compile(
    r"\b(?:strategic sale|M&A advisor|M&A boutique|expressions of interest|acquisition opportunity|interest from strategics|sale process|exploring)\b",
    re.I,
)
NEGATED_EVENT_PATTERN = re.compile(
    r"\bno\s+(?:M&A activity|acquisition interest|M&A|acquisition|strategic sale|sale)\b|\bno\b.{0,80}\b(?:reported|to date)\b",
    re.I,
)
AMOUNT_PATTERN = re.compile(
    r"\b(?:EUR|GBP|USD|\$)\s?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s?(?:M|m|million|k|K)?\b",
    re.I,
)
TIMELINE_PATTERN = re.compile(
    r"\b(?:Q[1-4]\s+20\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}|20\d{2}-\d{2}-\d{2})\b",
    re.I,
)


def _clean_excerpt(text: str, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(keyword) for keyword in keywords), re.I)


def _contains_keyword(text: str, keywords: list[str]) -> bool:
    return bool(_keyword_pattern(keywords).search(text))


def _keyword_priority(text: str, keywords: list[str]) -> int:
    for idx, keyword in enumerate(keywords):
        if re.search(re.escape(keyword), text, re.I):
            return idx
    return len(keywords)


def extract_relevant_excerpt(text: str, keywords: list[str], window: int = 350) -> str | None:
    clean = _clean_excerpt(text, len(text))
    match = None
    for keyword in keywords:
        match = re.search(re.escape(keyword), clean, re.I)
        if match:
            break
    if not match:
        return None

    start = max(0, match.start() - window // 2)
    end = min(len(clean), match.end() + window // 2)
    excerpt = clean[start:end].strip(" -;,.")
    return excerpt or None


def _company_names(scores: list[dict[str, Any]], index: LocalVectorIndex) -> list[str]:
    names = [row["company_name"] for row in scores if row.get("company_name")]
    for chunk in index.chunks:
        company = chunk.metadata.get("company")
        if company:
            names.append(company)
    return list(dict.fromkeys(names))


def _result_from_chunk(chunk: Any) -> dict[str, Any]:
    return {"text": chunk.text, "metadata": chunk.metadata, "score": 1.0}


def _company_chunks(index: LocalVectorIndex, company: str) -> list[dict[str, Any]]:
    return [_result_from_chunk(chunk) for chunk in index.chunks if chunk.metadata.get("company") == company]


def _document_type_rank(document_type: str | None, preferred: list[str]) -> int:
    if document_type in preferred:
        return preferred.index(document_type)
    return len(preferred)


def _structured_evidence_payload(
    structured_evidence: dict[str, str] | None,
    keywords: list[str],
) -> tuple[str, str, str] | None:
    if not structured_evidence:
        return None

    excerpt = structured_evidence.get("source_excerpt", "")
    if not _contains_keyword(excerpt, keywords):
        return None
    return (
        structured_evidence.get("source_file", "structured_scoring_output"),
        structured_evidence.get("document_type", "structured"),
        excerpt,
    )


def get_alert_evidence(
    index: LocalVectorIndex,
    company_name: str,
    alert_type: str,
    query: str,
    structured_evidence: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    config = ALERT_CONFIG[alert_type]
    keywords = config["keywords"]
    preferred_types = config["document_types"]

    structured = _structured_evidence_payload(structured_evidence, keywords)
    if structured:
        return structured

    retrieved = index.search(query, top_k=10, company=company_name)
    company_chunks = _company_chunks(index, company_name)
    candidates = retrieved + [chunk for chunk in company_chunks if chunk not in retrieved]

    preferred = [result for result in candidates if result["metadata"].get("document_type") in preferred_types]
    if preferred:
        candidates = preferred

    keyword_hits = [result for result in candidates if _contains_keyword(result["text"], keywords)]
    keyword_hits.sort(
        key=lambda result: (
            _keyword_priority(result["text"], keywords),
            _document_type_rank(result["metadata"].get("document_type"), preferred_types),
            -float(result.get("score", 0)),
        )
    )

    for result in keyword_hits:
        excerpt = extract_relevant_excerpt(result["text"], keywords)
        if excerpt and _contains_keyword(excerpt, keywords):
            metadata = result["metadata"]
            return (
                metadata.get("source", "unknown"),
                metadata.get("document_type", "unknown"),
                excerpt,
            )

    return "N/A", "N/A", NO_DIRECT_EVIDENCE


def _company_event_chunks(index: LocalVectorIndex, company: str) -> list[dict[str, Any]]:
    chunks = [
        _result_from_chunk(chunk)
        for chunk in index.chunks
        if chunk.metadata.get("company") == company
        and chunk.metadata.get("document_type") in EVENT_DOCUMENT_TYPES
    ]
    if chunks:
        return sorted(chunks, key=lambda result: result["metadata"].get("document_type") != "news")

    results = index.search(f"{company} fundraising acquisition CEO CFO governance news factsheet", top_k=8, company=company)
    return [result for result in results if result["metadata"].get("document_type") in EVENT_DOCUMENT_TYPES]


def _excerpt_around_match(text: str, match: re.Match[str], limit: int = 600) -> str:
    units = [
        re.sub(r"\s+", " ", unit).strip(" -")
        for unit in re.split(r"(?<=[.!?])\s+|\n+", text)
        if unit.strip()
    ]
    if units:
        for idx, unit in enumerate(units):
            if match.group(0).lower() in unit.lower():
                start = max(0, idx - 1)
                end = min(len(units), idx + 8)
                return _clean_excerpt(" ".join(units[start:end]), limit)

    start = max(0, match.start() - limit // 2)
    end = min(len(text), match.end() + limit // 2)
    return _clean_excerpt(text[start:end], limit)


def _first_matching_event(
    chunks: list[dict[str, Any]],
    pattern: re.Pattern[str],
    validator: Callable[[str, dict[str, Any]], bool] | None = None,
) -> tuple[str, str, str, re.Match[str]] | None:
    for result in chunks:
        for match in pattern.finditer(result["text"]):
            metadata = result["metadata"]
            excerpt = _excerpt_around_match(result["text"], match)
            if validator is None or validator(excerpt, result):
                return metadata.get("source", "unknown"), metadata.get("document_type", "unknown"), excerpt, match
    return None


def _trusted_company_context(excerpt: str, result: dict[str, Any], company: str) -> bool:
    metadata = result["metadata"]
    return _mentions_company(excerpt, company) or (
        metadata.get("company") == company and metadata.get("document_type") == "factsheet"
    )


def _is_actionable_fundraising(excerpt: str, result: dict[str, Any], company: str) -> bool:
    return (
        _trusted_company_context(excerpt, result, company)
        and bool(ACTIVE_FUNDRAISING_PATTERN.search(excerpt))
        and not INACTIVE_FUNDRAISING_PATTERN.search(excerpt)
    )


def _is_actionable_exit(excerpt: str, result: dict[str, Any], company: str) -> bool:
    return (
        _trusted_company_context(excerpt, result, company)
        and bool(ACTIVE_EXIT_PATTERN.search(excerpt))
        and not NEGATED_EVENT_PATTERN.search(excerpt)
    )


def _is_actionable_management_change(excerpt: str, result: dict[str, Any], company: str) -> bool:
    return _trusted_company_context(excerpt, result, company) and _contains_keyword(
        excerpt,
        ALERT_CONFIG["GOVERNANCE_FLAG"]["keywords"],
    )


def _extract_amount_timeline(excerpt: str, fallback: str = "Detected in source text") -> str:
    amount = AMOUNT_PATTERN.search(excerpt)
    timelines = TIMELINE_PATTERN.findall(excerpt)
    parts = []
    if amount:
        parts.append(amount.group(0))
    if timelines:
        quarter_timeline = next((timeline for timeline in timelines if timeline.upper().startswith("Q")), None)
        parts.append(quarter_timeline or timelines[-1])
    return " / ".join(parts) if parts else fallback


def _has_alert(alerts: list[AlertRow], company: str, alert_type: str) -> bool:
    return any(alert["company_name"] == company and alert["alert_type"] == alert_type for alert in alerts)


def _mentions_company(excerpt: str, company: str) -> bool:
    company_alias = re.sub(r"\b(sa|ltd|gmbh|bv)\b\.?", "", company.lower()).strip()
    text = excerpt.lower()
    return company.lower() in text or bool(company_alias and company_alias in text)


def _latest_financial_result(index: LocalVectorIndex, company: str) -> dict[str, Any] | None:
    financial_rows = [
        result
        for result in _company_chunks(index, company)
        if result["metadata"].get("document_type") == "financials"
    ]
    if not financial_rows:
        return None
    return max(financial_rows, key=lambda result: float(result["metadata"].get("date") or 0))


def _field_value(text: str, field: str) -> str | None:
    match = re.search(rf"\b{re.escape(field)}:\s*([^;]+)", text)
    return match.group(1).strip() if match else None


def _financial_structured_evidence(index: LocalVectorIndex, company: str, alert_type: str) -> dict[str, str] | None:
    result = _latest_financial_result(index, company)
    if not result:
        return None

    text = result["text"]
    metadata = result["metadata"]
    if alert_type == "RUNWAY_CRITICAL":
        runway = _field_value(text, "runway_months")
        cash = _field_value(text, "cash_eur_k")
        burn = _field_value(text, "burn_eur_k_monthly")
        if runway and burn:
            return {
                "source_file": metadata.get("source", "unknown"),
                "document_type": "financials",
                "source_excerpt": f"CSV financials show runway_months = {runway}, cash = {cash or 'N/A'}, monthly burn = {burn}.",
            }

    if alert_type == "REVENUE_DECLINE":
        revenue_growth = _field_value(text, "revenue_growth_pct")
        if revenue_growth:
            return {
                "source_file": metadata.get("source", "unknown"),
                "document_type": "financials",
                "source_excerpt": f"CSV financials show revenue_growth_pct = {revenue_growth}.",
            }

    return None


def add_alert(
    alerts: list[AlertRow],
    company: str,
    alert_type: str,
    trigger: str,
    source_file: str,
    document_type: str,
    source_excerpt: str,
) -> None:
    if source_excerpt != NO_DIRECT_EVIDENCE and not _contains_keyword(source_excerpt, ALERT_CONFIG[alert_type]["keywords"]):
        source_file = "N/A"
        document_type = "N/A"
        source_excerpt = NO_DIRECT_EVIDENCE

    alerts.append(
        {
            "company_name": company,
            "alert_type": alert_type,
            "trigger_value": trigger,
            "source_file": source_file,
            "document_type": document_type,
            "source_excerpt": source_excerpt,
            "recommended_action": RECOMMENDATIONS[alert_type],
        }
    )


def _add_scorecard_alerts(alerts: list[AlertRow], row: dict[str, Any], index: LocalVectorIndex) -> None:
    company = row["company_name"]

    if row["runway_months"] < 12:
        source_file, document_type, source_excerpt = get_alert_evidence(
            index,
            company,
            "RUNWAY_CRITICAL",
            f"{company} runway cash burn months financial snapshot",
            structured_evidence=_financial_structured_evidence(index, company, "RUNWAY_CRITICAL"),
        )
        add_alert(alerts, company, "RUNWAY_CRITICAL", f"{row['runway_months']} months", source_file, document_type, source_excerpt)

    if row["revenue_growth_pct"] < 0:
        source_file, document_type, source_excerpt = get_alert_evidence(
            index,
            company,
            "REVENUE_DECLINE",
            f"{company} revenue growth decline negative growth revenue",
            structured_evidence=_financial_structured_evidence(index, company, "REVENUE_DECLINE"),
        )
        add_alert(alerts, company, "REVENUE_DECLINE", f"{row['revenue_growth_pct']}%", source_file, document_type, source_excerpt)

    if row["esg_score"] < 15:
        source_file, document_type, source_excerpt = get_alert_evidence(
            index,
            company,
            "ESG_ALERT",
            f"{company} ESG environmental social governance unverified certification",
        )
        add_alert(alerts, company, "ESG_ALERT", f"ESG dimension {row['esg_score']}/25", source_file, document_type, source_excerpt)

    if row["total_score"] >= 70:
        score_evidence = {
            "source_file": "scoring_output",
            "document_type": "computed_score",
            "source_excerpt": (
                f"Scoring output shows score = {row['total_score']}, growth = {row['revenue_growth_pct']}%, "
                f"market score = {row['market_score']}/25, ESG score = {row['esg_score']}/25, "
                f"technology score = {row['technology_score']}/25."
            ),
        }
        source_file, document_type, source_excerpt = get_alert_evidence(
            index,
            company,
            "SCORE_PRIORITY",
            f"{company} score growth market ESG technology",
            structured_evidence=score_evidence,
        )
        add_alert(alerts, company, "SCORE_PRIORITY", f"Score {row['total_score']}", source_file, document_type, source_excerpt)


def _add_event_alerts(alerts: list[AlertRow], company: str, index: LocalVectorIndex) -> None:
    chunks = _company_event_chunks(index, company)

    fundraising = _first_matching_event(
        chunks,
        FUNDRAISING_PATTERN,
        lambda excerpt, result: _is_actionable_fundraising(excerpt, result, company),
    )
    if fundraising and not _has_alert(alerts, company, "FUNDRAISE_ACTIVE"):
        source_file, document_type, source_excerpt, _ = fundraising
        trigger = _extract_amount_timeline(source_excerpt, "Fundraising process detected")
        add_alert(alerts, company, "FUNDRAISE_ACTIVE", trigger, source_file, document_type, source_excerpt)

    exit_event = _first_matching_event(
        chunks,
        EXIT_PATTERN,
        lambda excerpt, result: _is_actionable_exit(excerpt, result, company),
    )
    if exit_event and not _has_alert(alerts, company, "STRATEGIC_EXIT"):
        source_file, document_type, source_excerpt, _ = exit_event
        trigger = _extract_amount_timeline(source_excerpt, "M&A / strategic options detected")
        add_alert(alerts, company, "STRATEGIC_EXIT", trigger, source_file, document_type, source_excerpt)

    management_change = _first_matching_event(
        chunks,
        MANAGEMENT_CHANGE_PATTERN,
        lambda excerpt, result: _is_actionable_management_change(excerpt, result, company),
    )
    if management_change and not _has_alert(alerts, company, "GOVERNANCE_FLAG"):
        source_file, document_type, source_excerpt, match = management_change
        trigger = _extract_amount_timeline(source_excerpt, f"Management signal: {match.group(0)}")
        add_alert(alerts, company, "GOVERNANCE_FLAG", trigger, source_file, document_type, source_excerpt)


def _add_legacy_fundraising_alerts(alerts: list[AlertRow], index: LocalVectorIndex) -> None:
    for company, round_info in FUNDRAISING_ROUNDS.items():
        if _has_alert(alerts, company, "FUNDRAISE_ACTIVE"):
            continue
        source_file, document_type, source_excerpt = get_alert_evidence(
            index,
            company,
            "FUNDRAISE_ACTIVE",
            f"{company} {round_info['query']}",
        )
        if source_excerpt != NO_DIRECT_EVIDENCE and FUNDRAISING_PATTERN.search(source_excerpt):
            trigger = _extract_amount_timeline(source_excerpt, round_info["amount"])
            add_alert(alerts, company, "FUNDRAISE_ACTIVE", trigger, source_file, document_type, source_excerpt)


def _add_legacy_exit_alerts(alerts: list[AlertRow], index: LocalVectorIndex) -> None:
    for company in EXIT_PROCESS_COMPANIES:
        if _has_alert(alerts, company, "STRATEGIC_EXIT"):
            continue
        source_file, document_type, source_excerpt = get_alert_evidence(
            index,
            company,
            "STRATEGIC_EXIT",
            f"{company} acquisition talks strategic sale M&A expressions of interest",
        )
        if source_excerpt != NO_DIRECT_EVIDENCE and EXIT_PATTERN.search(source_excerpt):
            add_alert(
                alerts,
                company,
                "STRATEGIC_EXIT",
                "M&A / strategic options detected",
                source_file,
                document_type,
                source_excerpt,
            )


def generate_alerts(scores: list[dict[str, Any]], index: LocalVectorIndex) -> list[AlertRow]:
    alerts: list[AlertRow] = []
    for row in scores:
        _add_scorecard_alerts(alerts, row, index)

    for company in _company_names(scores, index):
        _add_event_alerts(alerts, company, index)

    _add_legacy_fundraising_alerts(alerts, index)
    _add_legacy_exit_alerts(alerts, index)
    return alerts
