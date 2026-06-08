from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .rag import LocalVectorIndex


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
LLM_DISABLED = False

ACTION_BY_FLAG = {
    "PRIORITY": "ADVANCE TO DD",
    "WATCH": "MONITOR 90 DAYS",
    "LOW": "DEPRIORITISE",
}

NOTE_PROMPT = (
    "Write a concise investment note using ONLY the provided evidence snippets. "
    "Do not invent facts. Do not include raw document headers. Do not include CSV rows unless summarized. "
    "If evidence is insufficient, say so."
)

THESIS_QUERY = "{company_name} business model revenue growth market traction technology ESG investment thesis"
POSITIVE_QUERY = "{company_name} revenue growth partnerships patents IP validation ESG impact customers traction"
RISK_QUERY = "{company_name} risks runway competition regulatory governance dependency unverified adoption"

POSITIVE_KEYWORDS = [
    "growth",
    "revenue",
    "partnership",
    "LOI",
    "patent",
    "proprietary",
    "validated",
    "accuracy",
    "ESG",
    "impact",
    "customer",
    "traction",
    "margin",
]

RISK_KEYWORDS = [
    "risk",
    "runway",
    "burn",
    "competition",
    "regulatory",
    "dependency",
    "unverified",
    "governance",
    "CFO",
    "CEO",
    "adoption",
    "concentration",
    "decline",
]

THESIS_KEYWORDS = [
    "business model",
    "revenue",
    "growth",
    "market",
    "traction",
    "technology",
    "ESG",
    "partnership",
    "risk",
]

RAW_HEADER_PATTERN = re.compile(
    r"\b(?:COMPANY FACTSHEET|Document type|Source\s*:|Analyst\s*:|OVERVIEW|FINANCIALS|TEAM|KEY RISKS)\b",
    re.I,
)
CSV_ROW_PATTERN = re.compile(r"\bcompany_id:|fy_year:|revenue_eur_k:|burn_eur_k_monthly:", re.I)


def recommended_action(score_flag: str) -> str:
    return ACTION_BY_FLAG[score_flag]


def clean_chunk_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[=\-]{6,}", stripped):
            continue
        if RAW_HEADER_PATTERN.fullmatch(stripped.strip(" -:")):
            continue
        lines.append(stripped)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\bCOMPANY FACTSHEET\s+[A-Z\s&.-]+", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_sentences_around_keywords(text: str, keywords: list[str], max_sentences: int = 2) -> list[str]:
    cleaned = clean_chunk_text(text)
    sentences = [
        sentence.strip(" -")
        for sentence in re.split(r"(?<=[.!?])\s+|\s+-\s+|\n+", cleaned)
        if len(sentence.strip()) > 25
    ]
    matches = []
    for sentence in sentences:
        if any(re.search(re.escape(keyword), sentence, re.I) for keyword in keywords):
            if RAW_HEADER_PATTERN.search(sentence) or CSV_ROW_PATTERN.search(sentence):
                continue
            matches.append(sentence)
        if len(matches) >= max_sentences:
            break
    return matches


def _source_id(result: dict[str, Any]) -> tuple[str | None, int | None]:
    metadata = result["metadata"]
    return metadata.get("source"), metadata.get("chunk_id")


def _retrieved_evidence(
    company: str,
    index: LocalVectorIndex,
    query: str,
    keywords: list[str],
    max_items: int = 3,
    allow_funding: bool = False,
) -> list[dict[str, str]]:
    results = index.search(query, top_k=10, company=company)
    evidence = []
    seen_sources: set[str] = set()
    seen_text: set[str] = set()
    seen_chunks: set[tuple[str | None, int | None]] = set()

    for result in results:
        metadata = result["metadata"]
        doc_type = metadata.get("document_type", "unknown")
        if doc_type == "funding" and not allow_funding:
            continue
        identity = _source_id(result)
        if identity in seen_chunks:
            continue

        snippets = extract_sentences_around_keywords(result["text"], keywords, max_sentences=2)
        if not snippets:
            continue
        snippet = " ".join(snippets)
        normalized = re.sub(r"\W+", " ", snippet.lower()).strip()
        source = metadata.get("source", "unknown")
        if normalized in seen_text or source in seen_sources and len(evidence) >= 2:
            continue

        evidence.append(
            {
                "text": snippet,
                "source": source,
                "document_type": doc_type,
            }
        )
        seen_chunks.add(identity)
        seen_text.add(normalized)
        seen_sources.add(source)
        if len(evidence) >= max_items:
            break

    return evidence


def _fallback_thesis(row: dict[str, Any], evidence: list[dict[str, str]]) -> str:
    company = row["company_name"]
    thesis = [
        f"{company} screens as {row['score_flag']} with a composite score of {row['total_score']}/100.",
        (
            f"The financial profile shows {row['revenue_cagr_2023_2025_pct']}% revenue CAGR, "
            f"{row['gross_margin_pct']}% gross margin, and {row['runway_months']} months of runway."
        ),
    ]
    if evidence:
        thesis.append(f"Source evidence highlights {evidence[0]['text'].rstrip('.')}.")
    else:
        thesis.append("The retrieved corpus does not provide enough clean thesis evidence beyond the scoring outputs.")
    thesis.append(f"The recommended screening action is {recommended_action(row['score_flag'])}.")
    return " ".join(thesis)


def _llm_rewrite(kind: str, company: str, evidence: list[dict[str, str]], row: dict[str, Any]) -> str | None:
    global LLM_DISABLED
    if LLM_DISABLED or not OPENAI_API_KEY or not evidence:
        return None

    snippets = "\n".join(
        f"- {item['text']} ({item['source']}, {item['document_type']})"
        for item in evidence
    )
    instruction = (
        "Write 3-4 clean sentences for the Investment thesis section."
        if kind == "thesis"
        else "Rewrite each evidence snippet as one concise analyst bullet. Keep the source citation at the end of each bullet."
    )
    try:
        client = OpenAI(timeout=8.0, max_retries=0)
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "developer", "content": NOTE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Company: {company}\n"
                        f"Score: {row['total_score']}/100; flag: {row['score_flag']}\n"
                        f"Task: {instruction}\n\nEvidence snippets:\n{snippets}"
                    ),
                },
            ],
        )
        text = clean_chunk_text(response.output_text)
        if "COMPANY FACTSHEET" in text or "====" in text or CSV_ROW_PATTERN.search(text):
            return None
        return text
    except Exception:
        LLM_DISABLED = True
        return None


def _bullet_from_evidence(item: dict[str, str]) -> str:
    text = clean_chunk_text(item["text"]).rstrip(".")
    if not text:
        text = "No strong source-backed signal found"
    return f"{text}. ({item['source']}, {item['document_type']})"


def _bullets(company: str, evidence: list[dict[str, str]], row: dict[str, Any], kind: str) -> list[str]:
    rewritten = _llm_rewrite(kind, company, evidence, row)
    if rewritten:
        lines = [line.strip(" -") for line in rewritten.splitlines() if line.strip()]
        cited = [line for line in lines if "(" in line and ")" in line]
        if len(cited) >= min(3, len(evidence)):
            return [f"- {line}" if not line.startswith("-") else line for line in cited[:3]]

    bullets = [f"- {_bullet_from_evidence(item)}" for item in evidence[:3]]
    while len(bullets) < 3:
        bullets.append("- No strong source-backed signal found.")
    return bullets


def generate_company_note(row: dict[str, Any], alerts: list[dict[str, Any]], index: LocalVectorIndex) -> str:
    company = row["company_name"]
    company_alerts = [a for a in alerts if a["company_name"] == company]

    thesis_evidence = _retrieved_evidence(
        company,
        index,
        THESIS_QUERY.format(company_name=company),
        THESIS_KEYWORDS,
        max_items=4,
    )
    positives = _retrieved_evidence(
        company,
        index,
        POSITIVE_QUERY.format(company_name=company),
        POSITIVE_KEYWORDS,
        max_items=3,
    )
    risks = _retrieved_evidence(
        company,
        index,
        RISK_QUERY.format(company_name=company),
        RISK_KEYWORDS,
        max_items=3,
    )

    thesis = _llm_rewrite("thesis", company, thesis_evidence, row) or _fallback_thesis(row, thesis_evidence)

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
        *_bullets(company, positives, row, "positives"),
        "",
        "## Top 3 risks",
        *_bullets(company, risks, row, "risks"),
        "",
        "## Active alerts",
    ]
    if company_alerts:
        for alert in company_alerts:
            lines.append(f"- **{alert['alert_type']}**: {alert['trigger_value']} - {alert['recommended_action']}")
    else:
        lines.append("- No active monitoring alerts.")

    lines.extend(["", "## Recommended action", recommended_action(row["score_flag"])])
    return "\n".join(lines)
