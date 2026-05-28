from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from .ingestion import COMPANY_NAMES

FINANCIAL_FIELDS = [
    "fy_year",
    "revenue_eur_k",
    "revenue_growth_pct",
    "gross_margin_pct",
    "ebitda_eur_k",
    "arr_eur_k",
    "cash_eur_k",
    "burn_eur_k_monthly",
    "runway_months",
    "fte_count",
]


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _norm(value: float, low: float, high: float) -> float:
    if high == low:
        return 0
    return _clamp((value - low) / (high - low) * 100)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required scoring CSV not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def load_financials(dataset_dir: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(Path(dataset_dir) / "market_data" / "companies_financials_2023_2025.csv")
    for row in rows:
        for field in FINANCIAL_FIELDS:
            row[field] = float(row[field])
    return rows


def load_funding(dataset_dir: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(Path(dataset_dir) / "market_data" / "funding_rounds.csv")
    for row in rows:
        row["amount_eur_m"] = float(row["amount_eur_m"])
        row["post_money_valuation_eur_m"] = float(row["post_money_valuation_eur_m"])
    return rows


def latest_by_company(financials: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in financials:
        name = row["company_name"]
        if name not in latest or row["fy_year"] > latest[name]["fy_year"]:
            latest[name] = row
    return latest


def revenue_cagr(company_rows: list[dict[str, Any]]) -> float:
    rows = sorted(company_rows, key=lambda r: r["fy_year"])
    start, end = rows[0]["revenue_eur_k"], rows[-1]["revenue_eur_k"]
    periods = max(1, len(rows) - 1)
    if start <= 0:
        return 0.0
    return ((end / start) ** (1 / periods) - 1) * 100


def factsheet_text(dataset_dir: str | Path, company: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", company.lower()).strip("_")
    path = Path(dataset_dir) / "companies" / f"{slug}_factsheet.txt"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def extract_score_inputs(text: str) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([a-zA-Z0-9_]+)\s*:\s*(.+?)\s*$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key in {"market_coverage_ha", "esg_score_internal", "fte_count"}:
            inputs[key] = float(re.sub(r"[^0-9.\-]", "", value) or 0)
        elif key == "has_patent":
            inputs[key] = value.lower().startswith("true")
        else:
            inputs[key] = value
    return inputs


def keyword_points(text: str, keywords: list[str], max_points: float) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return min(max_points, hits * (max_points / max(1, len(keywords))))


def financial_score(row: dict[str, Any], cagr: float) -> float:
    growth = _norm(cagr, -10, 100) * 0.35
    margin = _norm(row["gross_margin_pct"], 35, 80) * 0.25
    runway = _norm(min(row["runway_months"], 36), 6, 36) * 0.25
    burn_eff = _norm(row["revenue_eur_k"] / max(row["burn_eur_k_monthly"] * 12, 1), 0.5, 3.0) * 0.15
    return round((growth + margin + runway + burn_eff) * 0.25, 2)


def technology_score(text: str, inputs: dict[str, Any]) -> float:
    score = 0.0
    score += 5 if inputs.get("has_patent") else keyword_points(text, ["patent", "proprietary"], 5)
    score += keyword_points(text, ["differentiation", "unique", "only company", "rare"], 5)
    score += keyword_points(text, ["dataset", "data moat", "library", "sensor network", "hectares"], 5)
    score += keyword_points(text, ["integration", "integrates", "API", "SAP", "John Deere", "Claas"], 5)
    score += keyword_points(text, ["accuracy", "precision", "validated", "field trial", "prediction"], 5)
    if "blockchain" in text.lower() and "overengineered" in text.lower():
        score -= 4
    return round(_clamp(score, 0, 25), 2)


def market_score(text: str, row: dict[str, Any]) -> float:
    score = 0.0
    score += keyword_points(text, ["addressable market", "CAGR", "tailwind", "fastest-growing", "Tier 1"], 5)
    score += _norm(row["revenue_eur_k"], 300, 20000) * 0.07
    score += keyword_points(text, ["France", "Spain", "Italy", "DACH", "Morocco", "global", "Europe", "Benelux"], 5)
    score += keyword_points(text, ["market share", "competitive", "competitors", "differentiation"], 4)
    score += keyword_points(text, ["partnership", "LOI", "Bayer", "OCP", "Cosun", "Credit"], 5)
    if row["sub_sector"] in {"Water Management", "Biological Inputs", "Precision Agriculture"}:
        score += 3
    if "revenue declined" in text.lower() or row["revenue_growth_pct"] < 0:
        score -= 4
    return round(_clamp(score, 0, 25), 2)


def esg_score(text: str, inputs: dict[str, Any]) -> float:
    internal = float(inputs.get("esg_score_internal", 55))
    score = internal / 4
    lower = text.lower()
    if "not independently verified" in lower or "unverified" in lower:
        score -= 4
    if "iso 14001" in lower or "third-party" in lower or "validated" in lower:
        score += 2
    if "cfo" in lower and ("vacant" in lower or "resigned" in lower):
        score -= 3
    return round(_clamp(score, 0, 25), 2)


def flag_from_score(score: float) -> str:
    if score >= 70:
        return "PRIORITY"
    if score >= 50:
        return "WATCH"
    return "LOW"


def score_company(
    company: str,
    latest_row: dict[str, Any],
    company_financials: list[dict[str, Any]],
    dataset_dir: str | Path,
) -> dict[str, Any]:
    text = factsheet_text(dataset_dir, company)
    inputs = extract_score_inputs(text)
    cagr = revenue_cagr(company_financials)

    financial = financial_score(latest_row, cagr)
    technology = technology_score(text, inputs)
    market = market_score(text, latest_row)
    esg = esg_score(text, inputs)
    total = round(financial + technology + market + esg, 2)

    return {
        "company_name": company,
        "company_id": latest_row["company_id"],
        "country": latest_row["country"],
        "sub_sector": latest_row["sub_sector"],
        "revenue_2025_eur_k": latest_row["revenue_eur_k"],
        "revenue_growth_pct": latest_row["revenue_growth_pct"],
        "revenue_cagr_2023_2025_pct": round(cagr, 1),
        "gross_margin_pct": latest_row["gross_margin_pct"],
        "runway_months": latest_row["runway_months"],
        "burn_eur_k_monthly": latest_row["burn_eur_k_monthly"],
        "financial_score": financial,
        "technology_score": technology,
        "market_score": market,
        "esg_score": esg,
        "total_score": total,
        "score_flag": flag_from_score(total),
    }


def compute_scores(dataset_dir: str | Path = "data/dataset") -> list[dict[str, Any]]:
    financials = load_financials(dataset_dir)
    latest = latest_by_company(financials)
    by_company = {name: [r for r in financials if r["company_name"] == name] for name in latest}
    rows: list[dict[str, Any]] = []
    for company in COMPANY_NAMES:
        if company not in latest:
            raise ValueError(f"Missing financial rows for required company: {company}")
        rows.append(score_company(company, latest[company], by_company[company], dataset_dir))
    return sorted(rows, key=lambda r: r["total_score"], reverse=True)
