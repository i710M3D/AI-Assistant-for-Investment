from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMPANY_NAMES = [
    "Verdant Farms SA",
    "GreenYield Technologies BV",
    "SoilSense AI Ltd",
    "AquaGrow Solutions Ltd",
    "HarvestLink GmbH",
    "BioRoot Innovations SA",
]


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, Any]


def normalise_text(text: str) -> str:
    replacements = {
        "â€”": "-",
        "â€“": "-",
        "ÃƒÂ©": "e",
        "Ã©": "e",
        "ÃƒÂ¨": "e",
        "Ã¨": "e",
        "ÃƒÂ´": "o",
        "Ã´": "o",
        "Ãƒâ€°": "E",
        "Ã‰": "E",
        "Ã‚Â±": "+/-",
        "Â±": "+/-",
        "Ã‚Â°C": "C",
        "Â°C": "C",
        "Ã¢â€°Â¥": ">=",
        "â‰¥": ">=",
        "Ã¢â‚¬": '"',
        "â€": '"',
        "™": "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def infer_document_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "companies" in parts or "factsheet" in name:
        return "factsheet"
    if "reports" in parts:
        return "report"
    if "news" in parts:
        return "news"
    if "financial" in name:
        return "financials"
    if "funding" in name:
        return "funding"
    return "unknown"


def infer_company(text: str, filename: str = "") -> str | None:
    haystack = f"{filename}\n{text}".lower()
    for company in COMPANY_NAMES:
        if company.lower() in haystack:
            return company
    slug_map = {re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_"): c for c in COMPANY_NAMES}
    for slug, company in slug_map.items():
        if slug in filename.lower():
            return company
    return None


def infer_date(text: str) -> str | None:
    patterns = [
        r"\b(20\d{2}-\d{2}-\d{2})\b",
        r"\b(\d{1,2} [A-Z][a-z]+ 20\d{2})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December) 20\d{2})\b",
        r"\b(Q[1-4] 20\d{2})\b",
        r"\b(FY20\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", normalise_text(text)).strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def load_txt_chunks(path: Path) -> list[DocumentChunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    doc_type = infer_document_type(path)
    company = infer_company(raw, path.name)
    date = infer_date(raw)
    chunks = []
    for idx, text in enumerate(chunk_text(raw)):
        chunks.append(
            DocumentChunk(
                text=text,
                metadata={
                    "source": path.name,
                    "path": str(path),
                    "document_type": doc_type,
                    "company": infer_company(text, path.name) or company,
                    "date": infer_date(text) or date,
                    "chunk_id": idx,
                },
            )
        )
    return chunks


def load_csv_chunks(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc_type = infer_document_type(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            company = row.get("company_name") or infer_company(" ".join(row.values()), path.name)
            pieces = [f"{key}: {value}" for key, value in row.items() if value not in (None, "")]
            text = normalise_text(f"{doc_type.title()} record. " + "; ".join(pieces))
            chunks.append(
                DocumentChunk(
                    text=text,
                    metadata={
                        "source": path.name,
                        "path": str(path),
                        "document_type": doc_type,
                        "company": company,
                        "date": row.get("round_date") or row.get("fy_year"),
                        "chunk_id": idx,
                    },
                )
            )
    return chunks


def load_corpus(dataset_dir: str | Path) -> list[DocumentChunk]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. Expected the assignment data under data/dataset/."
        )
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a folder: {dataset_path}")

    chunks: list[DocumentChunk] = []
    for path in sorted(dataset_path.rglob("*")):
        if path.suffix.lower() == ".txt":
            chunks.extend(load_txt_chunks(path))
        elif path.suffix.lower() == ".csv":
            chunks.extend(load_csv_chunks(path))
    if not chunks:
        raise ValueError(f"No TXT or CSV documents were loaded from {dataset_path}.")
    return chunks
