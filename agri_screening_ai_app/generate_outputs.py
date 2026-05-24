from __future__ import annotations

import csv
from pathlib import Path

from src.bootstrap import get_system
from src.rag import answer_question


SAMPLE_QUESTIONS = [
    "Which companies have an active fundraising process? What are the expected amounts?",
    "What are the main regulatory risks affecting biological input companies in this portfolio?",
    "Compare the water management impact claims of AquaGrow and Verdant Farms. Are they verified?",
    "Which company has the strongest revenue growth trajectory over the last 3 years?",
    "What are the main technology risks mentioned across all company factsheets?",
    "Based on the ESG framework document, which company has the strongest ESG profile, and why?",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(company: str) -> str:
    return company.lower().replace(" ", "_").replace(".", "").replace("-", "_") + "_note.md"


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    system = get_system("data/dataset")

    write_csv(output_dir / "company_scores.csv", system["scores"])
    write_csv(output_dir / "monitoring_alerts.csv", system["alerts"])

    lines = ["# Sample RAG Answers", ""]
    for question in SAMPLE_QUESTIONS:
        response = answer_question(question, system["index"])
        lines.extend([f"## {question}", "", response["answer"], "", "Sources:"])
        for source in response["sources"]:
            lines.append(f"- {source['filename']} ({source['document_type']}): {source['excerpt'][:240]}")
        lines.append("")
    (output_dir / "sample_rag_answers.md").write_text("\n".join(lines), encoding="utf-8")

    for company, note in system["notes"].items():
        (output_dir / safe_filename(company)).write_text(note, encoding="utf-8")

    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

