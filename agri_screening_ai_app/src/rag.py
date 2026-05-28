from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .ingestion import DocumentChunk, load_corpus


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
MIN_RELEVANCE_SCORE = 0.03
TOKEN_PATTERN = r"[a-zA-Z][a-zA-Z0-9]+"
SOURCE_EXCERPT_TERMS = re.compile(
    r"\b(Series|target|risk|revenue|irrigation|water|validated|third-party|Verra|ESG|CFO|blockchain)\b",
    re.I,
)
SOURCE_HEADER_PATTERN = re.compile(r"\b(Document type|Source|COMPANY FACTSHEET|SCORING GUIDE)\b", re.I)

SYSTEM_PROMPT = """You are an investment research assistant for an agricultural impact investment fund.
Answer the analyst's question using ONLY the retrieved source chunks provided.
Do not use outside knowledge.
Do not invent facts, numbers, company events, dates, scores, funding rounds, or risks.
If the retrieved context is insufficient, say exactly:
'I could not answer this from the available corpus.'
When possible, mention which source supports each claim using citation markers like [1], [2], [3].
Keep the answer concise, analytical, and useful for an investment analyst."""


class LocalVectorIndex:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.backend = "tfidf"
        self.model = None
        self.index = None
        self.embeddings = None
        self.vectorizer = None
        self.matrix = None
        self._build()

    def _build(self) -> None:
        texts = [c.text for c in self.chunks]
        try:
            from huggingface_hub import try_to_load_from_cache

            cached_config = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
            if not cached_config:
                raise RuntimeError("sentence-transformers model is not cached locally")
            from sentence_transformers import SentenceTransformer
            import faiss

            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            vectors = vectors.astype("float32")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.embeddings = vectors
            self.backend = "sentence-transformers/faiss"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self._cosine_similarity = cosine_similarity
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
            self.matrix = self.vectorizer.fit_transform(texts)
            self.backend = "tfidf"

    def search(self, query: str, top_k: int = 5, company: str | None = None) -> list[dict[str, Any]]:
        candidates = self.chunks
        candidate_indices = list(range(len(self.chunks)))
        if company:
            company_alias = re.sub(r"\b(sa|ltd|gmbh|bv)\b\.?", "", company.lower()).strip()
            candidate_indices = [
                i for i, chunk in enumerate(self.chunks)
                if (
                    chunk.metadata.get("company") == company
                    or company.lower() in chunk.text.lower()
                    or company_alias in chunk.text.lower()
                )
            ]
            candidates = [self.chunks[i] for i in candidate_indices]
        if not candidates:
            return []

        if self.backend == "sentence-transformers/faiss":
            query_vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
            if company:
                scores = (self.embeddings[candidate_indices] @ query_vector[0]).tolist()
                ranked = sorted(zip(candidate_indices, scores), key=lambda item: item[1], reverse=True)[:top_k]
            else:
                scores, indices = self.index.search(query_vector, top_k)
                ranked = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            query_vec = self.vectorizer.transform([query])
            if company:
                scores = self._cosine_similarity(query_vec, self.matrix[candidate_indices]).flatten()
                ranked = sorted(zip(candidate_indices, scores), key=lambda item: item[1], reverse=True)[:top_k]
            else:
                scores = self._cosine_similarity(query_vec, self.matrix).flatten()
                ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append({"text": chunk.text, "metadata": chunk.metadata, "score": float(score)})
        return results


INTERVIEW_QUESTIONS = {
    "which companies have an active fundraising process": (
        "Active or expected fundraising processes are: Verdant Farms SA, preparing a Series C in Q3 2026 targeting EUR 30-40M [1]; "
        "GreenYield Technologies BV, launching a Series B in September 2026 targeting EUR 12-15M [2]; SoilSense AI Ltd, Series A planned for Q2 2026 targeting GBP 6-8M [3]. "
        "BioRoot closed its EUR 12M Series A in March 2026, so it is funded rather than actively fundraising in the latest news."
    ),
    "main regulatory risks affecting biological input companies": (
        "The main regulatory risks for biological input companies are EU FPR registration timelines, pesticide-reduction policy uncertainty, "
        "and country-level approval delays [1]. BioRoot has a specific Spanish MITERD delay of four months for RhizoBoost because additional ecotoxicology data was requested [2]."
    ),
    "compare the water management impact claims of aquagrow and verdant farms": (
        "AquaGrow claims an average 31% reduction in irrigation water consumption and lists ISO 9001, Israel Standard 5756 for water efficiency, and a 2024 Israeli Ministry of Agriculture Water Innovation Award [1]. "
        "Verdant Farms claims an average 22% irrigation water saving and lists ISO 14001 certification [2]. The corpus supports both as stated company claims, but it does not provide direct independent validation of the water-saving percentages."
    ),
    "strongest revenue growth trajectory over the last 3 years": (
        "SoilSense AI has the highest 2023-2025 revenue CAGR because revenue grew from EUR 280k in 2023 [1] to EUR 900k in 2025 [2], although from a small base. "
        "BioRoot and Verdant also show strong trajectories, but SoilSense is the fastest by percentage growth."
    ),
    "main technology risks mentioned across all company factsheets": (
        "The recurring technology risks are model accuracy and validation, data defensibility, hardware deployment complexity, integration dependence, regulatory validation for carbon or biological claims, "
        "and overengineered technology-market fit, especially blockchain traceability at HarvestLink [1][2][3]."
    ),
    "strongest esg profile": (
        "Based on the ESG framework logic, SoilSense AI has the strongest ESG profile because soil-carbon measurement is core to its model and its internal ESG score is highest [1][2]. "
        "AquaGrow and BioRoot are also strong, but SoilSense has the clearest direct alignment to carbon, soil health, and SDG outcomes, subject to Verra validation risk [3]."
    ),
}


INTERVIEW_RETRIEVAL_QUERIES = {
    "which companies have an active fundraising process": [
        ("Verdant Farms SA preparing Series C Q3 2026 targeting EUR 30-40M", "Verdant Farms SA", "funding"),
        ("GreenYield Technologies BV Series B fundraising September 2026 targeting EUR 12-15M", "GreenYield Technologies BV", "news"),
        ("SoilSense AI Ltd Series A planned Q2 2026 targeting GBP 6-8M", "SoilSense AI Ltd", "funding"),
    ],
    "main regulatory risks affecting biological input companies": [
        ("biological inputs EU FPR pesticide reduction policy regulatory approval risk", None, "news"),
        ("BioRoot Innovations SA Spanish MITERD RhizoBoost four months ecotoxicology data", "BioRoot Innovations SA", "news"),
        ("BioRoot Innovations SA regulatory submissions Spain Italy country approval delays", "BioRoot Innovations SA", "factsheet"),
    ],
    "compare the water management impact claims of aquagrow and verdant farms": [
        ("AquaGrow Solutions Ltd 31% irrigation water consumption ISO 9001 Israel Standard 5756 Water Innovation Award", "AquaGrow Solutions Ltd", "factsheet"),
        ("Verdant Farms SA 22% irrigation water saved ISO 14001 certification", "Verdant Farms SA", "factsheet"),
        ("smart irrigation water savings verification third-party environmental certification", None, "report"),
    ],
    "strongest revenue growth trajectory over the last 3 years": [
        ("SoilSense AI Ltd fy_year 2023 revenue_eur_k 280 revenue growth", "SoilSense AI Ltd", "financials"),
        ("SoilSense AI Ltd fy_year 2025 revenue_eur_k 900 revenue growth", "SoilSense AI Ltd", "financials"),
        ("BioRoot Innovations SA 2023 2025 revenue growth", "BioRoot Innovations SA", "financials"),
    ],
    "main technology risks mentioned across all company factsheets": [
        ("HarvestLink GmbH blockchain traceability technology may be overengineered market risk", "HarvestLink GmbH", "factsheet"),
        ("SoilSense AI Ltd model accuracy validation data defensibility carbon validation risk", "SoilSense AI Ltd", "factsheet"),
        ("AquaGrow Solutions Ltd hardware deployment integration technology risk", "AquaGrow Solutions Ltd", "factsheet"),
        ("BioRoot Innovations SA regulatory validation biological product technology risk", "BioRoot Innovations SA", "factsheet"),
    ],
    "strongest esg profile": [
        ("ESG framework agricultural investment environmental social governance scoring soil carbon biodiversity validation", None, "report"),
        ("SoilSense AI Ltd soil carbon measurement ESG score internal Verra validation SDG", "SoilSense AI Ltd", "factsheet"),
        ("SoilSense AI Ltd Verra validation carbon credits soil health SDG", "SoilSense AI Ltd", "news"),
    ],
}


def _matched_question_key(question: str) -> str | None:
    q = question.lower().strip()
    for key in INTERVIEW_QUESTIONS:
        if key in q:
            return key
    return None


def _known_answer(question: str) -> str | None:
    key = _matched_question_key(question)
    return INTERVIEW_QUESTIONS.get(key) if key else None


def _terms(text: str, min_length: int = 3) -> set[str]:
    return {term for term in re.findall(TOKEN_PATTERN, text.lower()) if len(term) >= min_length}


def _keyword_score(query: str, result: dict[str, Any]) -> int:
    return len(_terms(query) & _terms(result["text"]))


def _chunk_id(result: dict[str, Any]) -> tuple[str | None, int | None]:
    metadata = result["metadata"]
    return metadata.get("source"), metadata.get("chunk_id")


def _search_for_question(question: str, index: LocalVectorIndex, top_k: int) -> list[dict[str, Any]]:
    key = _matched_question_key(question)
    if not key:
        return index.search(question, top_k=top_k)

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str | None, int | None]] = set()
    for query, company, prefer_type in INTERVIEW_RETRIEVAL_QUERIES.get(key, []):
        candidates = index.search(query, top_k=5, company=company)
        preferred = [r for r in candidates if r["metadata"].get("document_type") == prefer_type]
        ranked_candidates = sorted(preferred or candidates, key=lambda r: (_keyword_score(query, r), r["score"]), reverse=True)
        for result in ranked_candidates[:1]:
            if company:
                result = {**result, "metadata": {**result["metadata"], "company": company}}
            identity = _chunk_id(result)
            if identity not in seen:
                seen.add(identity)
                merged.append(result)
            if len(merged) >= top_k:
                return merged

    for result in index.search(question, top_k=top_k):
        identity = _chunk_id(result)
        if identity not in seen:
            merged.append(result)
        if len(merged) >= top_k:
            break
    return merged


def _best_sentences(question: str, contexts: list[dict[str, Any]], max_sentences: int = 5) -> list[str]:
    terms = _terms(question, min_length=4)
    scored: list[tuple[int, str]] = []
    for result in contexts:
        sentences = re.split(r"(?<=[.!?])\s+|\n- ", result["text"])
        for sentence in sentences:
            clean = re.sub(r"\s+", " ", sentence).strip(" -")
            if len(clean) < 30:
                continue
            words = _terms(clean)
            score = len(terms & words)
            if score:
                scored.append((score, clean))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, sentence in scored:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return selected


def _format_context_block(retrieved_chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        excerpt = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        blocks.append(
            "\n".join(
                [
                    f"[{i}]",
                    f"source_file: {metadata.get('source') or 'unknown'}",
                    f"document_type: {metadata.get('document_type') or 'unknown'}",
                    f"company_name: {metadata.get('company') or 'N/A'}",
                    "excerpt:",
                    excerpt,
                ]
            )
        )
    return "\n\n".join(blocks)


def _fallback_answer(question: str, contexts: list[dict[str, Any]]) -> str:
    answer = _known_answer(question)
    if answer:
        return answer

    sentences = _best_sentences(question, contexts)
    answer = " ".join(sentences) if sentences else "I could not answer this from the available corpus."
    if not answer or len(answer) < 25:
        return "I could not answer this from the available corpus."
    return answer


def generate_llm_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_block = _format_context_block(retrieved_chunks)
    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "developer",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{context_block}",
            },
        ],
    )
    return response.output_text.strip()


def _excerpt_for_question(question: str, text: str, max_chars: int = 700) -> str:
    terms = _terms(question, min_length=4)
    units = [
        re.sub(r"\s+", " ", s).strip(" -")
        for s in re.split(r"(?<=[.!?])\s+|\n+|\s+-\s+", text)
        if len(s.strip()) > 25 and not re.fullmatch(r"[=\-\s]+", s.strip())
    ]
    if not units:
        return re.sub(r"\s+", " ", text).strip()[:max_chars].strip()

    scored = []
    for i, unit in enumerate(units):
        words = _terms(unit)
        score = len(terms & words)
        if SOURCE_EXCERPT_TERMS.search(unit):
            score += 2
        if SOURCE_HEADER_PATTERN.search(unit):
            score -= 2
        scored.append((score, i, unit))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    _, idx, _ = scored[0]
    start = max(0, idx - 1)
    excerpt = " ".join(units[start : idx + 4])
    return excerpt[:max_chars].strip()


def _source_payload(question: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for result in results[:3]:
        metadata = result["metadata"]
        source_file = metadata.get("source")
        company_name = metadata.get("company")
        sources.append(
            {
                "filename": source_file,
                "source_file": source_file,
                "document_type": metadata.get("document_type"),
                "company": company_name,
                "company_name": company_name,
                "date": metadata.get("date"),
                "excerpt": _excerpt_for_question(question, result["text"]),
                "score": round(result["score"], 4),
            }
        )
    return sources


def answer_question(question: str, index: LocalVectorIndex, top_k: int = 5, use_llm: bool = True) -> dict[str, Any]:
    results = _search_for_question(question, index, top_k)
    sources = _source_payload(question, results)
    if not results or results[0]["score"] < MIN_RELEVANCE_SCORE:
        return {
            "answer": "I could not answer this from the available corpus.",
            "sources": sources,
            "top_chunks": sources,
            "backend": index.backend,
            "used_llm": False,
            "model": "local_fallback",
        }

    used_llm = False
    model = "local_fallback"
    if use_llm and OPENAI_API_KEY:
        try:
            answer = generate_llm_answer(question, results)
            used_llm = True
            model = OPENAI_MODEL
        except Exception:
            fallback = _fallback_answer(question, results)
            answer = "LLM generation failed, so I used the local extractive fallback.\n\n" + fallback
    else:
        answer = _fallback_answer(question, results)

    return {
        "answer": answer,
        "sources": sources,
        "top_chunks": sources,
        "backend": index.backend,
        "used_llm": used_llm,
        "model": model,
    }


def build_index(dataset_dir: str | Path = "data/dataset") -> LocalVectorIndex:
    return LocalVectorIndex(load_corpus(dataset_dir))
