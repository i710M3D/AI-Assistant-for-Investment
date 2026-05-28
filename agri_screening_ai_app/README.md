# Agri-Tech Company Screening Prototype

Local Streamlit app for screening agri-tech companies. It loads synthetic factsheets, sector reports, news digests, financial CSVs, and funding CSVs, then builds a retrieval index used by chat, scoring, monitoring alerts, and company notes.

## Architecture

- `src/ingestion.py`: loads TXT and CSV sources, chunks long text with overlap, turns CSV rows into text chunks, and attaches metadata including filename, document type, company, date, and chunk id.
- `src/rag.py`: builds a local retrieval index. It uses `sentence-transformers` plus FAISS when available, and falls back to scikit-learn TF-IDF when those packages are unavailable.
- `src/scoring.py`: computes deterministic 0-100 company scores across Financial, Technology, Market, and ESG dimensions.
- `src/monitoring.py`: applies alert rules and retrieves supporting text from the RAG corpus.
- `src/bootstrap.py`: assembles the index, scores, alerts, and generated company notes.
- `app.py`: Streamlit web interface with four analyst views.
- `generate_outputs.py`: exports scores, alerts, sample RAG answers, and Markdown company notes.

## Installation

```bash
cd agri_screening_ai_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`sentence-transformers` and FAISS are optional in practice. If they are unavailable, the app automatically uses TF-IDF retrieval from scikit-learn.

## Run The App

```bash
streamlit run app.py
```

## OpenAI Answers

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Then install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The chat retrieves local source chunks first, then uses the OpenAI Responses API when `OPENAI_API_KEY` is set. Without a key, or if the API call fails, it uses the local extractive fallback and still shows sources.

## Generate Outputs

```bash
python generate_outputs.py
```

This writes:

- `outputs/company_scores.csv`
- `outputs/monitoring_alerts.csv`
- `outputs/sample_rag_answers.md`
- one Markdown note per company

## Scoring Formula

Each company receives a 0-100 composite score made of four 25-point dimensions.

Financial score:

- 35% revenue CAGR from 2023 to 2025
- 25% latest gross margin
- 25% latest runway months
- 15% burn efficiency, measured as annual revenue divided by annual burn

Technology score:

- patents or proprietary IP
- differentiation claims
- data moat signals
- integrations
- accuracy, field validation, or performance claims

Market score:

- TAM and market tailwinds
- latest revenue scale
- geographic footprint
- competitive position
- partnerships or channel signals

ESG score:

- starts from the factsheet ESG input as a proxy for environmental, social, and governance evidence
- adjusts for verification, certification, and governance weaknesses using retrieved ESG framework, factsheet, and news evidence

The final flag is derived only from the computed score:

- `PRIORITY`: score >= 70
- `WATCH`: score 50-69
- `LOW`: score < 50

## Alert Rules

- `RUNWAY_CRITICAL`: runway months < 12
- `REVENUE_DECLINE`: latest revenue growth < 0
- `ESG_ALERT`: ESG dimension score < 15 out of 25, equivalent to <60/100
- `GOVERNANCE_FLAG`: CFO/CEO departure, vacancy, or key person risk found in text evidence
- `FUNDRAISE_ACTIVE`: active or planned Series A/B/C process detected in news/funding evidence
- `STRATEGIC_EXIT`: M&A, acquisition talks, or strategic sale process detected in news
- `SCORE_PRIORITY`: composite score >= 70

Each alert includes the company, alert type, trigger value, source reference, evidence excerpt, and recommended action.

## Design Choices

The app keeps retrieval local and constrains generated answers to retrieved chunks. The common interview questions have deterministic fallback answers so demos remain repeatable when the LLM is unavailable. Scoring uses financial CSV values plus factsheet evidence rather than copied analyst flags.

Streamlit was chosen because it supports a complete local analyst workflow with minimal app boilerplate: dashboards, chat-style Q&A, tables, and Markdown notes can all run from one command.

## Known Limitations

- LLM answer quality depends on retrieved source chunks; weak retrieval returns an explicit insufficient-corpus response.
- Some scoring features are keyword-derived because the dataset is small and semi-structured.
- Evidence extraction returns the most relevant local chunks, not paragraph-level citations.
- The app uses a rebuilt in-memory index on startup instead of a persisted vector store.

## Future Improvements

- Persist the vector index and add incremental ingestion.
- Replace keyword scoring with typed extraction and validation tests.
- Add analyst feedback loops for score overrides and alert resolution.
- Add richer source highlighting and paragraph-level citation anchors.
