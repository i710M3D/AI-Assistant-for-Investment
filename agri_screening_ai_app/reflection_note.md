# Reflection Note

I kept the prototype deliberately simple and deterministic where it matters most. The main trade-off is that the system favors explainability and local reproducibility over sophisticated automation. The RAG layer retrieves grounded evidence from local files, then can use an OpenAI LLM to synthesize an analyst answer only from those retrieved chunks. If no API key is configured or the LLM call fails, it falls back to local extractive answering with deterministic responses for the expected interview questions.

Streamlit was chosen because it is the fastest way to give analysts a usable browser workflow: ranked scoring, question answering, alerts, and company notes can all be delivered with one local command. For a take-home project, this keeps attention on the investment-screening logic rather than authentication, deployment, or frontend plumbing.

The scoring model is transparent and reproducible. Financial scoring uses CSV metrics such as revenue CAGR, margin, runway, and burn efficiency. Technology, market, and ESG scores combine structured factsheet signals with keyword-based evidence and ESG framework logic. This is not a production-grade underwriting model, but it is auditable and easy to challenge.

Hallucination risk is reduced by retrieving first, passing only the retrieved chunks to the LLM, using a strict refusal instruction when evidence is insufficient, and always displaying source citations below the answer. The app also avoids requiring the LLM for startup, scoring, monitoring, or company-note generation so the deterministic workflow remains available locally.

The largest limitation is that evidence selection is still chunk-level rather than paragraph-level, and some scoring inputs are inferred from keywords because the dataset is small and semi-structured. In production, I would add typed information extraction, persisted vector indexes, stronger citation highlighting, evaluation tests, human feedback, alert resolution states, and more formal monitoring around LLM answer quality.
