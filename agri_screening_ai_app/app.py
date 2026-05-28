from __future__ import annotations

import pandas as pd
import streamlit as st

from src.bootstrap import get_system
from src.rag import answer_question


DATASET_DIR = "data/dataset"
VIEWS = ["Company Scoring Dashboard", "Analyst Chat", "Monitoring Alerts", "Company Notes"]
SCORE_COLUMNS = [2.4, 1, 1.8, 1.2, 1, 1]
ALERT_COLUMNS = [
    "company_name",
    "alert_type",
    "trigger_value",
    "source_reference",
    "evidence",
    "recommended_action",
]

st.set_page_config(page_title="Agri Screening AI", layout="wide")


@st.cache_resource(show_spinner="Building local retrieval index...")
def load_system():
    return get_system(DATASET_DIR)


def badge(flag: str) -> str:
    colors = {"PRIORITY": "#0f766e", "WATCH": "#b45309", "LOW": "#991b1b"}
    return (
        f"<span style='background:{colors.get(flag, '#334155')};color:white;"
        "padding:4px 8px;border-radius:6px;font-size:0.8rem;font-weight:700;'>"
        f"{flag}</span>"
    )


def alert_counts_by_company(alerts: list[dict]) -> dict[str, int]:
    if not alerts:
        return {}
    return pd.DataFrame(alerts).groupby("company_name").size().to_dict()


def show_scores(scores: list[dict], alerts: list[dict]) -> None:
    st.subheader("Ranked Company Scores")
    with st.expander("Scoring formula"):
        st.write(
            "Total score is the deterministic sum of four 25-point dimensions: financial, technology, market, and ESG. "
            "Financial scoring uses CSV metrics for revenue CAGR, gross margin, runway, and burn efficiency. "
            "Technology, market, and ESG scores use factsheet evidence."
        )
    alert_counts = alert_counts_by_company(alerts)
    for row in scores:
        with st.container(border=True):
            cols = st.columns(SCORE_COLUMNS)
            cols[0].markdown(f"**{row['company_name']}**")
            cols[1].write(row["country"])
            cols[2].write(row["sub_sector"])
            cols[3].metric("Score", f"{row['total_score']:.1f}")
            cols[4].markdown(badge(row["score_flag"]), unsafe_allow_html=True)
            cols[5].metric("Alerts", alert_counts.get(row["company_name"], 0))
            st.progress(int(row["total_score"]))
            st.caption(
                f"Financial {row['financial_score']}/25 | Technology {row['technology_score']}/25 | "
                f"Market {row['market_score']}/25 | ESG {row['esg_score']}/25"
            )


def show_chat(index) -> None:
    st.subheader("Analyst Question Answering")
    default = "Which companies have an active fundraising process? What are the expected amounts?"
    question = st.text_input("Ask a question about the corpus", value=default)
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question before asking.")
            st.stop()
        response = answer_question(question, index)
        st.markdown("### Answer")
        if response.get("used_llm"):
            st.success(f"LLM-generated answer using OpenAI ({response.get('model')})")
        else:
            st.info("Local extractive fallback")
        st.write(response["answer"])
        st.markdown("### Sources")
        if response["sources"]:
            for source in response["sources"]:
                with st.expander(f"{source['filename']} - {source['document_type']} - score {source['score']}"):
                    st.caption(f"Company: {source.get('company') or 'N/A'} | Date: {source.get('date') or 'N/A'}")
                    st.write(source["excerpt"])
        else:
            st.info("No source chunks met the relevance threshold.")


def show_alerts(alerts: list[dict]) -> None:
    st.subheader("Active Monitoring Alerts")
    if alerts:
        df = pd.DataFrame(alerts)
        st.dataframe(
            df[ALERT_COLUMNS],
            use_container_width=True,
            hide_index=True,
        )
        for alert in alerts:
            with st.container(border=True):
                st.markdown(f"**{alert['alert_type']}** - {alert['company_name']}")
                st.write(alert["trigger_value"])
                st.caption(alert["source_reference"])
                st.write(alert["evidence"])
                st.info(alert["recommended_action"])
    else:
        st.success("No active alerts.")


def show_notes(scores: list[dict], notes: dict[str, str]) -> None:
    st.subheader("Structured Company Notes")
    company = st.selectbox("Select company", [row["company_name"] for row in scores])
    st.markdown(notes[company])


try:
    system = load_system()
except Exception as exc:
    st.error(f"Could not load the screening dataset: {exc}")
    st.stop()

scores = system["scores"]
alerts = system["alerts"]
notes = system["notes"]
index = system["index"]

st.title("AI-Assisted Company Screening")
st.caption(f"Retrieval backend: {index.backend}. OpenAI answers are used when configured; local fallback remains available.")

view = st.sidebar.radio("View", VIEWS)

if view == "Company Scoring Dashboard":
    show_scores(scores, alerts)
elif view == "Analyst Chat":
    show_chat(index)
elif view == "Monitoring Alerts":
    show_alerts(alerts)
else:
    show_notes(scores, notes)
