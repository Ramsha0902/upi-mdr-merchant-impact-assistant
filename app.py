import pickle
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from sentence_transformers import CrossEncoder, SentenceTransformer


# ------------------------------------------------------------
# Page setup and styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="UPI MDR Merchant Impact Assistant",
    page_icon="₹",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f8fafc;
        }

        .hero {
            background: linear-gradient(135deg, #0f766e, #0f4c5c);
            color: white;
            padding: 2rem 2.2rem;
            border-radius: 18px;
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
        }

        .hero p {
            margin: 0.6rem 0 0 0;
            font-size: 1.05rem;
            opacity: 0.92;
        }

        .section-heading {
            color: #0f4c5c;
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.6rem;
        }

        .source-card {
            background: white;
            border-left: 4px solid #14b8a6;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.55rem;
            color: #334155;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 0.8rem;
        }

        div.stButton > button {
            background-color: #0f766e;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
        }

        div.stButton > button:hover {
            background-color: #115e59;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>💳 UPI MDR Merchant Impact Assistant</h1>
        <p>
            Ask questions about the selected official UPI MDR documents
            or calculate MDR for a regular merchant transaction.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "⚠️ Educational prototype only. This is not legal, tax, financial, "
    "or compliance advice. Review the official source before making a decision."
)

st.sidebar.header("ℹ️ About this demo")

st.sidebar.write(
    "This assistant searches selected official UPI MDR documents, "
    "shows the supporting source pages, and includes a regular-P2M MDR calculator."
)

st.sidebar.markdown(
    """
**Official sources**

- [Department of Financial Services FAQ](https://financialservices.gov.in/sites/default/files/2026-09/FAQs---Merchant-Discount-Rate--MDR--on-Select-UPI--P2M--Transactions_0.pdf)
- [Ministry of Finance / PIB release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2310586&lang=1&reg=48)
"""
)


# ------------------------------------------------------------
# App settings
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
GEMINI_MODEL = "gemini-2.5-flash"


# ------------------------------------------------------------
# Load data and models
# ------------------------------------------------------------
@st.cache_resource
def load_search_assets():
    chunks_df = pd.read_csv(DATA_DIR / "upi_mdr_chunks.csv")

    embeddings = np.load(
        DATA_DIR / "upi_mdr_dense_embeddings.npy"
    )

    with open(DATA_DIR / "upi_mdr_bm25.pkl", "rb") as file:
        bm25_index = pickle.load(file)

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return chunks_df, embeddings, bm25_index, dense_model, reranker


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)

    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (
        (scores - scores.min())
        / (scores.max() - scores.min())
    )


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def readable_document_name(document_name):
    if document_name.startswith("FAQs---"):
        return "Department of Financial Services FAQ"

    if document_name.startswith("Press Release"):
        return "Ministry of Finance / PIB release"

    return document_name


# ------------------------------------------------------------
# Retrieve official source evidence
# ------------------------------------------------------------
def retrieve_evidence_chunks(question, primary_k=3):
    (
        chunks_df,
        embeddings,
        bm25_index,
        dense_model,
        reranker
    ) = load_search_assets()

    question_embedding = dense_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    dense_scores = embeddings @ question_embedding

    bm25_scores = bm25_index.get_scores(
        tokenize(question)
    )

    hybrid_scores = (
        0.55 * normalize_scores(dense_scores)
        + 0.45 * normalize_scores(bm25_scores)
    )

    candidate_k = min(24, len(chunks_df))

    candidate_indices = np.argsort(
        hybrid_scores
    )[::-1][:candidate_k]

    candidate_pairs = [
        (question, chunks_df.iloc[index]["chunk_text"])
        for index in candidate_indices
    ]

    reranker_scores = reranker.predict(
        candidate_pairs
    )

    reranked_indices = candidate_indices[
        np.argsort(reranker_scores)[::-1]
    ][:primary_k]

    selected_indices = []
    retrieval_roles = {}

    for index in reranked_indices:
        if index not in selected_indices:
            selected_indices.append(index)
            retrieval_roles[index] = "primary_result"

        row = chunks_df.iloc[index]

        same_page_neighbours = chunks_df[
            (chunks_df["document_name"] == row["document_name"])
            & (chunks_df["page_number"] == row["page_number"])
            & (
                chunks_df["chunk_number"].isin(
                    [
                        row["chunk_number"] - 1,
                        row["chunk_number"] + 1
                    ]
                )
            )
        ]

        for neighbour_index in same_page_neighbours.index:
            if neighbour_index not in selected_indices:
                selected_indices.append(neighbour_index)
                retrieval_roles[
                    neighbour_index
                ] = "adjacent_context"

    evidence_df = chunks_df.loc[selected_indices].copy()

    evidence_df["retrieval_role"] = [
        retrieval_roles[index]
        for index in evidence_df.index
    ]

    evidence_df = evidence_df.reset_index(drop=True)

    evidence_df["source_label"] = [
        f"Source {number}"
        for number in range(1, len(evidence_df) + 1)
    ]

    return evidence_df


def format_evidence_for_prompt(evidence_df):
    source_blocks = []

    for _, row in evidence_df.iterrows():
        source_blocks.append(
            f"""[{row["source_label"]}]
Document: {readable_document_name(row["document_name"])}
Page: {row["page_number"]}
Passage:
{row["chunk_text"]}"""
        )

    return "\n\n".join(source_blocks)


# ------------------------------------------------------------
# Generate answer only from retrieved source passages
# ------------------------------------------------------------
def generate_answer(question, evidence_df, client):
    source_passages = format_evidence_for_prompt(
        evidence_df
    )

    prompt = f"""
You are a source-grounded educational assistant for the UPI MDR framework.

Answer the user's question using ONLY the official source passages below.
Do not use outside knowledge.
Do not invent rules, exceptions, amounts, dates, examples, or citations.
Use precise paraphrases of the source text.

If the passages do not sufficiently support an answer, reply exactly:

I could not find a sufficiently supported answer in the selected official UPI MDR documents.

Write in clear, plain language.

Citation rules:
- Cite every important factual statement.
- Use one source label per pair of brackets.
- Correct: [Source 1] [Source 2]
- Incorrect: [Source 1, Source 2].
- Use only labels that appear in the source passages below.

End with this sentence:

Educational information only; review the official source before making a payment or compliance decision.

USER QUESTION:
{question}

OFFICIAL SOURCE PASSAGES:
{source_passages}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return (response.text or "").strip()


# ------------------------------------------------------------
# Deterministic regular-P2M MDR calculator
# ------------------------------------------------------------
def calculate_regular_p2m_mdr(transaction_amount):
    amount = Decimal(str(transaction_amount))

    if amount <= 0:
        raise ValueError(
            "Transaction amount must be greater than ₹0."
        )

    if amount <= Decimal("2000"):
        mdr = Decimal("0.00")
        rule_applied = "No MDR applies up to ₹2,000."

    else:
        mdr = amount * Decimal("0.004")

        if amount >= Decimal("75000"):
            mdr = min(mdr, Decimal("300"))
            rule_applied = (
                "0.4% MDR applies, capped at ₹300."
            )
        else:
            rule_applied = "0.4% MDR applies."

    return {
        "transaction_amount": amount,
        "customer_charge": Decimal("0.00"),
        "merchant_mdr": mdr.quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP
        ),
        "rule_applied": rule_applied
    }


# ------------------------------------------------------------
# Question-answering interface
# ------------------------------------------------------------
st.markdown(
    '<div class="section-heading">💬 Ask a question</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Ask about the selected official UPI MDR documents",
    placeholder=(
        "Example: Do UPI customers have to pay MDR charges?"
    )
)

if st.button("🔎 Find answer"):
    if not question.strip():
        st.info("Please enter a question first.")

    elif "GEMINI_API_KEY" not in st.secrets:
        st.error(
            "The app owner has not configured the Gemini API secret yet."
        )

    else:
        try:
            with st.spinner(
                "Searching official documents and preparing an answer..."
            ):
                client = genai.Client(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                evidence_df = retrieve_evidence_chunks(
                    question
                )

                answer = generate_answer(
                    question,
                    evidence_df,
                    client
                )

                st.session_state["qa_result"] = {
                    "answer": answer,
                    "evidence": evidence_df
                }

        except Exception:
            st.error(
                "The answer could not be generated. "
                "Please try again shortly."
            )


if "qa_result" in st.session_state:
    answer = st.session_state["qa_result"]["answer"]
    evidence_df = st.session_state["qa_result"]["evidence"]

    st.markdown(
        '<div class="section-heading">✨ Answer</div>',
        unsafe_allow_html=True
    )

    st.markdown(answer)

    st.markdown(
        '<div class="section-heading">📚 Sources used</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Source labels in the answer, such as [Source 1], "
        "refer to the official document and page listed below."
    )

    for _, row in evidence_df.iterrows():
        document = readable_document_name(
            row["document_name"]
        )

        st.markdown(
            f"""
            <div class="source-card">
                <strong>{row["source_label"]}</strong><br>
                {document} · <strong>Page {row["page_number"]}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("📖 Read the retrieved official passages"):
        for _, row in evidence_df.iterrows():
            document = readable_document_name(
                row["document_name"]
            )

            st.markdown(
                f"**{row['source_label']} — {document}, "
                f"page {row['page_number']}**"
            )

            st.write(row["chunk_text"])


# ------------------------------------------------------------
# MDR calculator interface
# ------------------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-heading">🧮 Regular merchant MDR calculator</div>',
    unsafe_allow_html=True
)

st.caption(
    "For regular UPI person-to-merchant transactions only. "
    "The customer charge remains ₹0."
)

amount = st.number_input(
    "Enter transaction amount in ₹",
    min_value=0.0,
    step=1.0,
    format="%.2f"
)

if amount > 0:
    result = calculate_regular_p2m_mdr(amount)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transaction amount",
        f"₹{result['transaction_amount']:,.2f}"
    )

    col2.metric(
        "Customer charge",
        f"₹{result['customer_charge']:,.2f}"
    )

    col3.metric(
        "Merchant MDR",
        f"₹{result['merchant_mdr']:,.2f}"
    )

    st.info(
        f"✅ Rule applied: {result['rule_applied']}"
    )
