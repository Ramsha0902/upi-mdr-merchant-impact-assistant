
import pickle
import re
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from sentence_transformers import CrossEncoder, SentenceTransformer


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="UPI MDR Merchant Impact Assistant",
    page_icon="₹",
    layout="wide"
)

st.title("UPI MDR Merchant Impact Assistant")
st.caption(
    "An educational, source-grounded assistant built from official "
    "Department of Financial Services and Ministry of Finance documents."
)

st.warning(
    "Educational prototype only. This is not legal, tax, or financial advice. "
    "Review the official source before making a payment or compliance decision."
)

st.sidebar.header("About this demo")
st.sidebar.write(
    "Ask questions about the selected UPI MDR framework and calculate "
    "merchant-side MDR for regular P2M transactions."
)

st.sidebar.markdown(
    """
**Official sources**

- [Department of Financial Services FAQ](https://financialservices.gov.in/sites/default/files/2026-09/FAQs---Merchant-Discount-Rate--MDR--on-Select-UPI--P2M--Transactions_0.pdf)
- [Ministry of Finance / PIB release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2310586&lang=1&reg=48)
"""
)


# ------------------------------------------------------------
# Load data and AI models once
# ------------------------------------------------------------
DATA_DIR = "data"
GEMINI_MODEL = "gemini-2.5-flash"


@st.cache_resource
def load_search_assets():
    chunks_df = pd.read_csv(f"{DATA_DIR}/upi_mdr_chunks.csv")
    embeddings = np.load(f"{DATA_DIR}/upi_mdr_dense_embeddings.npy")

    with open(f"{DATA_DIR}/upi_mdr_bm25.pkl", "rb") as file:
        bm25_index = pickle.load(file)

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return chunks_df, embeddings, bm25_index, dense_model, reranker


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)

    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve_evidence_chunks(question, primary_k=3):
    chunks_df, embeddings, bm25_index, dense_model, reranker = load_search_assets()

    question_embedding = dense_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    dense_scores = embeddings @ question_embedding
    bm25_scores = bm25_index.get_scores(tokenize(question))

    hybrid_scores = (
        0.55 * normalize_scores(dense_scores)
        + 0.45 * normalize_scores(bm25_scores)
    )

    candidate_k = min(24, len(chunks_df))
    candidate_indices = np.argsort(hybrid_scores)[::-1][:candidate_k]

    candidate_pairs = [
        (question, chunks_df.iloc[index]["chunk_text"])
        for index in candidate_indices
    ]

    reranker_scores = reranker.predict(candidate_pairs)

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

        same_page = chunks_df[
            (chunks_df["document_name"] == row["document_name"])
            & (chunks_df["page_number"] == row["page_number"])
            & (
                chunks_df["chunk_number"].isin(
                    [row["chunk_number"] - 1, row["chunk_number"] + 1]
                )
            )
        ]

        for neighbour_index in same_page.index:
            if neighbour_index not in selected_indices:
                selected_indices.append(neighbour_index)
                retrieval_roles[neighbour_index] = "adjacent_context"

    evidence_df = chunks_df.loc[selected_indices].copy()
    evidence_df["retrieval_role"] = [
        retrieval_roles[index]
        for index in evidence_df.index
    ]

    evidence_df = evidence_df.reset_index(drop=True)

    evidence_df["source_label"] = [
        f"S{number}"
        for number in range(1, len(evidence_df) + 1)
    ]

    return evidence_df


def format_evidence_for_prompt(evidence_df):
    source_blocks = []

    for _, row in evidence_df.iterrows():
        source_blocks.append(
            f"""[{row["source_label"]}]
Document: {row["document_name"]}
Page: {row["page_number"]}
Chunk: {row["chunk_number"]}
Role: {row["retrieval_role"]}
Passage:
{row["chunk_text"]}"""
        )

    return "\n\n".join(source_blocks)


def generate_answer(question, evidence_df, client):
    source_passages = format_evidence_for_prompt(evidence_df)

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
- Correct: [S1] [S2]
- Incorrect: [S1, S2], [S1/S2], or [S1 and S2].
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


def calculate_regular_p2m_mdr(transaction_amount):
    amount = Decimal(str(transaction_amount))

    if amount <= 0:
        raise ValueError("Transaction amount must be greater than ₹0.")

    if amount <= Decimal("2000"):
        mdr = Decimal("0.00")
        rule_applied = "No MDR applies up to ₹2,000."

    else:
        mdr = amount * Decimal("0.004")

        if amount >= Decimal("75000"):
            mdr = min(mdr, Decimal("300"))
            rule_applied = "0.4% MDR applies, capped at ₹300."
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
# Question-answering section
# ------------------------------------------------------------
st.subheader("Ask a question")

question = st.text_area(
    "Ask about the selected UPI MDR documents",
    placeholder=(
        "Example: Do UPI customers have to pay MDR charges?"
    )
)

if st.button("Ask"):
    if not question.strip():
        st.info("Please enter a question first.")

    elif "GEMINI_API_KEY" not in st.secrets:
        st.error(
            "The app owner has not configured the Gemini API secret yet."
        )

    else:
        try:
            with st.spinner("Searching official documents and preparing an answer..."):
                client = genai.Client(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                evidence_df = retrieve_evidence_chunks(question)
                answer = generate_answer(
                    question,
                    evidence_df,
                    client
                )

            st.subheader("Answer")
            st.markdown(answer)

            st.subheader("Sources used")
            st.dataframe(
                evidence_df[
                    [
                        "source_label",
                        "document_name",
                        "page_number",
                        "chunk_number",
                        "retrieval_role"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

            with st.expander("View retrieved official passages"):
                for _, row in evidence_df.iterrows():
                    st.markdown(
                        f"**[{row['source_label']}] "
                        f"{row['document_name']} — "
                        f"page {row['page_number']}, "
                        f"chunk {row['chunk_number']}**"
                    )
                    st.write(row["chunk_text"])

        except Exception:
            st.error(
                "The answer could not be generated. "
                "Please try again shortly."
            )


# ------------------------------------------------------------
# Deterministic MDR calculator
# ------------------------------------------------------------
st.divider()
st.subheader("Regular merchant MDR calculator")
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

    st.info(f"Rule applied: {result['rule_applied']}")
