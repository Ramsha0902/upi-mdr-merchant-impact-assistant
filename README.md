# UPI MDR Merchant Impact Assistant

A source-grounded RAG chatbot and MDR calculator built from selected official Government of India documents on the UPI Merchant Discount Rate (MDR) framework.

## What it does

- Answers questions about the selected UPI MDR documents in plain language.
- Retrieves relevant passages using semantic search, BM25 keyword search, and reranking.
- Shows the source document, page, chunk, and retrieved passage for every answer.
- Includes a deterministic calculator for regular UPI person-to-merchant (P2M) MDR.

## MDR calculator scope

The calculator covers the verified regular-P2M rule in the selected documents:

- Up to ₹2,000: ₹0 MDR
- Above ₹2,000: 0.4% MDR
- ₹75,000 and above: MDR capped at ₹300
- Customer charge: ₹0

It does not calculate special-category or future small-merchant arrangements.

## How the answer system works

1. Official PDF documents are extracted into small text passages.
2. The app searches passages by meaning and exact keywords.
3. A reranker orders the strongest passages for the question.
4. Gemini writes an answer using only those retrieved passages.
5. The answer displays its sources for review.

## Tech stack

- Python
- Streamlit
- Sentence Transformers
- BM25
- Google Gemini API
- Pandas and NumPy

## Official source documents

- [Department of Financial Services FAQ](https://financialservices.gov.in/sites/default/files/2026-09/FAQs---Merchant-Discount-Rate--MDR--on-Select-UPI--P2M--Transactions_0.pdf)
- [Ministry of Finance / Press Information Bureau release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2310586&lang=1&reg=48)

## Important note

This is an educational prototype, not legal, tax, financial, or compliance advice. Review the official source documents before making decisions.
