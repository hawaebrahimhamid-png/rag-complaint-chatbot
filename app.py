import streamlit as st
import pandas as pd

from src.rag.pipeline import ask
from src.config import AppConfig


# Page configuration
st.set_page_config(
    page_title="CrediTrust RAG Assistant",
    page_icon="💳",
    layout="wide"
)


# Title
st.title("💳 CrediTrust Complaint Assistant")

st.write(
    """
    Ask questions about customer complaints.
    The assistant retrieves relevant complaint records
    and generates answers using the RAG pipeline.
    """
)


# 👇 ADD THE DASHBOARD METRICS HERE
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

metadata = pd.read_csv(
    PROJECT_ROOT / "vector_store" / "metadata.csv"
)

product_counts = metadata["product"].value_counts()

total_chunks = len(metadata)
unique_products = metadata["product"].nunique()

st.subheader("📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Complaint Chunks", total_chunks)

with col2:
    st.metric("Products", unique_products)

with col3:
    st.metric("Embedding Model", "all-MiniLM-L6-v2")

with col4:
    st.metric("Top-K Retrieval", AppConfig.TOP_K)

st.subheader("📈 Complaint Chunks by Product")

st.bar_chart(product_counts)


st.markdown("### 💡 Business Insight")

st.info(
    """
    Checking or Savings Account complaints make up the largest portion of the
    knowledge base. This suggests the RAG system has the most information
    available for this product category, which may lead to more comprehensive
    answers for related user questions.
    """
)


# 👇 THEN KEEP YOUR QUESTION INPUT
question = st.text_input(
    "Enter your question:",
    placeholder="Example: Why are customers complaining about credit cards?"
)


# Ask button
if st.button("🔍 Ask"):

    if question.strip():

        with st.spinner("Searching complaint database..."):

            answer, results = ask(question)


        # Answer section
        st.subheader("🤖 Generated Answer")

        st.success(answer)


        # Sources section
        st.subheader("📚 Retrieved Sources")


        if len(results) > 0:

            for i, row in results.iterrows():

                with st.expander(
                    f"Source {i+1} - Complaint ID: {row.get('complaint_id','N/A')}"
                ):

                    st.write(
                        row["text"]
                    )

        else:

            st.info("No relevant sources found.")

    else:

        st.warning(
            "Please enter a question before clicking Ask."
        )


# Clear button
if st.button("Clear"):

    st.rerun()
