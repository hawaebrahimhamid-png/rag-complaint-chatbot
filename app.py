import streamlit as st

from src.rag.pipeline import ask

st.set_page_config(
    page_title="CrediTrust RAG",
    page_icon="💳",
    layout="wide"
)

st.title("💳 CrediTrust Complaint Assistant")

st.markdown(
    "Ask questions about customer complaints."
)

question = st.text_input(
    "Enter your question"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching complaints..."):

            answer, results = ask(question)

        st.subheader("Answer")
        st.success(answer)

        st.subheader("Retrieved Sources")

        for i, row in results.iterrows():

            with st.expander(f"Source {i+1}"):

                st.write(row["text"])

    else:

        st.warning("Please enter a question.")

if st.button("Clear"):

    st.rerun()
