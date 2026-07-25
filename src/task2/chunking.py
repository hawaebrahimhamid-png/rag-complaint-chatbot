import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(sample_df):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for _, row in sample_df.iterrows():

        texts = splitter.split_text(
            row["clean_text"]
        )

        for text in texts:
            chunks.append(
                {
                    "complaint_id": row["Complaint ID"],
                    "product": row["Product"],
                    "text": text
                }
            )

    chunks_df = pd.DataFrame(chunks)

    return chunks_df
