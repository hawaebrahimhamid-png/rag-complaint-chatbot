import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import AppConfig


def create_chunks(
    sample_df: pd.DataFrame
) -> pd.DataFrame:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=AppConfig.CHUNK_SIZE,
        chunk_overlap=AppConfig.CHUNK_OVERLAP
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
