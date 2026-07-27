import os

import faiss
import numpy as np
import pandas as pd

from src.config import AppConfig


def create_vector_store(
    embeddings: np.ndarray,
    chunks_df: pd.DataFrame,
    output_path: str = AppConfig.VECTOR_STORE_PATH
) -> faiss.IndexFlatL2:

    os.makedirs(
        output_path,
        exist_ok=True
    )

    embedding_array = np.array(
        embeddings
    ).astype("float32")

    dimension = embedding_array.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embedding_array
    )

    faiss.write_index(
        index,
        f"{output_path}/faiss.index"
    )

    chunks_df.to_csv(
        f"{output_path}/metadata.csv",
        index=False
    )

    return index
