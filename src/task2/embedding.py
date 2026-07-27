import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import AppConfig


def generate_embeddings(
    chunks_df: pd.DataFrame
) -> np.ndarray:

    model = SentenceTransformer(
        AppConfig.EMBEDDING_MODEL
    )

    texts = chunks_df["text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=AppConfig.BATCH_SIZE,
        show_progress_bar=True
    )

    return embeddings.astype("float32")
