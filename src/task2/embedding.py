from sentence_transformers import SentenceTransformer


def generate_embeddings(chunks_df):

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = chunks_df["text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=2,
        show_progress_bar=True
    )

    return embeddings.astype("float32")
