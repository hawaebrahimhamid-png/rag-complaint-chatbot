import faiss
import numpy as np
import os


def create_vector_store(
    embeddings,
    chunks_df,
    output_path="../vector_store"
):

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
