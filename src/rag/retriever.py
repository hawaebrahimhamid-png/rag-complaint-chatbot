import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        index_path,
        metadata_path
    ):

        self.index = faiss.read_index(
            index_path
        )

        self.metadata = pd.read_csv(
            metadata_path
        )

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )


    def search(
        self,
        question,
        k=5
    ):

        question_embedding = self.model.encode(
            [question]
        )


        distances, indices = self.index.search(
            question_embedding,
            k
        )


        results = self.metadata.iloc[
            indices[0]
        ]


        return results
