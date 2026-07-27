from src.rag.retriever import Retriever


retriever = Retriever(
    "vector_store/faiss.index",
    "vector_store/metadata.csv"
)


results = retriever.search(
    "Why was my credit card payment declined?"
)


print(results)
