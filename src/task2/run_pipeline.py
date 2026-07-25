import pandas as pd

from src.task2.sampling import create_sample
from src.task2.chunking import create_chunks
from src.task2.embedding import generate_embeddings
from src.task2.vector_store import create_vector_store


print("Loading data...")

df = pd.read_csv(
    "data/processed/filtered_complaints.csv",
    nrows=10000
)

print("Data loaded:", df.shape)


sample_df = create_sample(
    df,
    1000
)

print("Sample created:", sample_df.shape)


sample_df.to_csv(
    "data/sample_complaints.csv",
    index=False
)


chunks_df = create_chunks(
    sample_df
)

print("Chunks created:", chunks_df.shape)


chunks_df.to_csv(
    "vector_store/chunks.csv",
    index=False
)


print("Starting embeddings...")

print("Starting embedding generation...")

embeddings = generate_embeddings(
    chunks_df
)

print("Embedding finished!")
print("Embedding shape:", embeddings.shape)

print("Embeddings created:", embeddings.shape)


print("Creating vector store...")

create_vector_store(
    embeddings,
    chunks_df
)


print("Task 2 pipeline completed successfully")
