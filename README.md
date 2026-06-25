# rag-complaint-chatbot
# Task 1: Exploratory Data Analysis and Data Preprocessing

## Objective

The objective of Task 1 was to explore the CFPB consumer complaint dataset, assess data quality, and prepare a clean dataset suitable for a Retrieval-Augmented Generation (RAG) complaint chatbot.

## Dataset Overview

The CFPB complaint dataset contains consumer complaints submitted across a variety of financial products and services. Initial exploratory analysis was performed to understand the dataset structure, identify missing values, and analyze complaint distribution across product categories.

## Exploratory Data Analysis

The following analyses were performed:

* Examined dataset dimensions and column structure.
* Analyzed complaint distribution across products.
* Identified records with and without consumer complaint narratives.
* Calculated complaint narrative lengths using word counts.
* Visualized narrative length distribution using a histogram.

### Key Findings

* The largest number of complaints belonged to the **Checking or Savings Account** category.
* **Credit Card** complaints represented the second largest category.
* **Money Transfer, Virtual Currency, or Money Service** complaints formed a significant portion of the dataset.
* **Payday Loan, Title Loan, Personal Loan, or Advance Loan** complaints represented the smallest category among the selected products.
* Complaint narratives varied significantly in length, ranging from very short descriptions to detailed narratives exceeding 6,000 words.

## Data Preprocessing

The dataset was prepared for downstream retrieval and embedding generation through the following preprocessing steps:

1. Filtered the dataset to retain only:

   * Credit Card
   * Checking or Savings Account
   * Money Transfer, Virtual Currency, or Money Service
   * Payday Loan, Title Loan, Personal Loan, or Advance Loan

2. Removed records with missing consumer complaint narratives.

3. Cleaned narrative text by:

   * Converting text to lowercase.
   * Removing special characters.
   * Normalizing whitespace.

4. Created a cleaned text column (`clean_text`) for future embedding generation.

## Output

The cleaned dataset was saved to:

```text
data/filtered_complaints.csv
```

This dataset will be used in subsequent tasks for text chunking, embedding generation, vector storage, and retrieval within the RAG pipeline.

## Files

```text
notebooks/task1_eda.ipynb
data/filtered_complaints.csv
```
# Task 2: Text Chunking, Embedding, and Vector Store Indexing

## Objective

The objective of Task 2 was to transform cleaned complaint narratives into a format suitable for semantic search and Retrieval-Augmented Generation (RAG). This involved sampling complaint records, splitting long narratives into smaller chunks, generating vector embeddings, and storing them in a vector database for efficient retrieval.

---

## Sampling Strategy

A stratified sampling approach was used to preserve the proportional distribution of complaints across product categories. This ensures that all product categories are represented fairly in the sample dataset.

### Product Categories

* Credit Card
* Mortgage
* Debt Collection
* Checking or Savings Account
* Vehicle Loan or Lease

### Sampling Details

* Sampling method: Stratified Sampling
* Sampling fraction: 0.02
* Final sample size: 14,536 complaints

This approach reduced computational requirements while maintaining representative coverage of the dataset.

---

## Text Chunking

Long complaint narratives were divided into smaller text segments before embedding generation.

### Chunking Parameters

* Chunk Size: 500 characters
* Chunk Overlap: 50 characters

### Why Chunking?

Embedding entire complaint narratives can reduce retrieval effectiveness because important information may be diluted across long texts. Chunking helps:

* Preserve semantic meaning
* Improve retrieval precision
* Reduce information loss at chunk boundaries
* Enable more relevant context retrieval during RAG

---

## Embedding Model

The embedding model used was:

`sentence-transformers/all-MiniLM-L6-v2`

### Reasons for Selection

* Lightweight and efficient
* Fast embedding generation
* Produces 384-dimensional embeddings
* Strong semantic similarity performance
* Commonly used baseline model for RAG applications

---

## Vector Store

FAISS (Facebook AI Similarity Search) was used to store and search embeddings efficiently.

### Configuration

* Index Type: `IndexFlatL2`
* Similarity Metric: Euclidean Distance (L2)

### Stored Metadata

For each chunk, the following metadata was retained:

* Complaint ID
* Product Category
* Chunk Index
* Chunk Text

This allows retrieved results to be traced back to their original complaint records.

---

## Output Files

The following files were generated and stored in the `vector_store/` directory:

```text
vector_store/
├── faiss_index.bin
├── chunk_metadata.csv
```

### File Descriptions

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| faiss_index.bin    | Persisted FAISS vector index containing complaint embeddings |
| chunk_metadata.csv | Metadata associated with each embedded text chunk            |

---

## Retrieval Test

A semantic retrieval function was implemented and tested using sample queries.

Example query:

```python
retrieve("credit card billing problem")
```

The retrieval system successfully returned the most relevant complaint chunks based on semantic similarity.

---

## Outcome

Task 2 successfully produced a searchable vector database of complaint narratives. The resulting FAISS index and metadata store will be used in Task 3 to build a Retrieval-Augmented Generation (RAG) pipeline capable of answering user questions about customer complaints.
