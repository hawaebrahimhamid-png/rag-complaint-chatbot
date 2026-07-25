# Task 1 completed (EDA + preprocessing finalized)

## Objective

The objective of Task 1 was to explore the CFPB consumer complaint dataset, assess data quality, and prepare a clean dataset suitable for a Retrieval-Augmented Generation (RAG) complaint chatbot.

## Dataset Overview

The CFPB complaint dataset contains consumer complaints submitted across a variety of financial products and services. Initial exploratory analysis was performed to understand the dataset structure, identify missing values, and analyze complaint distribution across product categories.

## Exploratory Data Analysis

The following analyses were performed:

- Examined dataset dimensions and column structure.
- Analyzed complaint distribution across products.
- Identified records with and without consumer complaint narratives.
- Calculated complaint narrative lengths using word counts.
- Visualized narrative length distribution using a histogram.

### Key Findings

- The largest number of complaints belonged to the **Checking or Savings Account** category.
- **Credit Card** complaints represented the second largest category.
- **Money Transfer, Virtual Currency, or Money Service** complaints formed a significant portion of the dataset.
- **Payday Loan, Title Loan, Personal Loan, or Advance Loan** complaints represented the smallest category among the selected products.
- Complaint narratives varied significantly in length, ranging from very short descriptions to detailed narratives exceeding 6,000 words.

## Data Preprocessing

The dataset was prepared for downstream retrieval and embedding generation through the following preprocessing steps:

1. Filtered the dataset to retain only:
   - Credit Card
   - Checking or Savings Account
   - Money Transfer, Virtual Currency, or Money Service
   - Payday Loan, Title Loan, Personal Loan, or Advance Loan

2. Removed records with missing consumer complaint narratives.

3. - Cleaned narrative text by:
   - Converting text to lowercase.
   - Removing common boilerplate phrases.
   - Removing special characters and unnecessary symbols.
   - Normalizing whitespace.

4. Created a cleaned text column (`clean_text`) for future embedding generation.

## Validation Results

After preprocessing, the dataset was validated to ensure it is ready for the RAG pipeline.

Checks performed:

- Confirmed that all retained records contain complaint narratives.
- Verified that cleaned text (`clean_text`) contains no missing values.
- Confirmed that only the required financial product categories remain.
- Reviewed cleaned text samples to ensure preprocessing removed unnecessary noise while preserving customer complaint information.

The final dataset is structured and ready for the next stages:

- Text chunking
- Embedding generation
- Vector database indexing
- Semantic retrieval

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
## Task 2: Chunking, Embedding & Vector Store Creation

### Overview

This task prepares complaint data for a Retrieval-Augmented Generation (RAG) system by transforming cleaned complaint records into searchable vector representations.

The pipeline performs:

1. Data sampling
2. Text chunking
3. Sentence Transformer embedding generation
4. FAISS vector database creation

---

### Sampling Strategy

The original processed dataset contains 80,667 complaint records. Due to local hardware memory limitations, a smaller subset was used for embedding generation and FAISS indexing.

For efficient processing, 10,000 records were loaded from the processed dataset, and a sample of 1,000 complaint records was selected for building the retrieval system.

The sampled dataset was used for:

- Text chunking
- Sentence Transformer embeddings
- FAISS vector indexing

The full processed dataset was retained, and the pipeline can be scaled to larger datasets with additional computational resources.

---

### Embedding Model

The project uses:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384

The model converts complaint text chunks into numerical vectors that capture semantic meaning.

---

### Vector Store

FAISS (Facebook AI Similarity Search) is used to store and retrieve similar complaint embeddings efficiently.

Generated artifacts:

vector_store/
├── faiss.index
├── metadata.csv
└── chunks.csv


---

### Task 2 Pipeline

Filtered Complaints Dataset
↓
Data Sampling
↓
Text Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Index
↓
Semantic Search Retrieval


---

### Output

Task 2 successfully creates a searchable vector database that will be used by the RAG retriever component in the next stage.
