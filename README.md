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


# Task 3: Building the RAG Core Logic and Evaluation

## Objective

Build a Retrieval-Augmented Generation (RAG) pipeline that answers customer complaint questions by retrieving relevant complaint excerpts from a FAISS vector store and generating responses using a language model.

## Workflow

1. Load the pre-built FAISS vector store and metadata.
2. Convert the user's question into an embedding using **all-MiniLM-L6-v2**.
3. Retrieve the most relevant complaint chunks using similarity search.
4. Create a prompt by combining the retrieved context and the user's question.
5. Generate an answer using **FLAN-T5 Small**.
6. Evaluate the RAG pipeline using representative customer complaint questions.

## Components

- **Retriever:** Uses FAISS and all-MiniLM-L6-v2 to retrieve relevant complaint chunks.
- **Prompt Template:** Instructs the language model to answer using only the retrieved context.
- **Generator:** Uses FLAN-T5 Small to generate responses.
- **Pipeline:** Combines retrieval, prompt construction, and answer generation into a single workflow.

## Evaluation

The RAG pipeline was evaluated using five representative questions covering:

- Credit cards
- Money transfers
- Checking accounts
- Personal loans
- Debt collection

For each question, the following were assessed:

- Generated answer
- Retrieved complaint sources
- Quality score (1–5)
- Comments and analysis

## Results

The retriever consistently returned relevant complaint excerpts. However, the FLAN-T5 Small model often generated short or incomplete responses, particularly when prompts approached the model's maximum input length.

## Project Files

```
src/rag/
├── retriever.py
├── generator.py
├── prompt.py
└── pipeline.py

notebooks/
└── task3_evaluation.ipynb

reports/
└── task3_evaluation.md
```

## Future Improvements

- Use a larger instruction-tuned language model (e.g., FLAN-T5 Base or Mistral).
- Improve prompt engineering for more detailed summaries.
- Tune retrieval parameters and chunk size.
- Add a reranking step to improve retrieval relevance.
