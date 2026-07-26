# Task 3: RAG Evaluation

## Overview

This evaluation measures the performance of the Retrieval-Augmented Generation (RAG) pipeline developed for CrediTrust complaint analysis.

The system uses:
- FAISS vector database for similarity search
- all-MiniLM-L6-v2 embeddings for retrieval
- FLAN-T5 Small for answer generation

Retrieval Configuration:
- The retriever uses similarity search with k=3.
- Although the recommended starting point was k=5, k=3 was selected because the available hardware has 8GB RAM and the FLAN-T5 Small model has a 512-token input limitation.
- Using fewer retrieved chunks reduced prompt length and helped avoid token overflow while maintaining relevant retrieval results.

## Evaluation Results

| Question | Generated Answer | Retrieved Sources | Score | Comments |
|---|---|---|---|---|
| What problems do customers report with credit cards? | They don't do anything with credit cards. | 1. Credit card complaint: customer unable to get help from customer service. <br> 2. Credit card complaint: fraud department unable to explain issue. | 3/5 | Retrieval was relevant but answer missed card cancellation, fraud handling, and customer service problems. |
| Why are money transfers failing? | Not assisting with getting money transferred to or from. | 1. Money transfer complaint: customers reported failed or delayed transfers. <br> 2. Customers reported missing transferred funds. | 4/5 | Answer matched the complaint but lacked details about delays and missing funds. |
| What complaints are common about checking accounts? | Several checking accounts were opened fraudulently without consent. | 1. Checking account complaint: accounts opened without customer permission. <br> 2. Complaint related to unauthorized account activity. | 4/5 | Good retrieval and answer, but the response was a single complaint rather than a complete summary. |
| Why are customers unhappy with personal loans? | They are holding funds of customers without explanation. | 1. Personal loan complaint: borrowers reported disputes and difficulty resolving issues. <br> 2. Customers reported collection problems and poor lender communication. | 3/5 | The answer captured customer frustration but missed important loan-related issues. |
| What complaints are common about debt collection? | A complaint. | 1. Debt collection complaint: customers reported repeated calls and harassment. <br> 2. Customers reported aggressive collection practices. | 2/5 | Retrieval was relevant, but the generated answer failed to summarize the retrieved information. |


## Overall Analysis

### Strengths

- FAISS retrieval returned semantically related complaint chunks.
- The embedding model successfully matched user questions with similar complaints.
- The RAG pipeline produced answers grounded in retrieved context.



### Weaknesses

- FLAN-T5 Small generated very short answers.
- Some important complaint details were omitted.
- Longer contexts may exceed the model token limit.
- Some prompts exceeded the FLAN-T5 Small maximum input length (512 tokens), which affected generation quality.

### Possible Improvements

- Use a larger instruction-following model such as FLAN-T5 Base or Mistral.
- Improve prompt instructions to encourage detailed summaries.
- Tune chunk size and retrieval parameters.
- Use reranking to improve retrieved document relevance.


## Conclusion

The RAG pipeline successfully retrieves relevant customer complaint information and generates responses based on retrieved context. The FAISS retriever and MiniLM embeddings provided good semantic search performance, while the FLAN-T5 Small generator produced useful but sometimes incomplete answers.

The evaluation shows that retrieval quality was generally strong, but generation quality was limited by model size and context length constraints. Future improvements should focus on using a stronger instruction-tuned model, improving prompt design, and optimizing retrieval strategies.
