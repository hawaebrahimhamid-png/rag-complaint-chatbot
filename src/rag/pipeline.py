from pathlib import Path

from src.rag.retriever import Retriever
from src.rag.prompt import PROMPT_TEMPLATE
from src.rag.generator import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

retriever = Retriever(
    str(PROJECT_ROOT / "vector_store" / "faiss.index"),
    str(PROJECT_ROOT / "vector_store" / "metadata.csv")
)


def ask(question):

    results = retriever.search(
        question,
        k=3
    )

    context = "\n".join(
        results["text"].tolist()
    )

    # Check if no useful context was found
    if results.empty:
        return (
            "I do not have enough information from the provided complaint data.",
            results
        )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    answer = generate_answer(prompt)

    return answer, results
