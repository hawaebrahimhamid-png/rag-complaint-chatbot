from pathlib import Path

from src.rag.retriever import Retriever
from src.rag.prompt import PROMPT_TEMPLATE
from src.rag.generator import generate_answer
from src.config import AppConfig

from typing import Tuple
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[2]


retriever = Retriever(
    str(PROJECT_ROOT / "vector_store" / "faiss.index"),
    str(PROJECT_ROOT / "vector_store" / "metadata.csv")
)


def ask(question: str) -> Tuple[str, pd.DataFrame]:

    results = retriever.search(
        question,
        k=AppConfig.TOP_K
    )

    if results.empty:
        return (
            "I do not have enough information from the provided complaint data.",
            results
        )


    context = "\n".join(
        results["text"].tolist()
    )


    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )


    answer = generate_answer(prompt)


    return answer, results

