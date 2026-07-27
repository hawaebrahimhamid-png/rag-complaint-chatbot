from src.rag.pipeline import ask

question = (
    "Why are customers complaining "
    "about credit card payments?"
)

answer = ask(question)

print(answer)
