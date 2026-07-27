from transformers import pipeline


generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def generate_answer(prompt):

    response = generator(
        prompt,
        max_length=100,
        min_length=20,
        do_sample=False
    )

    return response[0]["generated_text"]
