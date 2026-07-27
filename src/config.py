from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATOR_MODEL = "google/flan-t5-small"

    TOP_K = 3

    MAX_NEW_TOKENS = 100
    MIN_NEW_TOKENS = 20
