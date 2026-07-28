from src.config import AppConfig


def test_embedding_model_exists():
    assert AppConfig.EMBEDDING_MODEL is not None


def test_generator_model_exists():
    assert AppConfig.GENERATOR_MODEL is not None


def test_top_k_is_positive():
    assert AppConfig.TOP_K > 0
