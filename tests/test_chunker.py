from rag_engine.documents.chunker import Chunker


def test_splits_long_text_without_exceeding_chunk_size():
    text = " ".join(f"Sentence number {index} about a topic." for index in range(200))
    chunks = Chunker(chunk_size=200, overlap=20).split(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_short_text_is_returned_as_one_chunk():
    assert Chunker(chunk_size=500, overlap=50).split("Hello world.") == ["Hello world."]


def test_empty_text_returns_no_chunk():
    assert Chunker().split("   ") == []


def test_invalid_settings_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        Chunker(chunk_size=100, overlap=100)
