import uuid

from rag_engine.retrieval.bm25 import run_bm25
from rag_engine.retrieval.pipeline import RetrievalPipeline


def test_bm25_ranks_the_relevant_document_first():
    raw_data = [
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "title": "Plants",
            "content": "Photosynthesis converts light energy into chemical energy.",
            "metadata_json": {},
            "parent_chunk_id": None,
            "collection": "bio",
            "doc_type": "document",
            "tags": None,
            "language": "en",
            "source_name": "plants.txt",
        },
        {
            "chunk_id": uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "title": "History",
            "content": "The industrial revolution changed European cities.",
            "metadata_json": {},
            "parent_chunk_id": None,
            "collection": "history",
            "doc_type": "document",
            "tags": None,
            "language": "en",
            "source_name": "history.txt",
        },
    ]

    results = run_bm25(raw_data, "photosynthesis light", top_k=5)
    assert results
    assert results[0]["title"] == "Plants"


def test_rrf_puts_documents_found_by_both_modes_first():
    dense = [
        {"chunk_id": "a", "score": 0.9},
        {"chunk_id": "b", "score": 0.8},
    ]
    sparse = [
        {"chunk_id": "b", "score": 12.0},
        {"chunk_id": "c", "score": 8.0},
    ]

    fused = RetrievalPipeline.reciprocal_rank_fusion(dense, sparse, limit=3)
    assert fused[0]["chunk_id"] == "b"
    assert {item["chunk_id"] for item in fused} == {"a", "b", "c"}
