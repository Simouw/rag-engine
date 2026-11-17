from __future__ import annotations

from rag_engine.config import settings

SAMPLE = """# Photosynthesis

Photosynthesis converts light energy into chemical energy in chloroplasts.
It produces oxygen and glucose from carbon dioxide and water.

# Respiration

Cellular respiration releases energy from glucose inside mitochondria.
"""


async def _upload(client, **overrides):
    data = {
        "collection": "biology",
        "doc_type": "notes",
        "tags": "biology,plants",
        "language": "en",
    }
    data.update(overrides)
    return await client.post(
        "/api/documents",
        files={"file": ("notes.txt", SAMPLE.encode(), "text/plain")},
        data=data,
    )


async def test_upload_search_and_ask(client):
    response = await _upload(client)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["document"]["status"] == "indexed"
    assert payload["chunks_indexed"] >= 1

    document_id = payload["document"]["id"]

    search = await client.post(
        "/api/search",
        json={"query": "photosynthesis light energy", "top_k": 5},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results
    assert results[0]["document_id"] == document_id
    assert results[0]["content"]

    ask = await client.post(
        "/api/ask",
        json={"question": "How does photosynthesis work?", "top_k": 5},
    )
    assert ask.status_code == 200, ask.text
    answer = ask.json()
    assert answer["sources"]
    assert answer["model"] == "test-model"
    assert answer["usage"]["total_tokens"] == 18


async def test_search_filters_exclude_other_collections(client):
    await _upload(client)

    response = await client.post(
        "/api/search",
        json={
            "query": "photosynthesis",
            "filters": {"collection": "mathematics"},
        },
    )
    assert response.status_code == 200
    assert response.json()["results"] == []

    tagged = await client.post(
        "/api/search",
        json={"query": "photosynthesis", "filters": {"tags": ["plants"]}},
    )
    assert tagged.status_code == 200
    assert tagged.json()["results"]


async def test_list_chunks_and_delete_document(client):
    payload = (await _upload(client)).json()
    document_id = payload["document"]["id"]

    chunks = await client.get(f"/api/documents/{document_id}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()
    assert chunks.json()[0]["metadata"]["chunk_role"] == "child"

    assert (await client.delete(f"/api/documents/{document_id}")).status_code == 204
    assert (await client.get(f"/api/documents/{document_id}")).status_code == 404


async def test_stats_and_health(client):
    await _upload(client)

    health = await client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    stats = await client.get("/api/stats")
    assert stats.status_code == 200
    assert stats.json()["documents"] == 1
    assert stats.json()["collections"] == ["biology"]


async def test_api_key_protection(client, monkeypatch):
    monkeypatch.setattr(settings, "RAG_API_KEY", "secret-key")

    unauthorized = await client.get("/api/stats")
    assert unauthorized.status_code == 401

    authorized = await client.get("/api/stats", headers={"X-API-Key": "secret-key"})
    assert authorized.status_code == 200

    bearer = await client.get("/api/stats", headers={"Authorization": "Bearer secret-key"})
    assert bearer.status_code == 200

    # Health stays public for probes.
    assert (await client.get("/api/health")).status_code == 200


async def test_unsupported_extension_is_rejected(client):
    response = await client.post(
        "/api/documents",
        files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
        data={"collection": "misc"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ingestion_failed"


async def test_failed_ingestion_leaves_no_document_or_file(client, tmp_path):
    from pathlib import Path

    from rag_engine.config import settings

    upload_dir = Path(settings.UPLOAD_DIR)
    before = set(upload_dir.glob("*"))

    response = await client.post(
        "/api/documents",
        files={"file": ("broken.pdf", b"this is not a pdf", "application/pdf")},
        data={"collection": "broken"},
    )
    assert response.status_code == 422
    assert set(upload_dir.glob("*")) == before
    assert (await client.get("/api/documents?collection=broken")).json() == []


async def test_disable_chunk_and_reindex(client):
    payload = (await _upload(client)).json()
    document_id = payload["document"]["id"]

    chunks = (await client.get(f"/api/documents/{document_id}/chunks")).json()
    target = next(chunk for chunk in chunks if "chloroplast" in chunk["content"])

    patched = await client.patch(
        f"/api/documents/{document_id}/chunks/{target['id']}",
        json={"is_enabled": False},
    )
    assert patched.status_code == 200
    assert patched.json()["is_enabled"] is False

    search = await client.post("/api/search", json={"query": "chloroplasts"})
    assert all(result["chunk_id"] != target["id"] for result in search.json()["results"])

    reindexed = await client.post(f"/api/documents/{document_id}/reindex")
    assert reindexed.status_code == 200
    assert reindexed.json()["document"]["status"] == "indexed"
