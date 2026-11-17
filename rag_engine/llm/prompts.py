"""Prompts used by the engine. All prompts are English by default."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer the user's question using the provided sources.

Rules:
- Ground every factual claim in the sources. Cite sources inline with [1], [2], ...
- If the sources do not contain the answer, say clearly that the indexed documents do not cover it. Do not invent facts.
- Be concise, structured and specific. Prefer short paragraphs or bullet points.
- Do not add a "Sources" section: the API returns the source list separately.
- If sources disagree, mention the disagreement instead of choosing silently."""

RERANK_SYSTEM_PROMPT = """You are a search relevance judge. You score document excerpts against a query.
Return only a JSON object. Be strict: a high score means the excerpt directly and completely answers the query."""

REWRITE_SYSTEM_PROMPT = """You rewrite a follow-up question into a standalone search query.
Keep the user's language. Return only the rewritten query, without quotes or explanations."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered sources for the LLM."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        page = metadata.get("page_start")
        location = f", page {page}" if page else ""
        blocks.append(
            f"[{index}] {chunk['title']}{location}\n"
            f"{chunk.get('parent_content') or chunk['content']}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        "Sources:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer using only the sources above. Cite them with [n]."
    )
