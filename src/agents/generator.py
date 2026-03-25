"""Generates answers with inline [Source N] citations."""

import logging

from langchain_ollama import ChatOllama

from src.utils import StageTimer

logger = logging.getLogger(__name__)

GENERATE_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Be concise and accurate. "
    "IMPORTANT: Cite your sources inline using [Source N] notation, "
    "where N is the source number shown in brackets before each context block. "
    "Every factual claim must have a citation.\n"
    "If the context does not contain enough information, "
    'say "I don\'t have enough information to answer this."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

GENERATE_WITH_HISTORY_PROMPT = (
    "You are a helpful assistant in a multi-turn conversation. "
    "Answer the question using ONLY the provided context. "
    "Be concise and accurate. "
    "IMPORTANT: Cite your sources inline using [Source N] notation, "
    "where N is the source number shown in brackets before each context block. "
    "Every factual claim must have a citation.\n"
    "If the context does not contain enough information, "
    'say "I don\'t have enough information to answer this."\n\n'
    "Previous conversation:\n{history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Format chunks into numbered context string and return citation metadata."""
    citations: list[dict] = []
    parts: list[str] = []

    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        filename = meta.get("filename", "unknown")
        page = meta.get("page_number", 0)
        label = f"[Source {i}]: {filename}"
        if page:
            label += f", page {page}"

        parts.append(f"{label}\n{c['text']}")
        citations.append(
            {
                "source_number": i,
                "filename": filename,
                "page_number": page,
                "chunk_index": meta.get("chunk_index", 0),
                "text_preview": c["text"][:200],
                "score": c.get("grade", c.get("hybrid_score", c.get("score", 0.0))),
            }
        )

    return "\n\n---\n\n".join(parts), citations


def generate_answer(
    question: str,
    chunks: list[dict],
    llm: ChatOllama,
    conversation_history: str = "",
) -> tuple[str, list[dict]]:
    """Generate an answer with inline citations.

    Returns (answer_text, citations_list).
    """
    with StageTimer("generate_answer"):
        context, citations = _format_context(chunks)

        if conversation_history:
            prompt = GENERATE_WITH_HISTORY_PROMPT.format(
                history=conversation_history, context=context, question=question
            )
        else:
            prompt = GENERATE_PROMPT.format(context=context, question=question)

        try:
            response = llm.invoke(prompt)
            answer = str(response.content).strip()
            logger.info("Generated answer (len=%d, citations=%d)", len(answer), len(citations))
            return answer, citations
        except Exception:
            logger.exception("Generation failed")
            return "An error occurred while generating the answer.", []
