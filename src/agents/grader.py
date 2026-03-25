"""Scores each retrieved chunk for relevance (0.0 to 1.0)."""

import logging

from langchain_ollama import ChatOllama

from src.utils import StageTimer, parse_llm_json

logger = logging.getLogger(__name__)

GRADE_PROMPT = """You are a relevance grader. Given a user question and a document chunk,
score how relevant the chunk is to answering the question.

Question: {question}
Chunk: {chunk}

Reply with ONLY valid JSON (no markdown, no explanation):
{{"score": <float 0.0-1.0>, "reason": "<brief reason>"}}"""


def grade_chunk(question: str, chunk: str, llm: ChatOllama) -> dict[str, object]:
    """Score a single chunk's relevance. Returns {"score": float, "reason": str}."""
    prompt = GRADE_PROMPT.format(question=question, chunk=chunk)
    default = {"score": 0.0, "reason": "Failed to parse LLM response"}

    try:
        response = llm.invoke(prompt)
        content = str(response.content)
        result = parse_llm_json(content, default=default)
        return {
            "score": float(result.get("score", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    except Exception:
        logger.exception("Grading call failed")
        return default


def grade_chunks(question: str, chunks: list[dict], llm: ChatOllama) -> list[dict]:
    """Grade a list of chunks, attaching a grade to each one."""
    with StageTimer("grade_chunks"):
        graded: list[dict] = []
        for chunk in chunks:
            grade = grade_chunk(question, chunk["text"], llm)
            graded.append({**chunk, "grade": grade["score"], "grade_reason": grade["reason"]})
        return graded
