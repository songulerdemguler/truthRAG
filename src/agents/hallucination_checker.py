"""Checks generated answers against source documents for hallucinations."""

import logging

from langchain_ollama import ChatOllama

from src.utils import StageTimer, parse_llm_json

logger = logging.getLogger(__name__)

HALLUCINATION_PROMPT = (
    "You are a hallucination detector. Compare the answer against the source "
    "documents. Determine if the answer contradicts or goes beyond what the "
    "sources state.\n\n"
    "Sources:\n{sources}\n\n"
    "Answer: {answer}\n\n"
    "Reply with ONLY valid JSON (no markdown, no explanation):\n"
    '{{"hallucination": <true or false>, "confidence": <float 0.0-1.0>, '
    '"issues": "<description of issues or empty string>"}}'
)


def check_hallucination(
    answer: str, chunks: list[dict], llm: ChatOllama
) -> dict[str, bool | float | str]:
    """Check if the answer hallucinates beyond the provided sources.

    Returns {"hallucination": bool, "confidence": float, "issues": str}.
    """
    default: dict[str, bool | float | str] = {
        "hallucination": False,
        "confidence": 0.5,
        "issues": "Failed to parse LLM response",
    }

    with StageTimer("hallucination_check"):
        sources = "\n\n---\n\n".join(c["text"] for c in chunks)
        prompt = HALLUCINATION_PROMPT.format(sources=sources, answer=answer)

        try:
            response = llm.invoke(prompt)
            content = str(response.content)
            result = parse_llm_json(content, default=default)
            return {
                "hallucination": bool(result.get("hallucination", False)),
                "confidence": float(result.get("confidence", 0.5)),
                "issues": str(result.get("issues", "")),
            }
        except Exception:
            logger.exception("Hallucination check failed")
            return default
