"""Expands a user query into multiple alternative phrasings for better retrieval."""

import logging

from src.utils import StageTimer, parse_llm_json

logger = logging.getLogger(__name__)

EXPAND_PROMPT = """You are a search query optimizer. Given the user's question, generate {count} alternative phrasings that capture the same intent but use different words or perspectives.

Original question: {question}

Reply with ONLY valid JSON (no markdown, no explanation):
{{"queries": ["alternative 1", "alternative 2"]}}"""


def expand_query(question: str, llm, count: int = 2) -> list[str]:
    """Generate alternative query phrasings using the LLM.

    Returns a list starting with the original question, followed by alternatives.
    On failure, returns only the original question.
    """
    with StageTimer("query_expansion"):
        try:
            prompt = EXPAND_PROMPT.format(question=question, count=count)
            response = llm.invoke(prompt)
            content = str(response.content)
            default = {"queries": []}
            result = parse_llm_json(content, default=default)
            alternatives = result.get("queries", [])

            if not isinstance(alternatives, list) or not alternatives:
                logger.warning("Query expansion returned no alternatives.")
                return [question]

            # Deduplicate and limit
            expanded = [question]
            for alt in alternatives[:count]:
                alt = str(alt).strip()
                if alt and alt.lower() != question.lower() and alt not in expanded:
                    expanded.append(alt)

            logger.info("Expanded query into %d variants: %s", len(expanded), expanded)
            return expanded

        except Exception:
            logger.exception("Query expansion failed, using original query only.")
            return [question]
