"""Main RAG pipeline built as a LangGraph StateGraph."""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.generator import generate_answer
from src.agents.grader import grade_chunks
from src.agents.hallucination_checker import check_hallucination
from src.agents.web_search import search_web
from src.config import GRADE_THRESHOLD, MAX_RETRIES, QUERY_EXPANSION_ENABLED
from src.utils import StageTimer, get_correlation_id, get_llm

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    expanded_queries: list[str]
    chunks: list[dict]
    graded_chunks: list[dict]
    filtered_chunks: list[dict]
    answer: str
    confidence_score: float
    hallucination_detected: bool
    hallucination_issues: str
    used_web_search: bool
    retry_count: int
    sources: list[dict]
    citations: list[dict]
    conversation_history: str
    session_id: str


# Node functions


def expand_query_node(state: RAGState) -> RAGState:
    """Expand the question into multiple search variants."""
    if not QUERY_EXPANSION_ENABLED:
        return {**state, "expanded_queries": [state["question"]]}

    from src.agents.query_expander import expand_query

    llm = get_llm()
    expanded = expand_query(state["question"], llm)
    return {**state, "expanded_queries": expanded}


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve top-k chunks using hybrid search, merging results from expanded queries."""
    from src.retrieval.retriever import retrieve

    cid = get_correlation_id()
    queries = state.get("expanded_queries", [state["question"]])

    if len(queries) == 1:
        chunks = retrieve(queries[0])
    else:
        # Multi-query retrieval with deduplication
        seen_texts: set[str] = set()
        merged: list[dict] = []
        for q in queries:
            for chunk in retrieve(q):
                text_key = chunk["text"].strip()[:200]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    merged.append(chunk)
        # Sort by hybrid_score and take top results
        merged.sort(key=lambda c: c.get("hybrid_score", c.get("score", 0)), reverse=True)
        chunks = merged

    logger.info("[%s] Retrieved %d chunks for %d query variants", cid, len(chunks), len(queries))
    return {**state, "chunks": chunks, "used_web_search": False, "retry_count": 0}


def grade_node(state: RAGState) -> RAGState:
    """Grade each chunk for relevance."""
    llm = get_llm()
    graded = grade_chunks(state["question"], state["chunks"], llm)
    filtered = [c for c in graded if c["grade"] >= GRADE_THRESHOLD]
    avg_score = sum(c["grade"] for c in graded) / len(graded) if graded else 0.0
    cid = get_correlation_id()
    logger.info(
        "[%s] Grading: %d/%d above threshold (avg=%.2f)",
        cid,
        len(filtered),
        len(graded),
        avg_score,
    )
    return {
        **state,
        "graded_chunks": graded,
        "filtered_chunks": filtered,
        "confidence_score": avg_score,
    }


def web_search_node(state: RAGState) -> RAGState:
    """Fetch web results and re-grade them."""
    cid = get_correlation_id()
    logger.info("[%s] All chunk grades below threshold — falling back to web search.", cid)
    web_results = search_web(state["question"])
    if not web_results:
        return {**state, "used_web_search": True}

    llm = get_llm()
    graded_web = grade_chunks(state["question"], web_results, llm)
    filtered = [c for c in graded_web if c["grade"] >= GRADE_THRESHOLD]
    all_graded = state.get("graded_chunks", []) + graded_web
    avg_score = sum(c["grade"] for c in all_graded) / len(all_graded) if all_graded else 0.0
    return {
        **state,
        "graded_chunks": all_graded,
        "filtered_chunks": filtered,
        "confidence_score": avg_score,
        "used_web_search": True,
    }


def generate_node(state: RAGState) -> RAGState:
    """Generate answer from filtered chunks with inline citations."""
    llm = get_llm()
    chunks_to_use = state.get("filtered_chunks") or state.get("graded_chunks", [])
    conversation_history = state.get("conversation_history", "")

    answer, citations = generate_answer(state["question"], chunks_to_use, llm, conversation_history)

    sources = [
        {"text": c["text"], "score": c.get("grade", c.get("hybrid_score", 0.0))}
        for c in chunks_to_use
    ]
    return {**state, "answer": answer, "sources": sources, "citations": citations}


def hallucination_node(state: RAGState) -> RAGState:
    """Check the generated answer for hallucinations."""
    llm = get_llm()
    chunks_to_use = state.get("filtered_chunks") or state.get("graded_chunks", [])
    result = check_hallucination(state["answer"], chunks_to_use, llm)
    return {
        **state,
        "hallucination_detected": bool(result["hallucination"]),
        "hallucination_issues": str(result.get("issues", "")),
        "confidence_score": float(result["confidence"]),
    }


def regenerate_node(state: RAGState) -> RAGState:
    """Increment retry count and regenerate."""
    new_state: RAGState = {**state, "retry_count": state.get("retry_count", 0) + 1}
    return generate_node(new_state)


# Routing logic


def route_after_grade(state: RAGState) -> str:
    """Route to web search if no chunks pass the threshold, else generate."""
    if state.get("filtered_chunks"):
        return "generate"
    return "web_search"


def route_after_hallucination(state: RAGState) -> str:
    """Retry generation if hallucination detected and retries remain."""
    retry_count = state.get("retry_count", 0)
    if state.get("hallucination_detected") and retry_count < MAX_RETRIES:
        return "regenerate"
    return "done"


# Graph construction


def _build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("expand_query", expand_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("hallucination_check", hallucination_node)
    graph.add_node("regenerate", regenerate_node)

    graph.set_entry_point("expand_query")
    graph.add_edge("expand_query", "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "web_search": "web_search"},
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {"regenerate": "regenerate", "done": END},
    )
    graph.add_edge("regenerate", "hallucination_check")

    return graph


compiled_graph = _build_graph().compile()


def run_pipeline(
    question: str,
    session_id: str = "",
    conversation_history: str = "",
) -> dict:
    """Execute the full RAG pipeline and return the final result dict."""
    cid = get_correlation_id()
    logger.info("[%s] Pipeline started for: %s", cid, question[:80])

    with StageTimer("full_pipeline"):
        initial_state: RAGState = {
            "question": question,
            "expanded_queries": [],
            "chunks": [],
            "graded_chunks": [],
            "filtered_chunks": [],
            "answer": "",
            "confidence_score": 0.0,
            "hallucination_detected": False,
            "hallucination_issues": "",
            "used_web_search": False,
            "retry_count": 0,
            "sources": [],
            "citations": [],
            "conversation_history": conversation_history,
            "session_id": session_id,
        }
        final_state = compiled_graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "hallucination_detected": final_state.get("hallucination_detected", False),
        "sources": final_state.get("sources", []),
        "citations": final_state.get("citations", []),
        "used_web_search": final_state.get("used_web_search", False),
        "retry_count": final_state.get("retry_count", 0),
    }
