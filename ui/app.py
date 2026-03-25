"""Streamlit UI - chat interface with streaming support and analytics page."""

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="TruthRAG", page_icon="\u2705", layout="wide")

# session state init

if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []


def _get_or_create_session() -> str:
    """Ensure we have a conversation session."""
    if not st.session_state.session_id:
        try:
            resp = httpx.post(f"{API_URL}/session", timeout=10)
            resp.raise_for_status()
            st.session_state.session_id = resp.json()["session_id"]
        except Exception:
            st.session_state.session_id = "local"
    return st.session_state.session_id


# Sidebar nav + file upload

page = st.sidebar.radio("Navigation", ["Chat", "Analytics", "RAGAS Evaluation"], index=0)

with st.sidebar:
    st.divider()
    st.header("Document Ingestion")
    uploaded = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        with st.spinner("Ingesting..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = httpx.post(f"{API_URL}/ingest", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['chunks_added']} chunks from {uploaded.name}")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    if st.button("New Conversation"):
        st.session_state.session_id = ""
        st.session_state.messages = []
        st.rerun()

    st.divider()
    use_streaming = st.toggle("Stream responses", value=True)

# --- Chat page ---

if page == "Chat":
    st.title("TruthRAG \u2014 Self-Correcting RAG")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                # Badges
                cols = st.columns(4)
                with cols[0]:
                    score = meta.get("confidence_score", 0)
                    color = "red" if score < 0.4 else ("orange" if score < 0.7 else "green")
                    st.markdown(f":{color}[Confidence: {score:.2f}]")
                with cols[1]:
                    if meta.get("used_web_search"):
                        st.markdown(":blue[Web Search Used]")
                with cols[2]:
                    if meta.get("hallucination_detected"):
                        st.markdown(":red[Hallucination Detected]")
                with cols[3]:
                    if meta.get("retry_count", 0) > 0:
                        st.markdown(f":orange[Retries: {meta['retry_count']}]")

                # Citations
                citations = meta.get("citations", [])
                if citations:
                    with st.expander(f"Sources ({len(citations)})", expanded=False):
                        for c in citations:
                            label = f"**[Source {c['source_number']}]** {c['filename']}"
                            if c.get("page_number"):
                                label += f", page {c['page_number']}"
                            label += f" (score: {c['score']:.2f})"
                            st.markdown(label)
                            st.text(c["text_preview"])
                            st.divider()

    # Chat input
    if question := st.chat_input("Ask a question..."):
        session_id = _get_or_create_session()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get response
        with st.chat_message("assistant"):
            if use_streaming:
                # Streaming response
                try:
                    full_answer = ""
                    metadata = {}
                    placeholder = st.empty()

                    with httpx.stream(
                        "POST",
                        f"{API_URL}/query/stream",
                        json={"question": question, "session_id": session_id},
                        timeout=300,
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = json.loads(line[6:])

                            if data["type"] == "metadata":
                                metadata = data
                            elif data["type"] == "token":
                                full_answer += data["content"]
                                placeholder.markdown(full_answer + "\u258c")
                            elif data["type"] == "done":
                                metadata["latency_ms"] = data.get("latency_ms", 0)
                            elif data["type"] == "error":
                                st.error(f"Error: {data['message']}")

                    placeholder.markdown(full_answer)

                    msg_metadata = {
                        "confidence_score": 0,
                        "citations": metadata.get("citations", []),
                        "used_web_search": metadata.get("used_web_search", False),
                        "hallucination_detected": False,
                        "retry_count": 0,
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "metadata": msg_metadata,
                    })

                except Exception as exc:
                    st.error(f"Query failed: {exc}")
            else:
                # Non-streaming response
                with st.spinner("Running pipeline..."):
                    try:
                        resp = httpx.post(
                            f"{API_URL}/query",
                            json={"question": question, "session_id": session_id},
                            timeout=300,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")
                        st.stop()

                st.markdown(result["answer"])

                msg_metadata = {
                    "confidence_score": result.get("confidence_score", 0),
                    "citations": result.get("citations", []),
                    "used_web_search": result.get("used_web_search", False),
                    "hallucination_detected": result.get("hallucination_detected", False),
                    "retry_count": result.get("retry_count", 0),
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": msg_metadata,
                })

                # Show badges inline
                cols = st.columns(4)
                with cols[0]:
                    score = result.get("confidence_score", 0)
                    color = "red" if score < 0.4 else ("orange" if score < 0.7 else "green")
                    st.markdown(f":{color}[Confidence: {score:.2f}]")
                with cols[1]:
                    if result.get("used_web_search"):
                        st.markdown(":blue[Web Search Used]")
                with cols[2]:
                    if result.get("hallucination_detected"):
                        st.markdown(":red[Hallucination Detected]")
                with cols[3]:
                    if result.get("retry_count", 0) > 0:
                        st.markdown(f":orange[Retries: {result['retry_count']}]")

                # Citations
                citations = result.get("citations", [])
                if citations:
                    with st.expander(f"Sources ({len(citations)})", expanded=False):
                        for c in citations:
                            label = f"**[Source {c['source_number']}]** {c['filename']}"
                            if c.get("page_number"):
                                label += f", page {c['page_number']}"
                            label += f" (score: {c['score']:.2f})"
                            st.markdown(label)
                            st.text(c["text_preview"])
                            st.divider()

# --- Analytics page ---

elif page == "Analytics":
    st.title("Analytics Dashboard")

    days = st.selectbox("Time range", [7, 14, 30, 90], index=2)

    try:
        summary_resp = httpx.get(f"{API_URL}/analytics/summary", params={"days": days}, timeout=10)
        summary_resp.raise_for_status()
        summary = summary_resp.json()
    except Exception as exc:
        st.error(f"Failed to load analytics: {exc}")
        st.stop()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", summary["total_queries"])
    with col2:
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")
    with col3:
        st.metric("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")
    with col4:
        st.metric("Hallucinations", summary["hallucination_count"])

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Web Searches", summary["web_search_count"])
    with col6:
        st.metric("Avg Retries", f"{summary['avg_retries']:.1f}")
    with col7:
        st.metric("Avg Sources", f"{summary['avg_sources']:.1f}")

    st.divider()

    # Confidence trend chart
    st.subheader("Confidence Trend")
    try:
        trend_resp = httpx.get(
            f"{API_URL}/analytics/trend", params={"days": days}, timeout=10
        )
        trend_resp.raise_for_status()
        trend = trend_resp.json()
        if trend:
            st.line_chart(
                data={
                    "Confidence": [t["avg_confidence"] for t in trend],
                    "Queries": [t["query_count"] for t in trend],
                },
            )
        else:
            st.info("No trend data yet.")
    except Exception as exc:
        st.warning(f"Could not load trend: {exc}")

    st.divider()

    # Recent queries table
    st.subheader("Recent Queries")
    try:
        recent_resp = httpx.get(f"{API_URL}/analytics/recent", params={"limit": 20}, timeout=10)
        recent_resp.raise_for_status()
        recent = recent_resp.json()
        if recent:
            st.dataframe(
                recent,
                column_config={
                    "question": st.column_config.TextColumn("Question", width="large"),
                    "confidence_score": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
                    "hallucination_detected": st.column_config.CheckboxColumn("Hallucination"),
                    "used_web_search": st.column_config.CheckboxColumn("Web Search"),
                    "num_sources": st.column_config.NumberColumn("Sources"),
                },
                use_container_width=True,
            )
        else:
            st.info("No queries recorded yet.")
    except Exception as exc:
        st.warning(f"Could not load recent queries: {exc}")

    st.divider()

    # Document usage
    st.subheader("Document Usage")
    try:
        doc_resp = httpx.get(
            f"{API_URL}/analytics/document-hits", params={"days": days}, timeout=10
        )
        doc_resp.raise_for_status()
        doc_hits = doc_resp.json()
        if doc_hits:
            st.bar_chart(
                data={d["filename"]: d["hit_count"] for d in doc_hits},
            )
        else:
            st.info("No document usage data yet.")
    except Exception as exc:
        st.warning(f"Could not load document hits: {exc}")

    st.divider()

    # Chunk hit rate
    st.subheader("Chunk Hit Rate")
    try:
        chunk_resp = httpx.get(
            f"{API_URL}/analytics/chunk-hits",
            params={"days": days, "limit": 20},
            timeout=10,
        )
        chunk_resp.raise_for_status()
        chunk_hits = chunk_resp.json()
        if chunk_hits:
            st.dataframe(
                chunk_hits,
                column_config={
                    "filename": st.column_config.TextColumn("File", width="medium"),
                    "chunk_index": st.column_config.NumberColumn("Chunk #"),
                    "page_number": st.column_config.NumberColumn("Page"),
                    "hit_count": st.column_config.NumberColumn("Hits"),
                    "avg_score": st.column_config.NumberColumn("Avg Score", format="%.3f"),
                },
                use_container_width=True,
            )
        else:
            st.info("No chunk hit data yet.")
    except Exception as exc:
        st.warning(f"Could not load chunk hits: {exc}")

# --- RAGAS Evaluation page ---

elif page == "RAGAS Evaluation":
    st.title("RAGAS Evaluation Dashboard")

    # Eval metrics summary
    eval_days = st.selectbox("Time range", [7, 14, 30, 90], index=2, key="eval_days")
    try:
        eval_resp = httpx.get(
            f"{API_URL}/analytics/eval-summary", params={"days": eval_days}, timeout=10
        )
        eval_resp.raise_for_status()
        eval_summary = eval_resp.json()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Evaluations", eval_summary["total_evals"])
        with col2:
            st.metric("Avg Faithfulness", f"{eval_summary['avg_faithfulness']:.3f}")
        with col3:
            st.metric("Avg Answer Relevancy", f"{eval_summary['avg_answer_relevancy']:.3f}")
        with col4:
            cr = eval_summary.get("avg_context_recall")
            st.metric("Avg Context Recall", f"{cr:.3f}" if cr else "N/A")
    except Exception as exc:
        st.warning(f"Could not load eval summary: {exc}")

    st.divider()

    # Upload eval dataset
    st.subheader("Run Batch Evaluation")
    st.markdown("Upload a JSON file: `[{\"question\": \"...\", \"ground_truth\": \"...\"}]`")
    eval_file = st.file_uploader("Upload evaluation dataset", type=["json"], key="eval_upload")

    if eval_file is not None:
        if st.button("Run Evaluation"):
            with st.spinner("Running RAGAS evaluation... This may take a while."):
                try:
                    files = {"file": (eval_file.name, eval_file.getvalue(), "application/json")}
                    resp = httpx.post(f"{API_URL}/evaluate", files=files, timeout=600)
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(f"Evaluation complete! Batch ID: {result['batch_id']}, {result['total']} samples")

                    if result.get("results"):
                        st.dataframe(
                            result["results"],
                            column_config={
                                "question": st.column_config.TextColumn("Question", width="large"),
                                "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.3f"),
                                "answer_relevancy": st.column_config.NumberColumn("Answer Relevancy", format="%.3f"),
                                "context_recall": st.column_config.NumberColumn("Context Recall", format="%.3f"),
                            },
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    st.divider()

    # Recent eval results
    st.subheader("Recent Evaluations")
    try:
        recent_resp = httpx.get(f"{API_URL}/analytics/eval-recent", params={"limit": 20}, timeout=10)
        recent_resp.raise_for_status()
        recent_evals = recent_resp.json()
        if recent_evals:
            st.dataframe(
                recent_evals,
                column_config={
                    "question": st.column_config.TextColumn("Question", width="large"),
                    "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.3f"),
                    "answer_relevancy": st.column_config.NumberColumn("Relevancy", format="%.3f"),
                    "context_recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "eval_type": st.column_config.TextColumn("Type"),
                    "batch_id": st.column_config.TextColumn("Batch"),
                },
                use_container_width=True,
            )
        else:
            st.info("No evaluations recorded yet.")
    except Exception as exc:
        st.warning(f"Could not load recent evaluations: {exc}")
