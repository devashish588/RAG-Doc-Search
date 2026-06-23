import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


st.set_page_config(page_title="RAG Document Search", layout="wide")
st.title("RAG Document Search")


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def get_documents() -> list[dict[str, Any]]:
    try:
        response = requests.get(api_url("/documents"), timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def upload_file(file) -> dict[str, Any]:
    files = {
        "file": (
            file.name,
            file.getvalue(),
            file.type or "application/octet-stream",
        )
    }
    response = requests.post(api_url("/upload"), files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def search_documents(query: str, top_k: int, source: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if source:
        payload["source"] = source
    response = requests.post(api_url("/search"), json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.caption(f"API: {API_BASE_URL}")
    uploaded = st.file_uploader("Upload document", type=["pdf", "txt", "md"])
    if st.button("Upload", type="primary", use_container_width=True, disabled=uploaded is None):
        try:
            result = upload_file(uploaded)
            st.success(f"{result['filename']} queued")
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(detail)

    if st.button("Refresh documents", use_container_width=True):
        st.rerun()

documents = get_documents()
completed_sources = sorted({doc["filename"] for doc in documents if doc.get("status") == "complete"})

left, right = st.columns([0.75, 0.25], gap="large")

with left:
    with st.form("search_form"):
        query = st.text_input("Ask", placeholder="How does AI learn from examples?")
        top_k = st.slider("Results", min_value=1, max_value=10, value=5)
        source_choice = st.selectbox("Source", ["All documents", *completed_sources])
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a question.")
        else:
            source = None if source_choice == "All documents" else source_choice
            try:
                response = search_documents(query=query, top_k=top_k, source=source)
                st.subheader("Answer")
                st.write(response["answer"])
                st.caption(f"Retrieved in {response['latency_ms']} ms")

                st.subheader("Sources")
                for index, result in enumerate(response["results"], start=1):
                    page = f" | page {result['page']}" if result.get("page") else ""
                    title = f"{index}. {result['source']}{page} | score {result['score']:.2f}"
                    with st.expander(title, expanded=index == 1):
                        st.write(result["text"])
            except requests.RequestException as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(detail)

with right:
    st.subheader("Documents")
    if not documents:
        st.info("No uploaded documents yet.")
    for doc in documents:
        status = doc.get("status", "unknown")
        chunks = doc.get("chunks_indexed", 0)
        label = f"{doc['filename']} - {status}"
        if status == "complete":
            st.success(f"{label} ({chunks} chunks)")
        elif status == "failed":
            st.error(f"{label}: {doc.get('error')}")
        else:
            st.info(label)

