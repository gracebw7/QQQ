import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="QQQ Research",
    layout="centered",
)

# Page header
st.title("Research Assistant")
st.caption("Search and retrieve research papers from arXiv.")

st.divider()

# Search form
with st.form("search_form"):
    query = st.text_input(
        "Search arXiv",
        placeholder="Enter a research topic or question",
    )

    col1, col2 = st.columns([1, 3], vertical_alignment="bottom")

    with col1:
        max_results = st.number_input(
            "Max papers",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

    with col2:
        submitted = st.form_submit_button(
            "Search papers",
            use_container_width=True,
            type="primary",
        )

# Store previous results
if "search_response" not in st.session_state:
    st.session_state.search_response = None

# Search
if submitted:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        st.session_state.search_response = None

        with st.spinner("Searching arXiv and downloading papers..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/api/search",
                    json={
                        "query": query,
                        "max_results": max_results,
                    },
                    timeout=700.0,
                )

                response.raise_for_status()
                st.session_state.search_response = response.json()

            except httpx.HTTPError as error:
                st.error(f"Search failed: {error}")

# Results
data = st.session_state.search_response

if data is not None:
    papers = data["papers"]

    st.subheader(f"Results ({len(papers)})")

    if not papers:
        st.info("No papers found. Try a different search.")

    for paper in papers:
        with st.container(border=True):
            st.markdown(f"#### {paper['title']}")

            st.caption(", ".join(paper["authors"]))

            with st.expander("Read abstract"):
                st.write(paper["abstract"])

            if paper["download_status"] in ("downloaded", "cached"):
                pdf_url = (
                    f"{API_BASE_URL}/api/documents/"
                    f"{paper['document_id']}/pdf"
                )

                st.link_button("View PDF", pdf_url)

            else:
                st.warning("PDF could not be downloaded.")
                st.link_button(
                    "View on arXiv",
                    paper["pdf_url"],
                )
