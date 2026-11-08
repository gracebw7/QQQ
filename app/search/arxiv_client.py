import arxiv
import re

client = arxiv.Client()

def search_papers(query: str, max_results: int = 5) -> list[dict]:
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    seen_ids = set()
    for result in client.results(search):
        arxiv_id = result.entry_id.rstrip("/").split("/")[-1]
        if arxiv_id in seen_ids:
            continue
        seen_ids.add(arxiv_id)
        document_id = arxiv_id.replace("/", "_")
        papers.append({
            "document_id": document_id,
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
        })
    return papers
