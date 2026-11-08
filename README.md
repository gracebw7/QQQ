# QQQ

Our model processes a user’s question and uses a database API to retrieve documents and provide
a response grounded in the retrieved documents with citations.
MVP limitations:
Our model handles singular questions asked by the user and requests the user to reformat their
prompt if that format is not used. If a user asks follow-up questions, the documents retrieved from the
previous question are preserved in the vector store, and new documents are also retrieved using the same
cycle. The previous answer, however, is not used as context.
When the user asks a question:

- The model queries the selected database API and reads returned documents into a vector store
- The model queries the vector store to select sentences to form an answer
- The model outputs the answer to the UI where the user can review the answer, which has citations
  for every claim and select the citation to view the sentence in the referenced document PDF
  If the model can’t find sufficient evidence, it responds with a clear abstention statement, i.e.
  “Unable to find an answer in desired database documents.”

# Run App:

```text
Run backend: uv run uvicorn app.api:app --reload
Run frontend: uv run streamlit run main.py
Run tests: uv run pytest
```

# Repo File Structure:

```text
QQQ/
|-- main.py                      # Simple Streamlit UI
|-- app/
|   |-- __init__.py
|   |-- schemas.py               # Search request and paper response models
|   |-- api.py                   # HTTP endpoints
|   |-- search/
|   |   |-- __init__.py
|   |   |-- arxiv_client.py      # arXiv search and result normalization
|   |   |-- prompts/             # Model prompts
|   |-- documents/              # PDF processing
|-- tests/
|   |-- fixtures/
|-- scripts/                    # Utility scripts
|-- docs/
|   |-- sys-arch.pdf
|-- data/                       # Local generated data (ignored by Git)
|   |-- raw/
|   |-- processed/
|-- .gitignore
|-- pyproject.toml
|-- uv.lock
```
