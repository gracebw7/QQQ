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

## Setup and run

Install `uv`, then open a terminal in the `QQQ` project folder.

```powershell
# Install project dependencies (creates .venv automatically)
uv sync

# Start the backend
uv run --with uvicorn uvicorn app.api:app --reload
```

Open a second terminal in the same folder and start the UI:

```powershell
uv run streamlit run main.py
```

Open http://localhost:8501 to use the app. Keep both terminals running.
Searches and PDF downloads require internet access.

```powershell
# Run tests (includes live arXiv requests)
uv run pytest

# Run only offline tests
uv run pytest -m "not integration"
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



# Dataset format

Every line of the `.jsonl` files is one training example. Each example has five parts:

```json
{
  "example_id": "v1_0001",
  "question": "How much does GaLore reduce optimizer state memory?",
  "evidence": [ ... ],
  "target": { ... },
  "metadata": { ... }
}
```

The model sees the **question** and the **evidence** and learns to produce the **target**. The **metadata** is only for us. The model never sees it.

### Top-level fields

| Field | What it means |
|---|---|
| `example_id` | A unique name for the example, like `v1_0001`. |
| `question` | The question a user asked. |
| `evidence` | The sentences the retriever found for this question. There are always 6. |
| `target` | The answer we want the model to give. |
| `metadata` | Notes about how the example was made. Not shown to the model. |

### `evidence`: the sentences the model gets

```json
{
  "source_id": "S6",
  "document_id": "arxiv:2403.03507",
  "sentence_id": "sentence_004",
  "text": "Our approach reduces memory usage by up to 65.5% in optimizer states ..."
}
```

| Field | What it means |
|---|---|
| `source_id` | A short label (`S1` to `S6`) that the answer uses to cite this sentence. `S1` is the closest match to the question and `S6` is the furthest. |
| `document_id` | Which paper the sentence comes from (its arXiv ID). |
| `sentence_id` | Where the sentence is in that paper. `sentence_004` is the paper's 5th sentence, counting from 0. |
| `text` | The sentence itself. |

`document_id` + `sentence_id` together tell the UI exactly where to jump in the PDF when someone clicks a citation.

### `target`: the answer the model should give

| Field | What it means |
|---|---|
| `status` | `"answered"` if the evidence answers the question, or `"abstained"` if it doesn't. |
| `claims` | The answer, broken into short statements. Empty when abstaining. |
| `claims[].text` | One statement. It must only say what the cited sentences say. |
| `claims[].citations` | Which evidence sentences back up the statement, like `["S1"]`. Every claim needs at least one. |
| `abstention_message` | When abstaining, always exactly `"Unable to find an answer in desired database documents."`. When answering, `null`. |

Example of an answered target:

```json
"target": {
  "status": "answered",
  "claims": [
    { "text": "GaLore reduces memory usage in optimizer states by up to 65.5%.", "citations": ["S6", "S3"] }
  ],
  "abstention_message": null
}
```

### `metadata`: notes for us, not the model

| Field | What it means |
|---|---|
| `split` | `train`, `val`, or `test`. Each paper is in only one split. |
| `example_type` | What kind of example this is (see the table below). |
| `gold_source_ids` | The evidence sentences that actually answer the question. Empty for abstain examples. |
| `gold_injected` | Answer sentences the retriever missed, which we added back in by hand. |
| `derived_from` | For an abstain example made from an answerable one, the ID of that original example. |
| `excluded_sentences` | Sentences we deliberately keep **out** of the evidence (explained below). |
| `retriever` | How the evidence sentences were ranked (MiniLM, or TF-IDF as a fallback). |
| `k` | How many evidence sentences there are (6). |
| `evidence_order` | How the evidence is sorted. `retrieval_score_desc` means best match first. |
| `judge_verdict` | Where the LLM-judge result will go once that check runs. Empty for now. |

### `example_type` values

| Value | What it means |
|---|---|
| `answerable_single` | One sentence answers the question. |
| `answerable_multi_sentence` | It takes several sentences from one paper. |
| `answerable_multi_doc` | It takes sentences from two different papers. |
| `unanswerable_gold_removed` | Same question as an answerable example, but with the answer sentences taken out. |
| `unanswerable_not_in_text` | The answer is somewhere we didn't extract, like a table. |
| `unanswerable_near_miss` | The evidence talks about the topic but doesn't actually answer. |
| `unanswerable_out_of_scope` | The question is about something the papers don't cover. |

### What `excluded_sentences` is:

The model has to learn when to say "I can't answer this." To teach that, we need examples where the evidence doesn't contain the answer. The easy way to make one is to take a question the model can answer but then remove the sentences that answer it.


