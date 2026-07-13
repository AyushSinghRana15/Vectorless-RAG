# 1. Lab Title

## Vectorless RAG with PageIndex

# What is Vectorless RAG?

**Vectorless RAG** is a retrieval-augmented generation approach that replaces traditional embedding-based vector search with LLM-driven reasoning over a document's structural tree.

Traditional RAG systems rely on chunking documents into passages, computing embeddings, and performing similarity searches against a vector database. While effective, this approach introduces complexity (embedding models, vector stores, similarity thresholds) and can lose document structure.

**Vectorless RAG** instead uses [PageIndex](https://www.pageindex.ai) to build a hierarchical tree of the document — capturing titles, sections, and summaries without embeddings. An LLM then reasons over this compact tree to select the most relevant nodes, retrieving only the evidence needed to answer the user's question.

This approach offers **Context Engineering** benefits:
- **Context Quarantine**: Each retrieval step isolates only relevant nodes, preventing context clutter.
- **Transparency**: The tree structure and LLM selection rationale are fully inspectable.
- **Simplicity**: No embedding model or vector database required.

# 2. Problem Statement / Use Case Overview

How do we build a RAG system that can answer questions over PDF documents — without embeddings, vector databases, or complex infrastructure — while maintaining transparency and explainability?

The workflow in this lab:
1. Load a PDF and submit it to PageIndex, which generates a hierarchical document tree.
2. Use an LLM (via the free Groq API) to reason over the tree and select relevant nodes.
3. Extract evidence text from selected nodes (with page attribution).
4. Generate a final answer with source pages and an explainability summary.
5. Evaluate the answer for groundedness and quality via LLM-as-Judge.

Each step is modular and inspectable, making the system auditable end-to-end.

# 3. Input Data

| Item | Detail |
|------|--------|
| PDF document | Any PDF file placed in a local `data/` folder |
| User question | Natural-language question about the PDF |
| `GROQ_API_KEY` | Free API key from [console.groq.com/keys](https://console.groq.com/keys) — used for LLM inference |
| `PAGEINDEX_API_KEY` | API key from [www.pageindex.ai](https://www.pageindex.ai) — used for document tree generation |

# 4. Processing

```
User Question + PDF
        │
        ▼
PageIndex Client ──── check cache ────►  data/cache/{stem}_tree.json
        │                                         │
        │                                    cache hit? ──yes──► load cached tree
        │                                         │
        │                                        no
        │                                         │
        ▼                                         │
submit PDF ──────────────────────────────►  PageIndex API
        │                                         │
        │                                    document tree
        │◄─────────────────────────────────────────
        │
        ▼
save tree to cache
        │
        ▼
LLM (Groq) ──── reason over tree ────►  selected node IDs + reasoning
        │
        ▼
Evidence Extractor ──── collect evidence (with pages) ──►  evidence + source pages
        │
        ▼
LLM (Groq) ──── generate answer ────►  final answer + explainability
```

1. **PageIndex Client** checks for a cached tree in `data/cache/`. If found, it loads the cached tree directly. If not, it submits the PDF to PageIndex, waits for processing, retrieves the tree, and saves it to cache for future runs.
2. **Tree Search LLM** receives the tree (without full text) and selects the node IDs most likely to contain the answer, producing a retrieval rationale. If no nodes are found, a retry with broader search is attempted.
3. **Evidence Extractor** pulls the full text from selected nodes, along with their page numbers and titles. Nodes are reordered by keyword relevance so the most relevant evidence comes first.
4. **Evidence Validation** checks if key phrases from the question appear in the retrieved evidence. If validation fails, a **keyword fallback** searches all nodes for keyword matches and merges the results.
5. **Answer LLM** uses only the evidence text and original question to produce a structured JSON response with the answer and explainability.

# 5. Output

A structured JSON response with two parts:

```json
{
  "final_answer": "concise answer for the user",
  "source_pages": [1, 3, 5],
  "explainability": {
    "retrieval_summary": "brief note about why the selected evidence is relevant",
    "evidence_used": ["short evidence note 1", "short evidence note 2"]
  }
}
```

The system also prints:
- Selected node IDs and retrieval rationale from the tree search step.
- Evidence sources with page numbers and node titles.
- Preview of the extracted evidence text.
- Groundedness check (whether the answer is supported by the evidence).
- LLM-as-Judge evaluation scoring correctness, completeness, conciseness, and faithfulness.

# 6. Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Meta Llama 3 via Groq — choose between `llama-3.3-70b-versatile` (larger) and `llama-3.1-8b-instant` (faster) at runtime |
| Document Indexing | [PageIndex](https://www.pageindex.ai) — hierarchical document tree generation |
| LLM Client | OpenAI Python SDK (compatible with Groq's endpoint) |
| PDF Processing | PyMuPDF (`pymupdf`) |
| Configuration | `python-dotenv` for `.env` file support |
| Environment | Supports both local Jupyter and Google Colab |
| Language | Python 3.11.0 |
| Runtime | Jupyter Notebook |

# 7. Underlying Concepts

- **Vectorless RAG** — Retrieval-augmented generation that uses LLM reasoning over document structure instead of embedding-based similarity search.
- **Context Engineering** — Deliberate design of what information enters the LLM's context window at each step, avoiding context distraction and context clash.
- **Context Quarantine** — Isolating sub-tasks into separate context windows so each LLM call receives only relevant information.
- **Document Tree** — A hierarchical representation of a document with titles, summaries, and text nodes — generated by PageIndex without embeddings.
- **Two-Pass Retrieval** — First pass selects relevant nodes via tree reasoning; second pass extracts evidence and generates the answer.
- **Evidence Validation** — Checking if key phrases from the question appear in the retrieved evidence to ensure retrieval quality.
- **Keyword Fallback** — A backup retrieval strategy that searches all nodes for keyword matches when LLM-based retrieval misses relevant sections.
- **Node Relevance Sorting** — Ordering selected nodes by keyword relevance so the most pertinent evidence is presented first.
- **Explainability** — Structured output that includes not just the answer but also why the evidence was selected and what was used.
- **Page Attribution** — Each evidence passage is tagged with its source page number, allowing the answer to cite specific pages.
- **Groundedness Check** — A secondary LLM call that verifies the answer is supported by the retrieved evidence.
- **LLM-as-Judge** — An automated evaluation framework that scores answer quality across correctness, completeness, conciseness, and faithfulness criteria.

> Refer to the PageIndex documentation for more on hierarchical document indexing.

# 8. Pre-requisites

- Basic familiarity with Python (functions, `import` statements, async/await).
- A **Groq API key** — free tier available at [console.groq.com/keys](https://console.groq.com/keys).
- A **PageIndex API key** — from [www.pageindex.ai](https://www.pageindex.ai).
- A PDF document to place in the `data/` folder.
- High-level understanding of what an LLM is and what RAG means.

# 9. Environment / Dependencies Setup

The cell below installs all required Python packages:

| Package | Purpose |
|---------|---------|
| `pageindex` | PageIndex SDK — submits PDFs and retrieves document trees |
| `openai` | OpenAI-compatible client (used to talk to Groq's endpoint) |
| `python-dotenv` | Loads environment variables from a `.env` file |
| `pymupdf` | PDF parsing and text extraction |
| `ipywidgets` | File upload widget and PDF selection dropdown (local Jupyter) |

Run this cell first — it only needs to be run once per session.

```bash
pip install -q --upgrade pageindex openai python-dotenv pymupdf ipywidgets
```

### Set up your API Keys

You will need two API keys. Click the links below to obtain them:

| Key | Where to get it |
|-----|------------------|
| `GROQ_API_KEY` | [https://console.groq.com/keys](https://console.groq.com/keys) |
| `PAGEINDEX_API_KEY` | [https://www.pageindex.ai](https://www.pageindex.ai) |

The notebook reads environment variables from a `.env` file or your shell.
If either key is missing, you will be prompted to enter it securely.

### Initialize the Clients

```python
from pathlib import Path
import getpass
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pageindex import PageIndexClient
import pageindex.utils as pi_utils

load_dotenv()

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
LLM_BASE_URL = "https://api.groq.com/openai/v1"

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
if not PAGEINDEX_API_KEY:
    PAGEINDEX_API_KEY = getpass.getpass("Enter your PAGEINDEX_API_KEY: ").strip()

LLM_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not LLM_API_KEY:
    LLM_API_KEY = getpass.getpass("Enter your GROQ_API_KEY: ").strip()

PDF_NAME = os.getenv("PDF_NAME")

MODELS = {
    "1": ("llama-3.3-70b-versatile", "Llama 3.3 70B — larger, more capable"),
    "2": ("llama-3.1-8b-instant", "Llama 3.1 8B — faster, lighter"),
}

print("Choose an LLM model:")
for key, (name, desc) in MODELS.items():
    print(f"  {key}. {name} — {desc}")

choice = input("Enter 1 or 2: ").strip()
LLM_MODEL = MODELS.get(choice, MODELS["1"])[0]
print(f"Using model: {LLM_MODEL}")

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY) if PAGEINDEX_API_KEY else None
llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"Expected JSON output, got: {text[:500]}")
    return json.loads(match.group(0))


async def call_llm(system_prompt: str, user_prompt: str, model: str = LLM_MODEL, temperature: float = 0.0) -> str:
    if llm_client is None:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    import openai
    import asyncio
    import random
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt == max_attempts:
                raise RuntimeError("Groq service is busy, please try again shortly.")
            wait = min(2 ** attempt + random.uniform(0, 2), 30)
            print(f"Rate limited, retrying in {wait:.0f}s... (attempt {attempt}/{max_attempts})")
            await asyncio.sleep(wait)


async def call_llm_and_parse(system_prompt: str, user_prompt: str, model: str = LLM_MODEL, temperature: float = 0.0) -> dict[str, Any]:
    reply = await call_llm(system_prompt, user_prompt, model, temperature)
    try:
        return extract_json(reply)
    except ValueError:
        print("Received malformed JSON, sending one corrective retry...")
        reply2 = await call_llm(
            system_prompt,
            f"Your previous reply was not valid JSON:\n\n{reply}\n\nReturn ONLY the valid JSON object, with no extra text, no markdown fences, and no explanation.",
            model,
            temperature,
        )
        try:
            return extract_json(reply2)
        except ValueError:
            raise RuntimeError("The AI model failed to return valid JSON after 2 attempts. Please try rephrasing your question.")


def preview_text(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."
```

---

### Build the Vectorless RAG Pipeline

This is the core of the lab. Follow each sub-step in order.

#### 1) Load a PDF from `data/`

Upload a PDF using the button, or place PDF files directly in the `data/` folder.
The `data/` directory is created automatically if it doesn't exist.
Then use the dropdown to select which PDF to process.

The notebook automatically detects whether you're running in Google Colab or local Jupyter and uses the appropriate upload method.

```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DATA_DIR.mkdir(exist_ok=True)

if IN_COLAB:
    from google.colab import files as colab_files
    print("Upload a PDF file (Colab mode):")
    uploaded = colab_files.upload()
    for fname, data in uploaded.items():
        save_path = DATA_DIR / fname
        save_path.write_bytes(data)
        print(f"Saved: {save_path}")
else:
    import ipywidgets as widgets
    from IPython.display import display

    upload_output = widgets.Output()
    upload_button = widgets.FileUpload(
        description='Upload PDF',
        accept='.pdf',
        multiple=False,
        style={'description_width': 'initial'},
    )
    display(upload_button)

    _saved_files = set()

    def on_upload_change(change):
        with upload_output:
            upload_output.clear_output()
            if change['new']:
                for entry in change['new']:
                    if entry.name in _saved_files:
                        continue
                    if not entry.content:
                        continue
                    save_path = DATA_DIR / entry.name
                    save_path.write_bytes(entry.content)
                    _saved_files.add(entry.name)
                    print(f"Saved: {save_path}")

    upload_button.observe(on_upload_change, names='value')
    display(upload_output)

# PDF selection dropdown
import ipywidgets as widgets
from IPython.display import display

pdf_files = sorted(p.name for p in DATA_DIR.glob('*.pdf'))
if not pdf_files:
    print("⚠ No PDF files found. Upload a PDF above, then re-run this cell.")
else:
    pdf_dropdown = widgets.Dropdown(
        options=pdf_files,
        value=pdf_files[0],
        description='Select PDF:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px'),
    )

    display(pdf_dropdown)

    pdf_path = DATA_DIR / pdf_files[0]

    def on_pdf_change(change):
        global pdf_path
        pdf_path = DATA_DIR / change['new']

    pdf_dropdown.observe(on_pdf_change, names='value')

# PDF validation
MAX_PDF_SIZE_MB = 20
with open(pdf_path, 'rb') as f:
    header = f.read(4)
if header != b'%PDF':
    raise ValueError(f"{pdf_path.name} does not appear to be a valid PDF file (missing %PDF signature).")

size_mb = pdf_path.stat().st_size / (1024 * 1024)
if size_mb > MAX_PDF_SIZE_MB:
    raise ValueError(f"{pdf_path.name} is {size_mb:.1f} MB, which exceeds the {MAX_PDF_SIZE_MB} MB limit.")

print(f"Using PDF: {pdf_path}")
```

#### 2) Build the PageIndex tree

Submit the PDF to PageIndex and retrieve the generated document tree. The tree is cached locally in `data/cache/` so subsequent runs skip the PageIndex API call for the same document. If the document is still processing, the cell polls with a progress bar until ready.

```python
import time

if 'pdf_path' not in dir():
    raise NameError(
        "'pdf_path' is not defined. Please run the PDF selection cell above first."
    )

CACHE_DIR.mkdir(exist_ok=True)
cache_path = CACHE_DIR / f"{pdf_path.stem}_tree.json"

if cache_path.exists():
    print(f"Loading cached tree from {cache_path}")
    tree = json.loads(cache_path.read_text())
else:
    if pi_client is None:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured.")

    submitted = pi_client.submit_document(str(pdf_path))
    doc_id = submitted.get("doc_id") or submitted.get("result", {}).get("doc_id")
    if not doc_id:
        raise RuntimeError(f"Could not read doc_id from PageIndex response: {submitted}")

    print(f"Submitted document id: {doc_id}")
    print("Waiting for PageIndex to process the document...")

    poll_interval = 5
    max_wait = 300
    elapsed = 0
    spinner = ["|", "/", "-", "\\"]
    while elapsed < max_wait:
        if pi_client.is_retrieval_ready(doc_id):
            break
        idx = (elapsed // poll_interval) % len(spinner)
        print(f"\r  {spinner[idx]}  Waiting... {elapsed}s / {max_wait}s", end="", flush=True)
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        raise RuntimeError(
            f"PageIndex did not finish within {max_wait}s. Rerun this cell later."
        )
    print(f"\r  Done! Processed in {elapsed}s.{' ' * 20}")

    tree_response = pi_client.get_tree(doc_id, node_summary=True)
    tree = tree_response.get("result", tree_response)

    cache_path.write_text(json.dumps(tree))
    print(f"Tree cached to {cache_path}")

print("Tree ready. Top-level preview:")
pi_utils.print_tree(tree[:2] if isinstance(tree, list) else tree)
```

#### 3) Prepare retrieval helpers

Build a node map and helper functions for extracting text and source metadata from selected nodes.

```python
node_map = pi_utils.create_node_mapping(tree)

def tree_as_prompt_text(document_tree: object) -> str:
    tree_copy = document_tree.copy() if isinstance(document_tree, dict) else document_tree
    tree_without_text = pi_utils.remove_fields(tree_copy, fields=["text"])
    return json.dumps(tree_without_text, indent=2)


def collect_evidence(node_ids: list[str], max_chars_per_node: int = 4000, max_nodes: int = 4) -> dict:
    """Return {'text': str, 'sources': [{'node_id': ..., 'page': ..., 'title': ...}]}"""
    parts: list[str] = []
    sources: list[dict] = []
    for node_id in node_ids[:max_nodes]:
        node = node_map.get(node_id)
        if not node:
            continue
        text = node.get("text", "")
        if len(text) > max_chars_per_node:
            text = text[:max_chars_per_node].rstrip() + "...[truncated]"
        title = node.get("title", node_id)
        page_index = node.get("page_index", "?")
        parts.append(f"[Page {page_index}] {title}\n{text}")
        sources.append({"node_id": node_id, "page": page_index, "title": title})
    return {"text": "\n\n".join(parts), "sources": sources}


print(f"Indexed nodes: {len(node_map)}")
```

#### 4) Build the answer pipeline

The `answer_question` function orchestrates the three-stage RAG pipeline: tree search, evidence extraction, and answer generation. It includes keyword fallback and evidence validation for robust retrieval.

```python
import re as _re


def _question_phrases(q: str) -> list[str]:
    """Extract key noun phrases from a question for evidence validation."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "does", "do", "did", "has", "have", "had", "will", "would",
        "can", "could", "shall", "should", "may", "might", "must",
        "for", "of", "in", "on", "at", "to", "by", "with", "from",
        "this", "that", "these", "those", "it", "its", "or", "and",
        "not", "no", "if", "than", "then", "so", "as", "but",
    }
    words = _re.findall(r"[a-zA-Z0-9%]+", q.lower())
    specific = [w for w in words if w not in stopwords and len(w) > 2]
    non_stop = [w for w in words if w not in stopwords]
    bigrams = [f"{non_stop[i]} {non_stop[i+1]}" for i in range(len(non_stop) - 1)]
    return specific + bigrams


def _evidence_has_phrases(evidence_text: str, phrases: list[str], threshold: float = 0.5) -> bool:
    """Check if enough key phrases from the question appear in the evidence."""
    if not phrases:
        return True
    text_lower = evidence_text.lower()
    bigrams = [p for p in phrases if " " in p]
    singles = [p for p in phrases if " " not in p]
    if bigrams:
        bigram_hits = sum(1 for bg in bigrams if bg in text_lower)
        if bigram_hits == 0:
            return False
    if singles:
        hits = sum(1 for s in singles if s in text_lower)
        if hits / len(singles) < threshold:
            return False
    return True


def _keyword_fallback_node_ids(question: str, tree_nodes: list, node_map: dict) -> list[str]:
    """Search all nodes for keyword matches when LLM retrieval misses."""
    phrases = _question_phrases(question)
    if not phrases:
        return []
    scores: list[tuple[int, str]] = []
    for node in tree_nodes:
        nid = node.get("node_id", "")
        n = node_map.get(nid)
        if not n:
            continue
        searchable = (
            n.get("title", "") + " " +
            n.get("summary", "") + " " +
            n.get("text", "")
        ).lower()
        hits = sum(1 for p in phrases if p in searchable)
        if hits > 0:
            scores.append((hits, nid))
    scores.sort(reverse=True)
    return [nid for _, nid in scores[:5]]


def _flatten_tree(tree_obj) -> list:
    """Recursively flatten a tree into a list of all nodes."""
    nodes = []
    if isinstance(tree_obj, list):
        for item in tree_obj:
            nodes.append(item)
            nodes.extend(_flatten_tree(item.get("children", [])))
    elif isinstance(tree_obj, dict):
        nodes.append(tree_obj)
        nodes.extend(_flatten_tree(tree_obj.get("children", [])))
    return nodes


async def answer_question(question: str, model: str = LLM_MODEL) -> dict[str, Any]:
    """Run retrieval -> evidence extraction -> final answer as one atomic step."""
    question = question.strip()
    if not question:
        raise ValueError("A question is required to continue.")

    all_nodes = _flatten_tree(tree)
    phrases = _question_phrases(question)

    # Step 1: reasoning-based tree search
    retrieval_system_prompt = """
You are a document retrieval assistant for a vectorless RAG system.

You will receive a user question and a PageIndex tree made of node titles and summaries.
IMPORTANT: Each node's title is only a label — the node may contain much more content
covering adjacent clauses, definitions, footnotes, and cross-references. A node titled
"5.12 Long-Term Discount" may also contain clauses 6.x, 7.x, footnotes, etc.
Read the summaries carefully and include any node whose content range could plausibly
contain the answer.

Err on the side of INCLUSION — it is better to retrieve an extra node than to miss
the one that contains the answer.

Return valid JSON with this shape:
{
  "thinking": "short, evidence-based retrieval explanation",
  "node_list": ["node_id_1", "node_id_2"]
}

Do not output markdown, prose, or extra keys.
Keep the explanation brief and grounded in the tree.
""".strip()

    retrieval_user_prompt = f"""
Question:
{question}

PageIndex tree:
{tree_as_prompt_text(tree)}
""".strip()

    retrieval_json = await call_llm_and_parse(retrieval_system_prompt, retrieval_user_prompt, model=model)
    selected_node_ids = retrieval_json.get("node_list", [])
    retrieval_thinking = retrieval_json.get("thinking", "")

    # Retry with broader search if nothing was found
    if not selected_node_ids:
        print("No relevant sections found on first attempt, retrying with a broader search...")
        retry_retrieval_user_prompt = f"""
Your previous search returned no relevant sections for this question:

{question}

Look again more carefully. Include any section that is even loosely or
indirectly relevant — err on the side of including a section rather than
excluding it, unless you are certain nothing in the document relates to
this question.

PageIndex tree:
{tree_as_prompt_text(tree)}
""".strip()
        retry_json = await call_llm_and_parse(retrieval_system_prompt, retry_retrieval_user_prompt, model=model)
        selected_node_ids = retry_json.get("node_list", [])
        retrieval_thinking = retry_json.get("thinking", "")

    if not selected_node_ids:
        return {
            "question": question,
            "selected_node_ids": [],
            "retrieval_thinking": retrieval_thinking,
            "evidence_text": "",
            "source_pages": [],
            "evidence_sources": [],
            "final_answer": "No relevant section was found in the document for this question.",
            "explainability": {},
        }

    # Step 2: evidence extraction (reorder by keyword relevance)
    phrases_for_sort = _question_phrases(question)
    def _node_relevance(nid: str) -> int:
        n = node_map.get(nid)
        if not n:
            return 0
        searchable = (n.get("title", "") + " " + n.get("summary", "") + " " + n.get("text", "")).lower()
        return sum(1 for p in phrases_for_sort if p in searchable)
    selected_node_ids = sorted(selected_node_ids, key=_node_relevance, reverse=True)

    evidence = collect_evidence(selected_node_ids)
    evidence_text = evidence["text"]
    evidence_sources = evidence["sources"]

    # Step 2b: validate evidence — does it contain question key phrases?
    if phrases and not _evidence_has_phrases(evidence_text, phrases, threshold=0.5):
        print(f"Evidence missing key question phrases, running keyword fallback...")
        fallback_ids = _keyword_fallback_node_ids(question, all_nodes, node_map)
        seen = set(selected_node_ids)
        for fid in fallback_ids:
            if fid not in seen:
                selected_node_ids.append(fid)
                seen.add(fid)
        evidence = collect_evidence(selected_node_ids)
        evidence_text = evidence["text"]
        evidence_sources = evidence["sources"]
        retrieval_thinking += "\n[Keyword fallback added nodes: " + ", ".join(fallback_ids) + "]"

    source_pages = sorted({s["page"] for s in evidence_sources if s["page"] != "?"})

    # Step 3: final answer generation
    final_system_prompt = """
You are a careful assistant answering questions about a PDF document.

Use only the evidence text provided by the notebook.
Do not invent facts or rely on outside knowledge.
If the evidence does not contain the answer, say so explicitly instead of guessing.
Return valid JSON with this shape:
{
  "final_answer": "concise answer for the user",
  "explainability": {
    "retrieval_summary": "brief note about why the selected evidence is relevant",
    "evidence_used": ["short evidence note 1", "short evidence note 2"]
  }
}

Keep the explanation short and grounded in the provided context.
""".strip()

    evidence_for_prompt = evidence_text.strip() or "(No evidence text was retrieved for this question.)"
    final_user_prompt = f"""
Question:
{question}

Evidence context:
{evidence_for_prompt}
""".strip()

    final_json = await call_llm_and_parse(final_system_prompt, final_user_prompt, model=model)

    return {
        "question": question,
        "selected_node_ids": selected_node_ids,
        "retrieval_thinking": retrieval_thinking,
        "evidence_text": evidence_text,
        "source_pages": source_pages,
        "evidence_sources": evidence_sources,
        "final_answer": final_json.get("final_answer", ""),
        "explainability": final_json.get("explainability", {}),
    }
```

---

### Run a Query Through the Pipeline

Now ask a question about the PDF. The system will retrieve relevant evidence and generate an answer with explainability. Each output section is displayed in its own cell for clarity.

#### 6a. Question & Retrieval Explainability

```python
QUESTION = input("Enter your question about the PDF: ").strip()

result = await answer_question(QUESTION)
```

```python
print("=" * 70)
print("QUESTION")
print("=" * 70)
print(result["question"])

print()
print("=" * 70)
print("RETRIEVAL EXPLAINABILITY")
print("=" * 70)
print("Selected node IDs:", result["selected_node_ids"])
print(result["retrieval_thinking"])
```

#### 6b. Retrieved Evidence

```python
if result.get("evidence_sources"):
    print("Source pages:", result.get("source_pages", []))
    print()
    for src in result["evidence_sources"]:
        print(f"  node={src['node_id']}  page={src['page']}  {src['title']}")
else:
    print("(No evidence retrieved.)")
```

```python
print("=" * 70)
print("EVIDENCE PREVIEW")
print("=" * 70)
print(preview_text(result["evidence_text"], limit=3000))
```

#### 6c. Final Answer

```python
print("=" * 70)
print("FINAL ANSWER")
print("=" * 70)
print(result["final_answer"])

print()
print("Source pages:", result.get("source_pages", []))
```

#### 6d. Explainability

```python
print("=" * 70)
print("EXPLAINABILITY")
print("=" * 70)
print(json.dumps(result["explainability"], indent=2))
```

#### 6e. Groundedness Check

```python
if result["evidence_text"]:
    import re as _re
    clean_evidence = _re.sub(r"\[Page \d+\] [^\n]+\n", "", result["evidence_text"])

    groundedness_system_prompt = """
You are a fact-checking assistant. You will be given an answer and the
evidence text it was supposed to be based on. The evidence below is
extracted directly from the source document. Determine whether the
answer's claims are actually supported by the evidence.

Return valid JSON with this shape:
{
  "grounded": true or false,
  "reason": "one short sentence explaining your judgment"
}

Do not output markdown, prose, or extra keys.
""".strip()

    groundedness_user_prompt = f"""
Answer to check:
{result['final_answer']}

Evidence it should be based on:
{clean_evidence}
""".strip()

    try:
        groundedness_json = await call_llm_and_parse(groundedness_system_prompt, groundedness_user_prompt, model=LLM_MODEL)
        if groundedness_json.get('grounded') is False:
            print("WARNING: This answer may not be fully supported by the retrieved evidence.")
            print(f"Reason: {groundedness_json.get('reason', 'No reason given.')}")
        else:
            print("PASSED — answer is supported by the retrieved evidence.")
    except Exception:
        print("(Groundedness check could not be completed.)")
else:
    print("(No evidence to check against.)")
```

#### 6f. LLM-as-Judge Evaluation

```python
judge_system_prompt = """
You are an impartial evaluator ("LLM-as-Judge") for a RAG system.
You will be given a question, the final answer produced by the system,
the evidence it was based on, and the explainability summary.

Score the answer on each criterion below using a 1-5 scale:

| Criterion        | 1 (Poor)                        | 5 (Excellent)                              |
|------------------|----------------------------------|---------------------------------------------|
| Correctness      | Factually wrong                  | Fully correct, no errors                    |
| Completeness     | Misses key parts of the question | Answers all parts of the question            |
| Conciseness      | Verbose or contains filler        | Tight, no unnecessary information           |
| Faithfulness     | Invents facts not in evidence    | Every claim traceable to provided evidence  |

Return valid JSON with this shape:
{
  "correctness": {"score": int, "reason": "short explanation"},
  "completeness": {"score": int, "reason": "short explanation"},
  "conciseness": {"score": int, "reason": "short explanation"},
  "faithfulness": {"score": int, "reason": "short explanation"},
  "overall_score": float,
  "overall_summary": "one-paragraph overall assessment"
}

overall_score should be the average of the four scores, rounded to one decimal.
Do not output markdown, prose, or extra keys.
""".strip()

judge_user_prompt = f"""
Question:
{result['question']}

Final Answer:
{result['final_answer']}

Evidence:
{result['evidence_text']}

Explainability:
{json.dumps(result['explainability'], indent=2)}
""".strip()

try:
    judge_json = await call_llm_and_parse(judge_system_prompt, judge_user_prompt, model=LLM_MODEL)

    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 70)
    for criterion in ["correctness", "completeness", "conciseness", "faithfulness"]:
        entry = judge_json.get(criterion, {})
        print(f"\n{criterion.upper()}: {entry.get('score', '?')}/5")
        print(f"  {entry.get('reason', '')}")
    print(f"\nOVERALL: {judge_json.get('overall_score', '?')}/5")
    print(f"{judge_json.get('overall_summary', '')}")
except Exception as e:
    print(f"(LLM-as-Judge evaluation could not be completed: {e})")
```

# 10. Conclusion

In this lab, you built a complete Vectorless RAG pipeline that retrieves and reasons over PDF documents without embeddings, vector databases, or complex infrastructure. Key takeaways:

- **Vectorless RAG eliminates the embedding bottleneck** by using LLM-driven tree reasoning instead of similarity search, reducing infrastructure complexity while maintaining retrieval quality.
- **Context Engineering principles** — particularly context quarantine and structured evidence collection — ensure each LLM call receives only the information it needs, improving accuracy and reducing noise.
- **Evidence validation and keyword fallback** provide a safety net: if LLM-based retrieval misses relevant sections, a keyword search catches them, improving recall without sacrificing precision.
- **End-to-end explainability** is built into the pipeline: from retrieval rationale and source page attribution to groundedness checks and LLM-as-Judge evaluation, every step is auditable.
- **PageIndex document trees** preserve document structure (titles, sections, summaries) that would be lost in traditional chunking-based approaches.
- **Environment flexibility** — the notebook supports both local Jupyter and Google Colab, making it easy to run anywhere.


