# 1. Lab Title

## Vectorless RAG: Multi-Hop Reasoning & Structured Data Retrieval

# What is Multi-Hop RAG?

**Multi-hop RAG** is a retrieval-augmented generation pattern where answering a single question requires fetching evidence from **multiple distant sections** of a document.

Traditional chunking-based RAG retrieves a single text chunk per query. This works for simple lookups but fails when the answer depends on combining facts from different parts of a document — for example, matching a date from one section against a condition defined in another, or combining definitions from separate chapters.

**Vectorless RAG** solves this by letting an LLM reason over the **entire document tree** and explicitly select any combination of nodes, regardless of where they appear. This enables true multi-hop reasoning without the noise and fragmentation introduced by text chunking.

# 2. Problem Statement / Use Case Overview

This lab demonstrates two advanced Vectorless RAG scenarios where standard chunking-based RAG fails:

### Scenario A: Cross-Reference Analysis

Many documents require cross-referencing multiple sections to answer a single question — a legal contract's definitions and obligations, a technical manual's specs and compatibility tables, or a research paper's methodology and results. Standard RAG retrieves a single chunk. If the relevant facts are in different sections, chunking either misses one or mixes in irrelevant surrounding text.

### Scenario B: Tabular Data & Structured Information

Documents often contain structured tables — comparison charts, schedules, matrices, or rate cards — that are critical for accurate answers. Standard chunking **shatters** tables:
- A table spanning one page might be split into 3-4 chunks.
- Headers get separated from their rows.
- Footnotes referencing table cells get lost.

Vectorless RAG retrieves **whole logical nodes**, preserving table structure, headers, and footnotes intact.

# 3. Input Data

| Item | Detail |
|------|--------|
| PDF document | Any complex, multi-section PDF with tables, cross-references, or structured data |
| Cross-reference question | A multi-hop question requiring facts from multiple sections |
| Table question | A question about a table or structured data |
| `GROQ_API_KEY` | Free API key from [console.groq.com/keys](https://console.groq.com/keys) |
| `PAGEINDEX_API_KEY` | API key from [www.pageindex.ai](https://www.pageindex.ai) |

# 4. Processing

```
PDF Document
    │
    ▼
PageIndex Client ──── submit PDF ────►  PageIndex API
    │                                         │
    │                                    document tree
    │◄─────────────────────────────────────────
    │
    ├─── Scenario A: Cross-Reference ──────────────────────────────────
    │         │
    │         ▼
    │    LLM (Groq) ──── select multiple nodes ────►  [section_a, section_b, ...]
    │         │
    │         ▼
    │    Evidence Extractor ──── collect all nodes ────►  combined evidence
    │         │
    │         ▼
    │    LLM (Groq) ──── reason over evidence ────►  answer + explainability
    │
    └─── Scenario B: Table Extraction ───────────────────────────────
              │
              ▼
         LLM (Groq) ──── identify table node ────►  [table_node]
              │
              ▼
         Evidence Extractor ──── collect full table ────►  structured table
              │
              ▼
         LLM (Groq) ──── read table + answer ────►  specific structured-data answer
```

1. **PageIndex Client** submits the PDF (or loads a cached tree) and retrieves a hierarchical tree of titles, summaries, and text.
2. **Scenario A**: The LLM selects **multiple nodes** from different sections and combines them for cross-reference analysis.
3. **Scenario B**: The LLM identifies the **table node** and retrieves it intact — preserving headers, rows, and footnotes.
4. **Answer LLM** uses only the retrieved evidence to produce a structured JSON response.

# 5. Output

### Scenario A: Cross-Reference Response

```json
{
  "final_answer": "The eligibility criteria require X and Y. The maximum limit is Z.",
  "explainability": {
    "section_a_summary": "relevant information found in section A",
    "section_b_summary": "relevant information found in section B",
    "evidence_used": ["node xyz: Section A title", "node abc: Section B title"]
  }
}
```

### Scenario B: Table-Based Response

```json
{
  "final_answer": "The Q3 revenue for Product X is $1.2M.",
  "table_structure": {
    "table_title": "Section 5.2 — Q3 Comparison Table",
    "relevant_row": "Product X",
    "relevant_column": "Q3 Revenue"
  },
  "explainability": {
    "evidence_used": ["Complete comparison table from node 123"]
  }
}
```

# 6. Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Meta Llama 3 via Groq — choose between `llama-3.3-70b-versatile` (larger) and `llama-3.1-8b-instant` (faster) at runtime |
| Document Indexing | [PageIndex](https://www.pageindex.ai) — hierarchical document tree generation |
| LLM Client | OpenAI Python SDK (compatible with Groq's endpoint) |
| PDF Processing | PyMuPDF (`pymupdf`) |
| Configuration | `python-dotenv` for `.env` file support |
| Language | Python 3.11.0 |
| Runtime | Jupyter Notebook |

# 7. Underlying Concepts

- **Multi-Hop Retrieval** — Fetching evidence from multiple distant sections of a document to answer a single question.
- **Vectorless RAG** — Retrieval-augmented generation that uses LLM reasoning over document structure instead of embedding-based similarity search.
- **Context Engineering** — Deliberate design of what information enters the LLM's context window at each step.
- **Context Quarantine** — Isolating sub-tasks into separate context windows so each LLM call receives only relevant information.
- **Table Preservation** — Retrieving whole logical nodes that contain complete tables, avoiding the fragmentation caused by text chunking.
- **Two-Pass Retrieval** — First pass selects relevant nodes via tree reasoning; second pass extracts evidence and generates the answer.
- **Explainability** — Structured output that includes not just the answer but also which sections were used and why.

> Refer to Lab 1 for the foundational Vectorless RAG concepts.

# 8. Pre-requisites

- Basic familiarity with Python (functions, `import` statements, async/await).
- A **Groq API key** — free tier available at [console.groq.com/keys](https://console.groq.com/keys).
- A **PageIndex API key** — from [www.pageindex.ai](https://www.pageindex.ai).
- A complex, multi-section PDF document to place in the `data/` folder.
- Completion of **Lab 1** (Vectorless RAG basics) recommended.

# 9. Environment / Dependencies Setup

The cell below installs all required Python packages:

| Package | Purpose |
|---------|---------|
| `pageindex` | PageIndex SDK — submits PDFs and retrieves document trees |
| `openai` | OpenAI-compatible client (used to talk to Groq's endpoint) |
| `python-dotenv` | Loads environment variables from a `.env` file |
| `pymupdf` | PDF parsing and text extraction |
| `networkx` | Graph construction for retrieval visualization |
| `matplotlib` | Plotting the multi-hop and table retrieval graphs |

Run this cell first — it only needs to be run once per session.

```bash
pip install -q --upgrade pageindex openai python-dotenv pymupdf networkx matplotlib
```

### Set up your API Keys

You will need two API keys. Click the links below to obtain them:

| Key | Where to get it |
|-----|------------------|
| `GROQ_API_KEY` | [https://console.groq.com/keys](https://console.groq.com/keys) |
| `PAGEINDEX_API_KEY` | [https://www.pageindex.ai](https://www.pageindex.ai) |

Set them as environment variables or paste them into a `.env` file in the same directory as this notebook:

```
GROQ_API_KEY=gsk_...
PAGEINDEX_API_KEY=...
```

### Initialize the Clients

```python
from pathlib import Path
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
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
LLM_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_BASE_URL = "https://api.groq.com/openai/v1"
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

if not PAGEINDEX_API_KEY:
    print("Set PAGEINDEX_API_KEY before running the PageIndex cells.")
if not LLM_API_KEY:
    print("Set GROQ_API_KEY before running the retrieval and answer cells.")

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
    response = await llm_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


async def call_llm_and_parse(system_prompt: str, user_prompt: str, model: str = LLM_MODEL, temperature: float = 0.0) -> dict[str, Any]:
    reply = await call_llm(system_prompt, user_prompt, model, temperature)
    try:
        return extract_json(reply)
    except ValueError:
        print("Received malformed JSON, sending one corrective retry...")
        corrective_prompt = f"Your previous reply was not valid JSON:\n\n{reply}\n\nReturn ONLY the valid JSON object, with no extra text, no markdown fences, and no explanation."
        reply2 = await call_llm(system_prompt, corrective_prompt, model, temperature)
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

Place a complex, multi-section PDF in the `data/` folder before running this cell. If `PDF_NAME` is set, the notebook uses that file; otherwise it picks the first PDF found.

```python
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Missing data folder: {DATA_DIR.resolve()}")

pdf_candidates = sorted(DATA_DIR.glob("*.pdf"))
if not pdf_candidates:
    raise FileNotFoundError(f"No PDF files found in {DATA_DIR.resolve()}")

if PDF_NAME:
    matching = [path for path in pdf_candidates if path.name == PDF_NAME]
    if not matching:
        available = ", ".join(path.name for path in pdf_candidates)
        raise FileNotFoundError(f"PDF_NAME={PDF_NAME!r} was not found in data/. Available files: {available}")
    pdf_path = matching[0]
else:
    pdf_path = pdf_candidates[0]

print(f"Using PDF: {pdf_path}")
```

#### 2) Build the PageIndex tree

Submit the PDF to PageIndex and retrieve the generated document tree. The tree is cached locally in `data/cache/` so subsequent runs skip the PageIndex API call for the same document. If the document is still processing, the cell polls with a progress bar until ready.

```python
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

Build a node map and helper functions for extracting text from selected nodes.

```python
node_map = pi_utils.create_node_mapping(tree)

def tree_as_prompt_text(document_tree: object) -> str:
    tree_copy = document_tree.copy() if isinstance(document_tree, dict) else document_tree
    tree_without_text = pi_utils.remove_fields(tree_copy, fields=["text"])
    return json.dumps(tree_without_text, indent=2)


def collect_node_text(node_ids: list[str]) -> str:
    parts: list[str] = []
    for node_id in node_ids:
        node = node_map.get(node_id)
        if not node:
            continue
        text = node.get("text", "")
        title = node.get("title", node_id)
        page_index = node.get("page_index", "?")
        parts.append(f"[node={node_id} page={page_index} title={title}]\n{text}")
    return "\n\n".join(parts)


print(f"Indexed nodes: {len(node_map)}")
```

---

### Scenario A: Multi-Hop Cross-Reference Analysis

#### A1) Define the cross-reference question

Ask a question that requires fetching facts from multiple sections of the document.

```python
CROSS_REFERENCE_QUESTION = input(
    "Enter a multi-hop question requiring facts from multiple sections: "
).strip()
if not CROSS_REFERENCE_QUESTION:
    raise ValueError("A question is required to continue.")
```

#### A2) Multi-hop tree search

The LLM receives the full document tree and explicitly identifies which sections to fetch.

```python
retrieval_system_prompt = """
You are a document retrieval assistant for a multi-hop question answering system.

You will receive a user question and a PageIndex tree made of node titles and summaries.
The question may require information from MULTIPLE distant sections of the document.

Identify ALL node IDs that contain relevant evidence. Common patterns:
- Definitions / Clarifications
- Dates / Timeframes / Deadlines
- Limits / Thresholds / Caps
- Conditions / Requirements / Eligibility
- Exclusions / Exceptions / Limitations

Return valid JSON with this shape:
{
  "thinking": "step-by-step reasoning about which sections are needed and why",
  "node_list": ["node_id_1", "node_id_2", ...]
}

Do not output markdown, prose, or extra keys.
Be thorough — missing a required node means an incomplete answer.
""".strip()

retrieval_user_prompt = f"""
Question:
{CROSS_REFERENCE_QUESTION}

PageIndex tree:
{tree_as_prompt_text(tree)}
""".strip()

retrieval_response = await call_llm(retrieval_system_prompt, retrieval_user_prompt)
retrieval_json = extract_json(retrieval_response)
selected_node_ids = retrieval_json.get("node_list", [])

print(f"Selected {len(selected_node_ids)} node(s):", selected_node_ids)
print("\nRetrieval reasoning:")
print(retrieval_json.get("thinking", ""))
```

#### A2b) Visualize the multi-hop retrieval

```python
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

G = nx.DiGraph()

node_info = []
for nid in selected_node_ids:
    node = node_map.get(nid, {})
    title = node.get('title', nid)
    page = node.get('page', '?')
    parent = node.get('parent_id', None)
    node_info.append({'id': nid, 'title': title, 'page': page, 'parent': parent})

G.add_node("Question", layer=0)
G.add_node("LLM\nTree Scan", layer=1)
G.add_edge("Question", "LLM\nTree Scan", label="analyzes tree")

colors = ["#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
node_colors = ["#4A90D9", "#3498DB"]

hop_labels = []
for idx, info in enumerate(node_info):
    hop_label = f"{info['id']}\n{info['title'][:25]}\npg {info['page']}"
    hop_labels.append(hop_label)
    G.add_node(hop_label, layer=2)
    G.add_edge("LLM\nTree Scan", hop_label, label=f"hop {idx+1}")
    node_colors.append(colors[idx % len(colors)])

for idx in range(len(node_info)):
    for jdx in range(idx + 1, len(node_info)):
        child = node_info[idx]
        candidate = node_info[jdx]
        if candidate['id'] == child.get('parent'):
            edge_label = f"parent of\n{child['id']}"
            G.add_edge(hop_labels[jdx], hop_labels[idx], label=edge_label, style="dashed")
        elif child['id'] == candidate.get('parent'):
            edge_label = f"parent of\n{candidate['id']}"
            G.add_edge(hop_labels[idx], hop_labels[jdx], label=edge_label, style="dashed")

G.add_node("Evidence\nMerged", layer=3)
G.add_node("Final\nAnswer", layer=4)
node_colors.append("#34495E")
node_colors.append("#2C3E50")

for hl in hop_labels:
    G.add_edge(hl, "Evidence\nMerged", label="contributes")

G.add_edge("Evidence\nMerged", "Final\nAnswer", label="generates")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")

assert len(node_colors) == len(G.nodes()), f"Color mismatch: {len(node_colors)} colors vs {len(G.nodes())} nodes"

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2800, alpha=0.9, ax=ax)

solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') != 'dashed']
dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']

nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color="#7F8C8D", arrows=True, arrowsize=20, width=2, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color="#E67E22", arrows=True, arrowsize=18, width=2, style="dashed", ax=ax)

nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)

edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True) if d.get('label')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color="#2C3E50", ax=ax)

legend_items = [mpatches.Patch(color=colors[i % len(colors)], label=f"{node_info[i]['id']}: {node_info[i]['title'][:35]}") for i in range(len(node_info))]
legend_items.append(mpatches.Patch(color="#E67E22", label="Tree parent/child link"))
ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)

ax.set_title("Multi-Hop Retrieval Graph — Node References", fontsize=14, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()
plt.show()
```

#### A3) Extract multi-hop evidence

```python
evidence_text = collect_node_text(selected_node_ids)
if not evidence_text.strip():
    raise RuntimeError("No evidence text was collected. Check the selected node IDs and tree mapping.")

print("Evidence preview (multi-hop):\n")
print(preview_text(evidence_text, limit=4000))
```

#### A4) Generate cross-reference answer

```python
answer_system_prompt = """
You are a document analysis assistant.

You have been given evidence from multiple sections of a document.
Analyze the evidence to answer the cross-reference question.

Return valid JSON with this shape:
{
  "final_answer": "clear, concise answer based on the combined evidence",
  "explainability": {
    "section_a_summary": "key information found in the first relevant section",
    "section_b_summary": "key information found in the second relevant section",
    "section_c_summary": "key information found in additional sections (if any)",
    "evidence_used": ["short evidence note 1", "short evidence note 2"]
  }
}

Ground every statement in the provided evidence. Do not invent facts.
""".strip()

answer_user_prompt = f"""
Question:
{CROSS_REFERENCE_QUESTION}

Evidence from document:
{evidence_text}
""".strip()

answer_response = await call_llm(answer_system_prompt, answer_user_prompt)
answer_json = extract_json(answer_response)

print("\nFinal answer:")
print(answer_json.get("final_answer", ""))
print("\nExplainability:")
print(json.dumps(answer_json.get("explainability", {}), indent=2))
```

#### A5) Groundedness check

```python
if evidence_text.strip():
    groundedness_system_prompt = """
You are a fact-checking assistant. You will be given an answer and the
evidence text it was supposed to be based on. Determine whether the
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
{answer_json['final_answer']}

Evidence it should be based on:
{evidence_text}
""".strip()

    try:
        groundedness_json = await call_llm_and_parse(groundedness_system_prompt, groundedness_user_prompt)
        if groundedness_json.get("grounded") is False:
            print("\nWARNING: This answer may not be fully supported by the retrieved evidence.")
            print(f"   Reason: {groundedness_json.get('reason', 'No reason given.')}")
        else:
            print("\nGroundedness check passed: answer is supported by the retrieved evidence.")
    except Exception:
        print("\n(Groundedness check could not be completed.)")
```

---

### Scenario B: Tabular Data & Structured Information

#### B1) Define the table query

```python
TABLE_QUESTION = input(
    "Enter a table-related question: "
).strip()
if not TABLE_QUESTION:
    raise ValueError("A question is required to continue.")
```

#### B2) Targeted tree search for tables

```python
table_retrieval_prompt = """
You are a document retrieval assistant specializing in structured data.

You will receive a user question and a PageIndex tree made of node titles and summaries.
The question likely requires information from a TABLE, CHART, or SCHEDULE in the document.

Look for nodes that contain:
- Comparison tables / data tables
- Schedules / matrices / rate cards
- Structured lists with columns and rows
- Any node whose title or summary suggests tabular data

Return valid JSON with this shape:
{
  "thinking": "which table or schedule was identified and why it matches the question",
  "node_list": ["node_id_1", "node_id_2"]
}

Do not output markdown, prose, or extra keys.
Prefer retrieving the ENTIRE table node over partial fragments.
""".strip()

table_user_prompt = f"""
Question:
{TABLE_QUESTION}

PageIndex tree:
{tree_as_prompt_text(tree)}
""".strip()

table_response = await call_llm(table_retrieval_prompt, table_user_prompt)
table_json = extract_json(table_response)
table_node_ids = table_json.get("node_list", [])

print(f"Selected {len(table_node_ids)} node(s):", table_node_ids)
print("\nRetrieval reasoning:")
print(table_json.get("thinking", ""))
```

#### B2b) Visualize the table retrieval

```python
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

table_titles = []
for nid in table_node_ids:
    node = node_map.get(nid, {})
    title = node.get('title', nid)
    page = node.get('page', '?')
    table_titles.append((nid, title, page))

T = nx.DiGraph()

tree_sections = [node.get('title', node_id) for node_id, node in list(node_map.items())[:12]]

T.add_node("Document\nTree", layer=0)

for section in tree_sections:
    T.add_node(section, layer=1)
    T.add_edge("Document\nTree", section)

T.add_node("LLM\nIdentifies\nTable", layer=2)
T.add_node("Retrieve\nFull Table", layer=3)
T.add_node("Answer", layer=4)

for section in tree_sections:
    T.add_edge(section, "LLM\nIdentifies\nTable")

T.add_edge("LLM\nIdentifies\nTable", "Retrieve\nFull Table")
T.add_edge("Retrieve\nFull Table", "Answer")

node_colors = []
node_sizes = []
selected_names = [x[1] for x in table_titles]
for node in T.nodes():
    if node in selected_names:
        node_colors.append("#E74C3C")
        node_sizes.append(3000)
    elif node == "Document\nTree":
        node_colors.append("#4A90D9")
        node_sizes.append(2500)
    elif node in ["LLM\nIdentifies\nTable", "Retrieve\nFull Table", "Answer"]:
        node_colors.append("#2ECC71")
        node_sizes.append(2500)
    else:
        node_colors.append("#BDC3C7")
        node_sizes.append(2000)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
pos = nx.multipartite_layout(T, subset_key="layer", align="horizontal")

nx.draw_networkx_nodes(T, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
nx.draw_networkx_edges(T, pos, edge_color="#7F8C8D", arrows=True, arrowsize=15, width=1.5, alpha=0.6, ax=ax)
nx.draw_networkx_labels(T, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)

legend_items = [
    mpatches.Patch(color="#4A90D9", label="Document Tree"),
    mpatches.Patch(color="#BDC3C7", label="Other Sections"),
    mpatches.Patch(color="#E74C3C", label="Selected Table Node"),
    mpatches.Patch(color="#2ECC71", label="Processing Steps"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=9, framealpha=0.9)

ax.set_title("Table Retrieval Graph — Full Node Preserved", fontsize=14, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()
plt.show()
```

#### B3) Extract table evidence

```python
table_evidence = collect_node_text(table_node_ids)
if not table_evidence.strip():
    raise RuntimeError("No table evidence was collected. Check the selected node IDs.")

print("Table evidence preview:\n")
print(preview_text(table_evidence, limit=4000))
```

#### B4) Generate table-based answer

```python
table_answer_prompt = """
You are a document analysis assistant answering questions from tables and structured data.

You have been given complete table data extracted from a document.
Answer the user's question by reading the table carefully.

Return valid JSON with this shape:
{
  "final_answer": "specific answer with exact values, dates, or figures from the table",
  "table_structure": {
    "table_title": "name or section of the table",
    "relevant_row": "the specific row or cell that answers the question",
    "relevant_column": "the specific column or category"
  },
  "explainability": {
    "evidence_used": ["specific table excerpt that was used"]
  }
}

Always include exact values and figures from the table. Do not approximate.
""".strip()

table_answer_user = f"""
Question:
{TABLE_QUESTION}

Table data:
{table_evidence}
""".strip()

table_final_response = await call_llm(table_answer_prompt, table_answer_user)
table_final_json = extract_json(table_final_response)

print("Final answer:")
print(table_final_json.get("final_answer", ""))
print("\nTable reference:")
print(json.dumps(table_final_json.get("table_structure", {}), indent=2))
print("\nExplainability:")
print(json.dumps(table_final_json.get("explainability", {}), indent=2))
```

#### B5) Groundedness check

```python
if table_evidence.strip():
    groundedness_system_prompt = """
You are a fact-checking assistant. You will be given an answer and the
evidence text it was supposed to be based on. Determine whether the
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
{table_final_json['final_answer']}

Evidence it should be based on:
{table_evidence}
""".strip()

    try:
        groundedness_json = await call_llm_and_parse(groundedness_system_prompt, groundedness_user_prompt)
        if groundedness_json.get("grounded") is False:
            print("\nWARNING: This answer may not be fully supported by the retrieved evidence.")
            print(f"   Reason: {groundedness_json.get('reason', 'No reason given.')}")
        else:
            print("\nGroundedness check passed: answer is supported by the retrieved evidence.")
    except Exception:
        print("\n(Groundedness check could not be completed.)")
```

# 10. Conclusion

In this lab, you extended Vectorless RAG to handle complex, real-world scenarios that break traditional chunking-based retrieval. Key takeaways:

- **Multi-hop retrieval** enables answering questions that require combining facts from distant sections of a document — something single-chunk retrieval cannot achieve. The LLM reasons over the full tree and explicitly selects every relevant node.
- **Table preservation** is a critical advantage of node-based retrieval. Tables, schedules, and structured data remain intact with their headers, rows, and footnotes — avoiding the fragmentation that chunking introduces.
- **Cross-reference analysis** provides users with a clear answer, supporting evidence from multiple sections, and full explainability, making the system suitable for audit and compliance workflows.
- **Visualization** of the retrieval graph (multi-hop and table paths) adds transparency by showing exactly which document sections were accessed and how evidence flows into the final answer.

To extend this pipeline, consider:
- Adding a pre-check step that validates input data against document constraints before answering.
- Supporting batch queries against the same document for multiple questions.
- Integrating OCR for scanned PDFs to improve table extraction accuracy.
- Building a comparison view that shows how chunking-based RAG would fail on the same queries.
