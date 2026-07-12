# 1. Lab Title

## Vectorless RAG: Multi-Hop Claims Adjudication & Benefit Tables

# What is Multi-Hop RAG?

**Multi-hop RAG** is a retrieval-augmented generation pattern where answering a single question requires fetching evidence from **multiple distant sections** of a document.

Traditional chunking-based RAG retrieves a single text chunk per query. This works for simple lookups but fails when the answer depends on combining facts from different parts of a document — for example, matching an incident date against a policy's effective period (page 2) and deductible limits (page 15).

**Vectorless RAG** solves this by letting an LLM reason over the **entire document tree** and explicitly select any combination of nodes, regardless of where they appear. This enables true multi-hop reasoning without the noise and fragmentation introduced by text chunking.

# 2. Problem Statement / Use Case Overview

This lab demonstrates two advanced Vectorless RAG scenarios where standard chunking-based RAG fails:

### Scenario A: Multi-Hop Claims Adjudication

When adjudicating an insurance claim, adjusters must cross-reference multiple policy sections:
- **Effective Dates** — when does coverage apply?
- **Deductible Limits** — what does the insured pay first?
- **Coverage Caps** — what is the maximum payout?

Standard RAG retrieves a single chunk. If these facts are in different sections, chunking either misses one or mixes in irrelevant surrounding text.

### Scenario B: Benefit Tables & Rate Charts

Insurance policies contain structured tables — rate charts, benefit schedules, co-pay matrices — that are critical for accurate answers. Standard chunking **shatters** tables:
- A table spanning one page might be split into 3-4 chunks.
- Headers get separated from their rows.
- Footnotes referencing table cells get lost.

Vectorless RAG retrieves **whole logical nodes**, preserving table structure, headers, and footnotes intact.

# 3. Input Data

| Item | Detail |
|------|--------|
| PDF document | A complex insurance policy with multi-section clauses, dates, deductibles, and rate tables |
| Adjudication question | A multi-hop question requiring facts from multiple sections (e.g., _"Is this claim within the effective dates and below the deductible?"_) |
| Table question | A question about a rate chart or benefit table (e.g., _"What is the coinsurance rate for out-of-network providers?"_) |
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
    ├─── Scenario A: Multi-Hop ──────────────────────────────────────
    │         │
    │         ▼
    │    LLM (Groq) ──── select multiple nodes ────►  [dates, deductibles, caps]
    │         │
    │         ▼
    │    Evidence Extractor ──── collect all nodes ────►  combined evidence
    │         │
    │         ▼
    │    LLM (Groq) ──── adjudicate claim ────►  decision + explainability
    │
    └─── Scenario B: Table Extraction ───────────────────────────────
              │
              ▼
         LLM (Groq) ──── identify table node ────►  [rate_chart_node]
              │
              ▼
         Evidence Extractor ──── collect full table ────►  structured table
              │
              ▼
         LLM (Groq) ──── read table + answer ────►  specific rate/benefit answer
```

1. **PageIndex Client** submits the PDF (or loads a cached tree) and retrieves a hierarchical tree of titles, summaries, and text.
2. **Scenario A**: The LLM selects **multiple nodes** from different sections (effective dates, deductibles, coverage caps) and combines them for adjudication.
3. **Scenario B**: The LLM identifies the **table node** and retrieves it intact — preserving headers, rows, and footnotes.
4. **Answer LLM** uses only the retrieved evidence to produce a structured JSON response.

# 5. Output

### Scenario A: Adjudication Response

```json
{
  "decision": "approve",
  "final_answer": "The claim is within the effective dates and below the deductible limit.",
  "explainability": {
    "effective_dates": "Policy period: Jan 1, 2024 – Dec 31, 2024",
    "deductible": "Annual deductible: $1,000",
    "coverage_limits": "Maximum payout: $500,000",
    "evidence_used": ["node xyz: Effective Dates section", "node abc: Deductible Limits section"]
  }
}
```

### Scenario B: Table-Based Response

```json
{
  "final_answer": "The coinsurance rate for out-of-network providers is 40%.",
  "table_structure": {
    "table_title": "Section 5.2 — Provider Fee Schedule",
    "relevant_row": "Out-of-Network",
    "relevant_column": "Coinsurance Rate"
  },
  "explainability": {
    "evidence_used": ["Complete fee schedule table from node 123"]
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
| Language | Python 3.x |
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
- A complex insurance policy PDF to place in the `data/` folder.
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


def preview_text(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."
```

---

### Build the Vectorless RAG Pipeline

This is the core of the lab. Follow each sub-step in order.

#### 1) Load a PDF from `data/`

Place a complex insurance policy PDF in the `data/` folder before running this cell. If `PDF_NAME` is set, the notebook uses that file; otherwise it picks the first PDF found.

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

### Scenario A: Multi-Hop Claims Adjudication

#### A1) Define the adjudication question

Ask a question that requires fetching facts from multiple sections of the policy.

```python
ADJUDICATION_QUESTION = input(
    "Enter a multi-hop claims question (e.g., 'Is this claim within the effective dates and below the deductible limit?'): "
).strip()
if not ADJUDICATION_QUESTION:
    raise ValueError("A question is required to continue.")
```

#### A2) Multi-hop tree search

The LLM receives the full document tree and explicitly identifies which sections to fetch.

```python
retrieval_system_prompt = """
You are a document retrieval assistant for a multi-hop insurance claims adjudication system.

You will receive a user question and a PageIndex tree made of node titles and summaries.
The question may require information from MULTIPLE distant sections of the policy.

Identify ALL node IDs that contain relevant evidence. Common patterns:
- Effective Dates / Policy Period
- Deductible Limits / Waiting Periods
- Coverage Caps / Sublimits
- Exclusions / Conditions
- Definition sections that clarify terms used elsewhere

Return valid JSON with this shape:
{
  "thinking": "step-by-step reasoning about which sections are needed and why",
  "node_list": ["node_id_1", "node_id_2", ...]
}

Do not output markdown, prose, or extra keys.
Be thorough — missing a required node means an incomplete adjudication.
""".strip()

retrieval_user_prompt = f"""
Question:
{ADJUDICATION_QUESTION}

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

#### A4) Generate adjudication answer

```python
adjudication_system_prompt = """
You are an insurance claims adjudication assistant.

You have been given evidence from multiple sections of an insurance policy.
Analyze the evidence to answer the adjudication question.

Return valid JSON with this shape:
{
  "decision": "approve" or "deny" or "needs_review",
  "final_answer": "clear, concise answer for the claims adjuster",
  "explainability": {
    "effective_dates": "policy period information found",
    "deductible": "deductible information found",
    "coverage_limits": "coverage cap information found",
    "evidence_used": ["short evidence note 1", "short evidence note 2"]
  }
}

Ground every statement in the provided evidence. Do not invent facts.
""".strip()

adjudication_user_prompt = f"""
Adjudication question:
{ADJUDICATION_QUESTION}

Evidence from policy:
{evidence_text}
""".strip()

adjudication_response = await call_llm(adjudication_system_prompt, adjudication_user_prompt)
adjudication_json = extract_json(adjudication_response)

print("Decision:", adjudication_json.get("decision", ""))
print("\nFinal answer:")
print(adjudication_json.get("final_answer", ""))
print("\nExplainability:")
print(json.dumps(adjudication_json.get("explainability", {}), indent=2))
```

---

### Scenario B: Benefit Tables & Rate Charts

#### B1) Define the table query

```python
TABLE_QUESTION = input(
    "Enter a table-related question (e.g., 'What is the coinsurance rate for out-of-network providers?'): "
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
- Rate charts / premium tables
- Benefit schedules / co-pay matrices
- Coverage limit tables
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

T = nx.DiGraph()

tree_sections = [
    "1 DECLARATIONS",
    "2 DEFINITIONS",
    "3 INSURING AGREEMENTS",
    "4 PROPERTY DAMAGE",
    "5 GENERAL LIABILITY",
    "6 CYBER & DATA BREACH",
    "7 GENERAL EXCLUSIONS",
    "8 CONDITIONS",
    "9 CLAIMS PROCEDURE",
    "10 Schedule of Limits",
    "11 ENDORSEMENTS",
    "12 Cross-Reference Index",
]

selected_titles_map = {x[1]: x for x in table_titles}

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
for node in T.nodes():
    if node in [x[1] for x in table_titles]:
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
You are an insurance benefits assistant answering questions from rate charts and tables.

You have been given complete table data extracted from a policy document.
Answer the user's question by reading the table carefully.

Return valid JSON with this shape:
{
  "final_answer": "specific answer with exact numbers, rates, or limits from the table",
  "table_structure": {
    "table_title": "name or section of the table",
    "relevant_row": "the specific row or cell that answers the question",
    "relevant_column": "the specific column or category"
  },
  "explainability": {
    "evidence_used": ["specific table excerpt that was used"]
  }
}

Always include exact numbers and rates from the table. Do not approximate.
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
