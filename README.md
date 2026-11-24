# Employee Policy Query Agent (GENAI-AGENT-009)

RAG-based HR assistant: grounded answers with `[Document, Page]` citations, multi-turn memory,
human escalation with a GenAI 3-line handoff summary, Streamlit chat UI, role-based access, admin analytics.

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your OPENAI_API_KEY
python data_generator.py                              # creates /data (3 PDFs + tickets CSV)
python ingest.py                                      # chunk + embed + store in ChromaDB
streamlit run app.py
python evaluate.py                                    # 50-query evaluation -> /output
```
Demo logins: `employee/employee123`, `hr/hr123`, `admin/admin123`.

## Architecture
```mermaid
flowchart LR
  U[User] --> UI[Streamlit UI + RBAC]
  UI --> A[PolicyAgent]
  A --> G{Guards: empty input, prompt injection, sensitive keywords}
  G -->|sensitive| E[Escalation + GenAI summary]
  G --> R[MMR Retriever fetch_k=10 k=4]
  R --> C[(ChromaDB hr_policy_kb)]
  R --> L[LLM temp 0.1, strict context + citations]
  L -->|I_DONT_KNOW / conf < 0.6 / HR request| E
  L --> UI
  A --> LOG[(audit_log.csv, genai_log.csv, escalation_log.csv)]
  LOG --> D[Admin dashboard - Plotly]
```

## Files
| File | Purpose |
|---|---|
| `data_generator.py` | Builds sample PDFs (40/10/15 pages) and 1000-row tickets CSV |
| `ingest.py` | pdfplumber/pandas loaders, chunking (800/150), embeddings, ChromaDB |
| `rag.py` | RAG chain, window memory (k=5), escalation, summary, caching, error handling |
| `app.py` | Streamlit UI: chat, feedback, escalations, admin analytics, uploads |
| `evaluate.py` | 30 in-scope + 10 out-of-scope + 10 escalation queries |

## Evaluation results
Run `python evaluate.py` and paste the contents of `output/eval_metrics.json` here.
