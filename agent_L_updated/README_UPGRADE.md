# NYAYA-QUANTUM upgrade notes

Models are unchanged (main model, Legal-Saul drafting model, the same embedding/reranker choices, OpenNyAI NER).
No new pip dependencies: all new modules are plain Python.

## New / changed files
| File | What it does |
|---|---|
| `rfae.py` | `rule_based_rfae` + `merge_rfae` moved here. Duplicate gaps are detected by keyword overlap (stemmed content words), with an optional embedding tie-break for BGE. |
| `grounding.py` | Claim-level check: each sentence must carry `[S#]` tags; its dates, amounts, sections, case names, citations and party names must appear in the cited chunk. Unsupported sentences are withheld. |
| `retrieval.py` | Page-aware chunking (PDF page numbers, paragraph ranges for txt/docx), per-case index, BM25 + dense hybrid (RRF), optional rerank, recall@k. |
| `contradictions.py` | Cross-document contradiction candidates (date / amount / negation) with both quotes; optional confirmation by the local model. |
| `entities.py`, `textutil.py` | Date/amount/provision/citation/name extraction and light stemming. |
| `indiankanoon.py` | Optional Indian Kanoon API client (needs a token; untested against the live service). |
| `eval_harness.py` | Baseline vs pipeline ablation: claims supported %, fabricated citations, abstentions, recall@k. |
| `legal_ai.py`, `nyaya_quantum_app.py` | Wired to the above. New tabs: **Case Chat (grounded)** and **Case Review**. |

## Behaviour changes
* The hard-coded "verified citations" list and the landmark index notes are off (`legal_ai.REFERENCE_LIST_ENABLED = False`).
  A citation is accepted only if its text occurs in a retrieved passage.
* The three fictional seed cases are gone, and old ones are deleted from an existing Chroma store on startup.
* The Scholar tab is kept but labelled UNGROUNDED, and all its citations are marked unverified.
* Retrieval is scoped to the active case (plus the real-judgment library for drafting, if ticked).
  Re-index your files once: old chunks have no document/page metadata.

## Run
    # same dependencies as before (the install line shown in the app sidebar)
    streamlit run nyaya_quantum_app.py

    python tests/test_all.py          # 22 unit tests, no models needed
    python tests/smoke_app.py         # runs the whole app script against stubs

    python eval_harness.py --case-dir ./my_case --questions eval/questions.example.json --out results.json \
        --model qwen2.5:7b --embedding BAAI/bge-large-en-v1.5 --reranker BAAI/bge-reranker-base [--llm-judge]
