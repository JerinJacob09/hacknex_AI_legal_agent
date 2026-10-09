# ⚖️ NYAYA-QUANTUM

A local, multi-agent assistant for Indian legal work. It helps drafters and advocates review a draft, find missing facts and evidence, chat with their case files with checked citations, and spot contradictions across documents.

Everything runs on your own machine: the language models (through Ollama), the embeddings and the vector store. The only optional online parts are the first-time model downloads and the Indian Kanoon API.

> **Disclaimer:** this is a drafting and research aid, not legal advice. A qualified advocate must verify every output before it is used.

---

## Features

The app is a Streamlit page with six tabs:

| Tab | What it does |
|---|---|
| 🧬 **Document Comparison Mutation Room** | Upload a raw or flawed draft. The agents analyse it, rewrite it, and a critic agent reviews the rewrite (up to 3 rounds). Shows a before/after comparison. |
| ⚔️ **Litigation Win-Strategy Room** | Builds a litigation strategy from the draft and the retrieved case material. |
| 🎓 **Scholar Sandbox (ungrounded)** | Free legal Q&A with no retrieval. Clearly labelled **UNGROUNDED**; every citation in it is marked unverified. |
| 📋 **RFAE Checklist (Missing Logs)** | RFAE = *Request for Additional Evidence*. Lists the facts and documents missing from the draft (dates, valuation, stamp duty, jurisdiction, parties, arbitration seat, and so on), each with a reason and a severity (critical, important, minor). Downloadable as a text file. |
| 💬 **Case Chat (grounded)** | Ask questions about your uploaded case files. Every answer sentence must carry a source tag such as `[S2]`, and unsupported sentences are withheld. |
| 🔍 **Case Review** | Finds contradictions between documents (dates, amounts, negations) and shows both quotes. |

### How hallucinations are controlled
- **Claim-level grounding:** each sentence must cite a retrieved passage. Its dates, amounts, sections, case names, citations and party names must appear in that passage, otherwise the sentence is withheld.
- **Citations are accepted only if the text appears in a retrieved passage.** The old hard-coded "verified citations" list and the fictional seed cases were removed.
- **Missing facts are not invented.** The agents flag them as RFAE items instead.
- If the answer is not in your documents, the app says so instead of guessing.

---

## Models

| Role | Model | Runs on |
|---|---|---|
| Main agent LLM (analyst, critic, chat) | `qwen2.5:7b` | Ollama |
| Drafting LLM (optional) | Legal-Saul-Multiverse-7b (GGUF, Q4_K_M) | Ollama. Falls back to the main model if not installed |
| Embeddings (selectable in the sidebar) | `BAAI/bge-large-en-v1.5` (default), `law-ai/InLegalBERT`, `law-ai/InCaseLawBERT`, `nlpaueb/legal-bert-base-uncased` | Local (sentence-transformers) |
| Reranker (optional) | `BAAI/bge-reranker-base` or `bge-reranker-large` | Local |
| Legal NER (optional) | OpenNyAI `en_legal_ner_trf` | Local (spaCy) |

Agent framework: [smolagents](https://github.com/huggingface/smolagents) with LiteLLM, Supervisor-Worker design.

---

## Architecture

```
Uploaded files (.txt .md .pdf .docx)
        │
        ▼
 Page-aware chunking ──► Chroma vector store (one store per embedding model)
        │                         │
        │            BM25 + dense search, merged by reciprocal rank fusion
        │                         │         (optional reranker)
        ▼                         ▼
   Supervisor ──► Analyst agent (with retrieval tool)
        │         Scholar agent (no tools)
        │         Critic agent (JSON verdict: approved / issues)
        ▼
 Grounding check ──► RFAE gap list ──► Contradiction check ──► UI
```

- Chunks are about 900 characters with 120 overlap, and keep PDF page numbers or paragraph ranges.
- Retrieval is scoped to the active case (plus the real-judgment library for drafting, if ticked).
- The drafting prompt is limited to the first 8000 characters of a document.

---

## Requirements

- Python 3.10 or newer (recommended)
- [Ollama](https://ollama.com) installed and running
- Enough RAM for a 7B model (about 8 GB or more; a GPU makes it much faster)
- Disk space for the models (several GB)

---

## Installation

1. **Install the Python packages**

   ```bash
   pip install streamlit smolagents[litellm] langchain-chroma chromadb langchain-huggingface \
       sentence-transformers langchain-core langchain-text-splitters pypdf python-docx datasets spacy
   ```

2. **Pull the models in Ollama**

   ```bash
   ollama pull qwen2.5:7b
   ollama pull hf.co/mradermacher/Legal-Saul-Multiverse-7b-GGUF:Q4_K_M   # optional drafting model
   ```

   If that quant tag is not available, open the model page on Hugging Face and use the tag it lists.

3. **Optional: legal NER**

   ```bash
   pip install spacy
   pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl
   ```

   Check the OpenNyAI GitHub or Hugging Face page if this URL has changed.

4. **Optional: Indian Kanoon**

   Get an API token and either paste it in the sidebar or set it before starting:

   ```bash
   export INDIANKANOON_TOKEN=your_token      # Windows PowerShell: $env:INDIANKANOON_TOKEN="your_token"
   ```

---

## Run

Make sure Ollama is running, then:

```bash
streamlit run nyaya_quantum_app.py
```

The sidebar shows whether Ollama, the model, the embeddings and Chroma are ready. The embedding model is downloaded from Hugging Face the first time you use it.

### Typical workflow
1. In the sidebar, upload your case files and click **Embed into Chroma**. Start with 2 to 3 files.
2. Open the **Mutation Room**, upload a draft and run the analysis.
3. Read the **RFAE Checklist** to see what is missing, and the **Strategy Room** for the plan.
4. Use **Case Chat** to ask questions with checked citations, and **Case Review** to find contradictions.

> If you indexed files with an older version, re-index them once. Old chunks have no document or page metadata.

### Sidebar settings
- **Ollama host / model:** default `http://localhost:11434` and `qwen2.5:7b`
- **Max critique rounds:** 1 to 3 (default 2)
- **Embedding model, reranker, legal drafting LLM, legal NER**
- **Real-judgment library:** bulk-ingest judgments from a folder, a Hugging Face dataset, or Indian Kanoon. Embedding thousands of judgments on CPU is slow, so start with 20 to 50.

---

## Project structure

| File | Purpose |
|---|---|
| `nyaya_quantum_app.py` | Streamlit app, agents, tabs |
| `legal_ai.py` | Model choices, sidebar settings, ingestion, helpers |
| `retrieval.py` | Chunking, BM25 + dense hybrid search (RRF), optional rerank, recall@k |
| `grounding.py` | Claim-level source check |
| `rfae.py` | Missing-evidence rules and merging/deduplication of gap lists |
| `contradictions.py` | Cross-document contradiction candidates |
| `entities.py`, `textutil.py` | Date, amount, section, citation and name extraction; light stemming |
| `indiankanoon.py` | Optional Indian Kanoon API client |
| `eval_harness.py` | Evaluation: baseline versus full pipeline |
| `tests/` | Unit tests and a smoke test |
| `eval/questions.example.json` | Example evaluation questions |

---

## Tests

```bash
python tests/test_all.py      # 22 unit tests, no models needed
python tests/smoke_app.py     # runs the whole app script against stubs
```

---

## Evaluation

`eval_harness.py` compares a baseline (same local LLM, plain dense retrieval, no checks) with the full pipeline.

```bash
python eval_harness.py --case-dir ./my_case --questions eval/questions.example.json --out results.json \
    --model qwen2.5:7b --embedding BAAI/bge-large-en-v1.5 --reranker BAAI/bge-reranker-base [--llm-judge]
```

It reports, per system: share of answer sentences supported by a retrieved chunk, fabricated-citation count, abstention rate, and retrieval recall@k (dense-only versus hybrid).

Questions file format (JSON list):

```json
[
  {"q": "When was the agreement executed?", "gold": [{"doc": "agreement.pdf", "page": 1}]},
  {"q": "What deposit was paid?",           "gold": [{"text": "Rs. 5,00,000"}]}
]
```

**Caveat:** "supported" is measured with the same checker the pipeline uses as a gate, so it favours the pipeline by construction. Use `--llm-judge` to add an independent yes/no check by the local model. The example file has only 3 questions; write a larger set for real results.

---

## Known limitations
- 7B local models can still make reasoning mistakes. The grounding check verifies facts and citations against sources, not the quality of the legal reasoning.
- Scanned PDFs are not supported (no OCR). Convert them to searchable text first.
- Drafting uses the first 8000 characters of a document.
- English text only.
- The Legal-Saul model is trained mostly on non-Indian law, so check its drafts against Indian statutes. The IPC to BNS, CrPC to BNSS and Evidence Act to BSA changes need a human check.
- The Indian Kanoon client is untested against the live service.
- The Scholar Sandbox is ungrounded and its citations are unverified.
- Speed depends on your hardware; CPU-only runs are slow.

---

## Troubleshooting
- **"Ollama is not reachable":** start Ollama (`ollama serve`) and check the host in the sidebar.
- **"Model is not installed":** run `ollama pull <model name>` shown in the message.
- **Embedding or Chroma warnings:** install the missing packages shown in the sidebar, then restart.
- **Empty or poor answers after changing the embedding model:** each embedding model has its own store, so click **Embed into Chroma** again.
- **PDF or DOCX not read:** `pip install pypdf python-docx`.
