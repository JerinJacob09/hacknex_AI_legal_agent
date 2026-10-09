# Deploying NYAYA-QUANTUM on Render

## The key limit
Render has **no GPU**. Its largest plans are CPU only (Pro Plus 8 GB RAM, Pro Max 16 GB; check render.com/pricing for current rates).
A 7B model such as `qwen2.5:7b` would be very slow on CPU and needs at least the 8 GB plan, so this setup puts only the
**Streamlit app, embeddings and Chroma** on Render. The **LLM runs elsewhere** and Render calls it.

## Choose where the LLM runs
| Mode | How | Pros | Cons |
|---|---|---|---|
| A. Your own Ollama (recommended for the jury story) | Run Ollama on your PC/server/GPU box and expose it through a secure tunnel (Cloudflare Tunnel with Access, or Tailscale). Set `NYAYA_OLLAMA_HOST`. | Keeps the "your own open-weight model" story, no per-call cost | Your machine must be on during the demo. An open Ollama port has no login: never expose it bare |
| B. Hosted API via LiteLLM | `NYAYA_LLM_BACKEND=litellm`, `NYAYA_LLM_MODEL=<litellm id>`, plus the provider's key env var | Always on, fast | Prompts and case text go to a third party, so the "100% local / confidential" claim no longer holds. The app says "hosted LLM mode" |
| C. Ollama on Render (Docker private service, 8 GB+ plan, persistent disk) | Separate service | Fully on Render | About $175/month or more, CPU-only so slow; not recommended for a demo |

In mode B the Legal-Saul drafting model is not used (the main model does drafting).

## Steps
1. Push this folder to a GitHub repo (`render.yaml` must be at the repo root).
2. Render dashboard: New > Blueprint > pick the repo. Render reads `render.yaml`.
3. When prompted, fill in the `sync: false` variables:
   - `NYAYA_APP_PASSWORD`: any strong password
   - Mode A: `NYAYA_OLLAMA_HOST`; Mode B: set `NYAYA_LLM_BACKEND=litellm`, `NYAYA_LLM_MODEL` and the provider key
   - `INDIANKANOON_TOKEN` (optional)
4. Deploy. The first build is slow (installs torch CPU), and the first start downloads the embedding model to the disk.
5. Open the URL, enter the password, check the sidebar shows LLM online and embeddings ready, upload 2 to 3 files, click "Embed into Chroma".

## Settings reference
| Variable | Default | Meaning |
|---|---|---|
| `NYAYA_PERSIST_DIR` | `./nyaya_chroma_store` | Where Chroma stores the index (set to the Render disk) |
| `HF_HOME` | Hugging Face default | Model cache (set to the disk so restarts do not re-download) |
| `NYAYA_EMBEDDING` | BGE large | Default embedding model. Use `BAAI/bge-small-en-v1.5` on 2 GB RAM |
| `NYAYA_LLM_BACKEND` | `ollama` | `ollama` or `litellm` |
| `NYAYA_OLLAMA_HOST` / `NYAYA_MODEL` | `http://localhost:11434` / `qwen2.5:7b` | Mode A settings |
| `NYAYA_LLM_MODEL` | empty | Mode B LiteLLM model id |
| `NYAYA_APP_PASSWORD` | empty (no gate) | Password screen for the whole app |

## Known limits of this setup
- Not tested on a live Render service from here; do a trial deploy before the jury.
- Persistent disks work on a single instance only (no scaling out) and need a paid plan.
- Reranker and legal NER are off by default on 2 GB RAM. BGE small is weaker than BGE large for retrieval, so re-run your eval numbers with the model you deploy.
- Render may restart or redeploy the service; the index survives on the disk but in-memory session state does not.
- Never put API keys or the password in the repo; use the dashboard variables.
