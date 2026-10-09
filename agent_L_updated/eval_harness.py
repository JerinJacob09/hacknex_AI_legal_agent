"""Ablation harness: baseline (same local LLM, plain dense retrieval, no checks) vs the full pipeline
(hybrid BM25+dense retrieval, optional rerank, source-tagged answers, claim-level grounding gate).

Reports per system: share of answer sentences supported by a retrieved chunk, fabricated-citation count,
abstention rate, and retrieval recall@k (dense-only vs hybrid). Models are NOT changed: it calls the same Ollama
model names the app uses.

Questions file (JSON list):
  [{"q": "When was the agreement executed?", "gold": [{"doc": "agreement.pdf", "page": 1}]},
   {"q": "What deposit was paid?",           "gold": [{"text": "Rs. 5,00,000"}]}]

Usage:
  python eval_harness.py --case-dir ./my_case --questions questions.json --out results.json \\
      [--model qwen2.5:7b] [--host http://localhost:11434] [--embedding BAAI/bge-large-en-v1.5] [--reranker BAAI/bge-reranker-base]

Caveat printed in the report: 'supported' is measured with the same deterministic checker the pipeline uses as a
gate, so it favours the pipeline by construction. Use --llm-judge to add an independent yes/no check by the local model."""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import grounding as gr
import retrieval as rt

GROUNDED_SYSTEM = (
    "You answer questions about a legal case file using ONLY the numbered sources provided.\n"
    "Rules: (1) End every sentence with the tag of the source it relies on, e.g. [S2] or [S1][S3]. "
    "(2) Use only names, dates, amounts, section numbers and case citations that appear in the sources. "
    "(3) Do not use outside knowledge and never invent a citation. "
    f"(4) If the sources do not answer the question, reply exactly: {gr.NOT_FOUND}"
)
BASELINE_SYSTEM = "You are a legal assistant. Answer the question using the context. Be concise."


def ollama_chat(host, model, system, user, timeout=300):
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 8192},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    request = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())["message"]["content"].strip()


def baseline_system(store, llm, case_ids, k=4):
    def run(question):
        chunks = rt.dense_search(store, question, k, case_ids)
        context = "\n\n".join(c.text for c in chunks)
        return {"chunks": chunks, "raw": llm(BASELINE_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION: {question}"), "final": None}
    return run


def pipeline_system(store, llm, case_ids, reranker=None, k=6):
    def run(question):
        chunks = rt.hybrid_search(store, question, k=k, fetch_k=20, case_ids=case_ids, reranker=reranker)
        sources = gr.number_sources(chunks)
        raw = llm(GROUNDED_SYSTEM, f"SOURCES:\n{gr.format_sources(sources)}\n\nQUESTION: {question}")
        grounded = gr.ground_answer(raw, sources, mode="drop")
        return {"chunks": chunks, "raw": raw, "final": grounded.text, "grounded": grounded}
    return run


def evaluate(questions, systems, k=6, judge=None):
    """systems: {name: callable(question)->{'chunks','raw','final'}}. Returns (summary, per_question_rows)."""
    rows, totals = [], {name: {"claims": 0, "supported": 0, "fabricated": 0, "abstain": 0, "recall": [], "judge_yes": 0, "judge_n": 0} for name in systems}
    for item in questions:
        for name, run in systems.items():
            out = run(item["q"])
            text = out["final"] if out.get("final") is not None else out["raw"]
            ref = "\n".join(c.text for c in out["chunks"])
            scored = gr.score_text(text if text != gr.NOT_FOUND else "", out["chunks"])
            fabricated = gr.fabricated_authorities(text, ref)
            recall = rt.recall_at_k(out["chunks"], item.get("gold"), k)
            t = totals[name]
            t["claims"] += scored["total"]
            t["supported"] += scored["supported"]
            t["fabricated"] += len(fabricated)
            t["abstain"] += 1 if text.startswith(gr.NOT_FOUND.rstrip(".")) else 0
            if recall is not None:
                t["recall"].append(recall)
            if judge is not None and scored["total"]:
                verdict = judge(item["q"], text, ref)
                if verdict is not None:
                    t["judge_n"] += 1
                    t["judge_yes"] += 1 if verdict else 0
            rows.append({"system": name, "question": item["q"], "answer": text, "raw_answer": out["raw"],
                         "claims": scored["total"], "supported": scored["supported"], "unsupported": scored["unsupported_sentences"],
                         "fabricated_authorities": fabricated, "recall_at_k": recall,
                         "retrieved": [f"{c.label()} [{c.id}]" for c in out["chunks"]]})
    summary = {}
    for name, t in totals.items():
        summary[name] = {
            "questions": len(questions),
            "claims": t["claims"],
            "claims_supported_pct": round(100 * t["supported"] / t["claims"], 1) if t["claims"] else None,
            "fabricated_citations": t["fabricated"],
            "abstained": t["abstain"],
            "recall_at_k": round(sum(t["recall"]) / len(t["recall"]), 3) if t["recall"] else None,
            "judge_supported_pct": round(100 * t["judge_yes"] / t["judge_n"], 1) if t["judge_n"] else None,
        }
    return summary, rows


def format_table(summary):
    cols = ["claims", "claims_supported_pct", "fabricated_citations", "abstained", "recall_at_k", "judge_supported_pct"]
    names = list(summary)
    lines = ["metric".ljust(24) + "".join(n.ljust(22) for n in names)]
    for col in cols:
        lines.append(col.ljust(24) + "".join(str(summary[n][col]).ljust(22) for n in names))
    lines.append("")
    lines.append("Note: 'claims_supported_pct' uses the pipeline's own deterministic checker, so it favours the pipeline by construction.")
    return "\n".join(lines)


def _build_real(args):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=args.embedding, encode_kwargs={"normalize_embeddings": True})
    store = Chroma(collection_name="eval_case", embedding_function=embeddings)
    reranker = None
    if args.reranker:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker, max_length=512)
    new = 0
    for path in sorted(Path(args.case_dir).rglob("*")):
        if path.suffix.lower() in (".txt", ".md", ".pdf", ".docx"):
            pages, error = rt.read_pages(path.name, path.read_bytes())
            if error:
                print("skip:", error, file=sys.stderr)
                continue
            added, _ = rt.index_chunks(store, rt.make_chunks(pages, path.name, "eval"))
            new += added
    print(f"indexed {new} chunks", file=sys.stderr)
    return store, reranker, embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--embedding", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--reranker", default="")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--llm-judge", action="store_true")
    args = parser.parse_args(argv)
    store, reranker, _ = _build_real(args)
    llm = lambda system, user: ollama_chat(args.host, args.model, system, user)
    judge = None
    if args.llm_judge:
        def judge(question, answer, context):
            reply = llm("You are a strict fact checker. Reply with only YES or NO.",
                        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nIs every factual statement in the ANSWER supported by the CONTEXT?")
            return reply.strip().upper().startswith("YES")
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    systems = {"baseline": baseline_system(store, llm, ["eval"], k=4), "pipeline": pipeline_system(store, llm, ["eval"], reranker, k=args.k)}
    summary, rows = evaluate(questions, systems, k=args.k, judge=judge)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_table(summary))


if __name__ == "__main__":
    main()
