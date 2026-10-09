"""Executes nyaya_quantum_app.py top-to-bottom with stub streamlit/langchain/smolagents modules and a fake LLM.
It verifies the wiring (imports, names, indexing -> retrieval -> chat -> review), not model quality."""
import io
import json
import os
import runpy
import sys
import types
import urllib.request
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from fakes import FakeDoc, FakeStore  # noqa: E402

AGREEMENT = ("INDEMNITY AND SERVICES AGREEMENT\n\nThis agreement was executed on 12 March 2023 between Alpha Traders and Beta Logistics. "
             "Beta Logistics shall indemnify Alpha Traders up to Rs. 5,00,000. Any dispute shall be referred to arbitration.")
PLAINT = ("Alpha Traders states that the agreement was executed on 21 April 2023. Beta Logistics shall indemnify Alpha Traders up to Rs. 3,00,000. "
          "Stamp duty was not paid on the instrument.")


class Uploaded:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


STATE = {"run": 0}
session = {}


class SessionState(dict):
    def __getattr__(self, k):
        return self[k]


def build_streamlit(run):
    st = MagicMock(name="streamlit")
    st.session_state = SessionState(session)
    st.cache_resource = lambda *a, **k: (lambda f: f) if not (a and callable(a[0])) else a[0]
    st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    st.text_input.side_effect = lambda label, value="", **k: value
    st.text_area.side_effect = lambda label, **k: ""
    st.number_input.side_effect = lambda label, value=1, **k: value
    st.selectbox.side_effect = lambda label, options, index=0, **k: list(options)[index]
    st.checkbox.side_effect = lambda label, value=False, key=None, **k: True if key in ("cr_verify", "nq_use_library") else value
    st.slider.return_value = 2
    st.file_uploader.side_effect = lambda label, **k: ([Uploaded("agreement.txt", AGREEMENT.encode()), Uploaded("plaint.txt", PLAINT.encode())] if k.get("key") == "kb_upload" else None)
    wanted = {1: "Embed into Chroma", 2: "Run case review", 4: "Execute MultiAgent Legal Transformation"}.get(run)
    st.button.side_effect = lambda label, **k: label == wanted
    st.chat_input.side_effect = lambda *a, key=None, **k: ("When was the agreement executed?" if (run == 3 and key == "case_chat_input") else None)
    st.status.return_value.__enter__.return_value = MagicMock()
    return st


class FakeEmb:
    def embed_query(self, t):
        return [0.0]

    def embed_documents(self, ts):
        return [[1.0, 0.0] for _ in ts]


class FakeChroma(FakeStore):
    def __init__(self, collection_name=None, embedding_function=None, persist_directory=None):
        super().__init__()


class FakeModel:
    def __init__(self, *a, **k):
        pass

    def __call__(self, messages):
        system = messages[0]["content"][0]["text"]
        user = messages[1]["content"][0]["text"]
        if "ONLY the numbered sources" in system:
            return types.SimpleNamespace(content="The agreement was executed on 12 March 2023 [S1]. The agreement was also signed in Paris on 5 June 2022 [S1].")
        if "List the critical facts missing" in user:
            return types.SimpleNamespace(content='{"missing": [{"item": "Date of breach and notice", "why": "limitation", "severity": "critical"}, {"item": "Board resolution authorising the signatory", "why": "authority", "severity": "minor"}]}')
        if "PRECEDENT BRIEF for the draft" in user:
            return types.SimpleNamespace(content="Relevant authority: Foo v. Bar (2001) 3 SCC 5 supports the claim.")
        if "fact checker" in system:
            return types.SimpleNamespace(content='{"contradiction": true, "explanation": "The two documents give different execution dates."}')
        return types.SimpleNamespace(content="")


def install_stubs(run):
    mods = {
        "streamlit": build_streamlit(run),
        "smolagents": types.SimpleNamespace(ToolCallingAgent=lambda **k: None, LiteLLMModel=FakeModel, Tool=type("Tool", (), {"__init__": lambda self: None})),
        "langchain_chroma": types.SimpleNamespace(Chroma=FakeChroma),
        "langchain_huggingface": types.SimpleNamespace(HuggingFaceEmbeddings=lambda **k: FakeEmb()),
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.documents": types.SimpleNamespace(Document=FakeDoc),
        "langchain_text_splitters": types.SimpleNamespace(RecursiveCharacterTextSplitter=object),
    }
    sys.modules.update(mods)
    return mods["streamlit"]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    urllib.request.urlopen = lambda *a, **k: FakeResponse(json.dumps({"models": [{"name": "qwen2.5:7b"}, {"name": "hf.co/mradermacher/Legal-Saul-Multiverse-7b-GGUF:Q4_K_M"}]}).encode())
    # one shared store across reruns, like st.cache_resource
    shared = FakeChroma()
    FakeChroma.__init__ = lambda self, *a, **k: self.__dict__.update(shared.__dict__)
    # pre-seed what an older version would have left behind: it must be purged
    shared.add_documents([FakeDoc("FICTIONAL TRAINING CASE", {"kind": "seed", "source": "mock"}), FakeDoc("REFERENCE INDEX ENTRY", {"kind": "landmark", "source": "x"})], ids=["seed-mock-001", "landmark-x"])
    session["draft_text"] = "This agreement is between Alpha Traders and Beta Logistics. Any dispute shall be settled by arbitration in a place to be decided later. Beta shall indemnify Alpha."
    for run in (0, 1, 2, 3, 4):
        for name in [m for m in sys.modules if m in ("nyaya_quantum_app", "legal_ai")]:
            del sys.modules[name]
        st = install_stubs(run)
        runpy.run_path(os.path.join(HERE, "..", "nyaya_quantum_app.py"), run_name="__main__")
        session.update(dict(st.session_state))
        print(f"run {run} executed cleanly")
    ids = shared.ids
    assert "seed-mock-001" not in ids and "landmark-x" not in ids, "stale reference entries were not purged"
    assert any(i.startswith("c-") for i in ids), "case chunks were not indexed"
    chat = session["case_chat"]
    answer = chat[-1]
    print("CHAT ANSWER :", answer["content"])
    print("STATS       :", answer["stats"], "| withheld:", [(w[0][:50], w[1]) for w in answer["withheld"]])
    print("SOURCES     :", [(t, v["label"]) for t, v in answer["sources"].items()])
    review = session["case_review"]
    print("REVIEW GAPS :", len(review["gaps"]), "| CONTRADICTIONS:", [(c["kind"], c["detail"], c["confirmed"]) for c in review["candidates"]])
    result = session["result"]
    items = [r["item"] for r in result["rfae"]]
    print("PIPELINE RFAE:", [(r["item"][:48], r["severity"], r["source"]) for r in result["rfae"]])
    print("PIPELINE WARN:", [w[:110] for w in result["warnings"]])
    print("READINESS    :", result["readiness"]["status"])
    assert sum("breach" in i.lower() and "notice" in i.lower() for i in items) == 1, "duplicate RFAE gap was not merged"
    assert any("Board resolution" in i for i in items), "genuinely new LLM gap should be kept"
    assert any("Foo v. Bar" in w or "3 SCC 5" in w for w in result["warnings"]), "invented authority was not flagged"
    assert "12 March 2023" in answer["content"] and "Paris" not in answer["content"]
    assert review["candidates"], "expected the date/amount conflict between the two documents"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
