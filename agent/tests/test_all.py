import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import contradictions as cd
import entities as ent
import eval_harness as eh
import grounding as gr
import indiankanoon as ik
import retrieval as rt
import rfae
from fakes import FakeStore

AGREEMENT = (
    "INDEMNITY AND SERVICES AGREEMENT\n\nThis agreement was executed on 12 March 2023 between Alpha Traders and Beta Logistics. "
    "Beta Logistics shall indemnify Alpha Traders up to Rs. 5,00,000. Stamp duty of Rs. 100 was paid on the instrument. "
    "Any dispute shall be referred to arbitration seated at Chennai."
)
PLAINT = (
    "Alpha Traders states that the agreement was executed on 21 April 2023. Beta Logistics shall indemnify Alpha Traders up to Rs. 3,00,000. "
    "Stamp duty was not paid on the instrument by Alpha Traders. A legal notice of breach was sent on 14.04.2023 under Section 138."
)
AFFIDAVIT = "Stamp duty was paid on the instrument by Alpha Traders at the time of execution. The parties signed at Chennai."


def build_store():
    store = FakeStore()
    for name, text in (("agreement.txt", AGREEMENT), ("plaint.txt", PLAINT), ("affidavit.txt", AFFIDAVIT)):
        pages, err = rt.read_pages(name, text.encode())
        assert err is None
        rt.index_chunks(store, rt.make_chunks(pages, name, "case1"))
    # a document belonging to another case must never leak into case1 results
    pages, _ = rt.read_pages("other.txt", b"Gamma Industries paid Rs. 9,99,999 on 01 January 2020 under a different contract.")
    rt.index_chunks(store, rt.make_chunks(pages, "other.txt", "case2"))
    return store


class RfaeDuplicateTests(unittest.TestCase):
    def test_user_example_is_merged(self):
        rule = [{"item": "Dates of agreement, breach, cause of action and any notice", "why": "limitation", "severity": "critical", "source": "rule"}]
        agent = [{"item": "Date of breach and notice", "why": "", "severity": "important"}]
        merged = rfae.merge_rfae(rule, agent)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "rule")

    def test_unrelated_items_stay_separate(self):
        a = "Valuation of the suit or subject matter and amounts claimed"
        for b in ("Place of cause of action, party residences and places of performance",
                  "Stamping details: stamp duty paid, denomination, state and date",
                  "Date of the alleged offence or proceeding"):
            self.assertFalse(rfae.is_duplicate(a, b), b)

    def test_paraphrases_merge(self):
        pairs = [("Stamping details: stamp duty paid, denomination, state and date", "Stamp duty details"),
                 ("Arbitration seat and venue, and the full arbitration clause", "Seat of arbitration"),
                 ("Full identity, capacity and addresses of all parties", "Party addresses"),
                 ("Valuation of the suit or subject matter and amounts claimed", "Suit valuation and claim amount")]
        for a, b in pairs:
            self.assertTrue(rfae.is_duplicate(a, b), (a, b))

    def test_higher_severity_wins_and_new_item_kept(self):
        rule = [{"item": "Stamping details: stamp duty paid, denomination, state and date", "why": "", "severity": "important", "source": "rule"}]
        agent = [{"item": "Stamp duty details", "why": "inadmissible if unstamped", "severity": "critical"},
                 {"item": "Board resolution authorising the signatory", "why": "authority", "severity": "minor"}]
        merged = rfae.merge_rfae(rule, agent)
        self.assertEqual(len(merged), 2)
        stamp = next(m for m in merged if "tamp" in m["item"])
        self.assertEqual(stamp["severity"], "critical")
        self.assertEqual(stamp["why"], "inadmissible if unstamped")

    def test_embedding_only_confirms_borderline_lexical_matches(self):
        always_same = lambda texts: [[1.0, 0.0], [1.0, 0.0]]
        self.assertFalse(rfae.is_duplicate("Stamp duty paid", "Arbitration clause seat venue", always_same))  # no shared word: never merged
        self.assertTrue(rfae.is_duplicate("Service of legal notice on respondent", "Respondent legal notice proof of delivery postal receipt", always_same))
        orthogonal = lambda texts: [[1.0, 0.0], [0.0, 1.0]]
        self.assertFalse(rfae.is_duplicate("Service of legal notice on respondent", "Respondent legal notice proof of delivery postal receipt", orthogonal))

    def test_rules_still_fire_on_sample_draft(self):
        draft = "Any dispute shall be settled by arbitration in a place to be decided later. This agreement is valid even if it is not stamped under the Indian Penal Code."
        items = rfae.rule_based_rfae(draft)
        self.assertTrue(any("seat" in i["item"].lower() for i in items))
        self.assertTrue(any(i["severity"] == "critical" for i in items))
        self.assertTrue(rfae.readiness(items)["status"].startswith("Not ready"))


class EntityTests(unittest.TestCase):
    def test_dates_and_values_normalised(self):
        dates, _ = ent.extract_dates("on 12th March, 2023 and 14.04.2023 and April 2024")
        self.assertEqual(dates, {"2023-03-12", "2023-04-14", "2024-04"})
        self.assertEqual(ent.extract_values("Rs. 5,00,000"), ent.extract_values("5 lakh"))

    def test_sentence_split_keeps_tags_and_abbreviations(self):
        parts = ent.split_sentences("State v. Kumar was cited. [S1] The sum was Rs. 5 lakh [S2]. Next sentence here.")
        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[0].endswith("[S1]."))


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [gr.Chunk(id="c-1", text=AGREEMENT, doc="agreement.txt", loc="¶1-2"), gr.Chunk(id="c-2", text=PLAINT, doc="plaint.txt", loc="¶1-1")]
        self.sources = gr.number_sources(self.chunks)

    def test_supported_and_fabricated_claims(self):
        answer = (
            "The agreement was executed on 12 March 2023 between Alpha Traders and Beta Logistics [S1].\n"
            "Beta Logistics must indemnify Alpha Traders up to Rs. 5 lakh [S1].\n"
            "Beta Logistics must indemnify Alpha Traders up to Rs. 7,00,000 [S1].\n"
            "A notice of breach was sent on 20 April 2023 [S2].\n"
            "Section 420 was invoked in the notice [S2].\n"
            "Gamma Traders acted as guarantor [S1].\n"
            "The court relied on Arnesh Kumar v. State of Bihar (2014) 8 SCC 273 [S2].\n"
            "The deposit was Rs. 3,00,000 and nothing else is said here without a tag."
        )
        result = gr.ground_answer(answer, self.sources, mode="drop")
        self.assertEqual(result.supported, 2)
        self.assertEqual(result.unsupported, 6)
        self.assertIn("12 March 2023", result.text)
        for bad in ("7,00,000", "20 April", "420", "Gamma", "Arnesh", "3,00,000"):
            self.assertNotIn(bad, result.text)

    def test_citation_must_come_from_cited_chunk(self):
        sources = gr.number_sources([gr.Chunk(id="c-9", text="The Court applied Arnesh Kumar v. State of Bihar (2014) 8 SCC 273 to the arrest.", doc="judgment.txt")])
        good = gr.ground_answer("The Court applied Arnesh Kumar v. State of Bihar (2014) 8 SCC 273 to the arrest [S1].", sources)
        self.assertEqual((good.supported, good.unsupported), (1, 0))
        wrong = gr.ground_answer("The Court applied Arnesh Kumar v. State of Bihar (2014) 8 SCC 999 to the arrest [S1].", sources)
        self.assertEqual(wrong.supported, 0)

    def test_nonexistent_tag_and_all_withheld(self):
        result = gr.ground_answer("The agreement was executed on 12 March 2023 [S7].", self.sources)
        self.assertEqual(result.supported, 0)
        self.assertTrue(result.text.startswith(gr.NOT_FOUND))

    def test_flag_mode_keeps_text_with_marker(self):
        result = gr.ground_answer("Beta Logistics must pay Rs. 9,00,000 under the agreement [S1].", self.sources, mode="flag")
        self.assertIn("unsupported", result.text)

    def test_fabricated_authorities(self):
        ref = "Reliance was placed on Arnesh Kumar v. State of Bihar (2014) 8 SCC 273."
        bad = gr.fabricated_authorities("See Arnesh Kumar v. State of Bihar (2014) 8 SCC 273 and AIR 1999 SC 1.", ref)
        self.assertEqual(bad, ["AIR 1999 SC 1"])


class RetrievalTests(unittest.TestCase):
    def test_pages_and_metadata(self):
        pages = [(1, "p.1", "First page text about the agreement."), (2, "p.2", "Second page about stamp duty.")]
        chunks = rt.make_chunks(pages, "doc.pdf", "case1")
        self.assertEqual([c.page for c in chunks], [1, 2])
        self.assertEqual(chunks[1].label(), "doc.pdf, p.2")
        self.assertEqual(len({c.id for c in chunks}), 2)

    def test_case_isolation_and_hybrid(self):
        store = build_store()
        hits = rt.hybrid_search(store, "stamp duty paid instrument", k=3, case_ids=["case1"])
        self.assertTrue(hits)
        self.assertTrue(all(h.case_id == "case1" for h in hits))
        self.assertFalse(any("Gamma" in h.text for h in hits))

    def test_bm25_finds_rare_term_dense_misses(self):
        store = build_store()
        bm = rt._bm25_for(store, ["case1"])
        top = bm.search("arbitration seated Chennai", 1)
        self.assertEqual(top[0].doc, "agreement.txt")

    def test_recall(self):
        store = build_store()
        hits = rt.hybrid_search(store, "security deposit", k=3, case_ids=["case1"])
        self.assertEqual(rt.recall_at_k(hits, [{"text": "Rs. 3,00,000"}]), 1.0)
        self.assertEqual(rt.recall_at_k(hits, [{"doc": "nothing.pdf"}]), 0.0)


class ContradictionTests(unittest.TestCase):
    def test_finds_date_amount_and_negation_conflicts(self):
        chunks = rt.fetch_all(build_store(), ["case1"])
        kinds = {c["kind"] for c in cd.find_candidates(chunks)}
        self.assertTrue({"date", "number", "negation"} <= kinds, kinds)

    def test_no_false_positive_on_identical_facts(self):
        chunks = [gr.Chunk(id="a", text="Delivery shall be completed within 30 days of the order.", doc="x"),
                  gr.Chunk(id="b", text="The order requires delivery to be completed within 30 days.", doc="y")]
        self.assertEqual(cd.find_candidates(chunks), [])

    def test_llm_verification_filters(self):
        chunks = rt.fetch_all(build_store(), ["case1"])
        cands = cd.find_candidates(chunks)
        rejected = cd.verify_candidates(cands, lambda p: '{"contradiction": false, "explanation": "different things"}')
        self.assertEqual(cd.report(rejected), [])
        self.assertEqual(len(cd.report(rejected, include_rejected=True)), len(cands))


class IndianKanoonTests(unittest.TestCase):
    def test_parsing_with_fake_transport(self):
        client = ik.IndianKanoon("tok")
        client._post = lambda path, params=None: ({"docs": [{"tid": 1, "title": "<b>A v. B</b>", "docsource": "Supreme Court", "headline": "x &amp; y"}]} if path == "/search/"
                                                  else {"title": "A v. B", "doc": "<p>Held: yes.</p><p>Second.</p>", "citation": ""})
        self.assertEqual(client.search("q")[0]["title"], "A v. B")
        self.assertIn("Held: yes.", client.fetch(1)["text"])


class EvalHarnessTests(unittest.TestCase):
    def test_baseline_vs_pipeline_on_fakes(self):
        store = build_store()

        def fake_llm(system, user):
            if system == eh.BASELINE_SYSTEM:  # sloppy baseline: right fact plus an invented citation and amount
                return "The agreement was executed on 12 March 2023. The indemnity is capped at Rs. 8,00,000. See Foo v. Bar (2001) 3 SCC 5."
            return "The agreement was executed on 12 March 2023 [S1]. The indemnity is capped at Rs. 8,00,000 [S1]."

        systems = {"baseline": eh.baseline_system(store, fake_llm, ["case1"]), "pipeline": eh.pipeline_system(store, fake_llm, ["case1"])}
        questions = [{"q": "When was the agreement executed and what is the indemnity cap?", "gold": [{"doc": "agreement.txt"}]}]
        summary, rows = eh.evaluate(questions, systems)
        self.assertGreater(summary["pipeline"]["claims_supported_pct"], summary["baseline"]["claims_supported_pct"])
        self.assertEqual(summary["pipeline"]["fabricated_citations"], 0)
        self.assertGreaterEqual(summary["baseline"]["fabricated_citations"], 1)
        self.assertIn("claims_supported_pct", eh.format_table(summary))


if __name__ == "__main__":
    unittest.main(verbosity=2)
