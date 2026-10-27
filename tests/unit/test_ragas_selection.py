"""Offline regression checks for limited paid evaluation and evidence contexts."""
import json
from types import SimpleNamespace

from src.matching.fixtures import ROOT, fixture_jobs, fixture_profile
from src.matching.schemas import NarrativeDraft, NarrativeClaim
from src.rag.embeddings import HashEmbeddings
from src.rag.service import CVService
from src.rag.store import MemoryVectorStore
from scripts.evaluate_matching_ragas import select_ranked_matches, evidence_contexts


def _fixture():
    service = CVService(HashEmbeddings(), MemoryVectorStore())
    chunks = service.ingest(ROOT / "sample_cv.md", "synthetic")
    return service, chunks, fixture_profile("synthetic", chunks)


def test_limit_one_generates_only_one_llm_explanation():
    service, chunks, profile = _fixture()
    calls = []

    def generate(messages, schema):
        assert schema is NarrativeDraft
        calls.append(messages)
        comparisons = json.loads(messages[1]["content"])["comparisons"]
        return NarrativeDraft(claims=[NarrativeClaim(
            kind=r["kind"], requirement=r["requirement"], status=r["status"],
            text=f"Verified {r['requirement']}", evidence=r["evidence"]
        ) for r in comparisons])

    provider = SimpleNamespace(generate=generate)
    selected = select_ranked_matches(fixture_jobs(), "synthetic", service,
                                     profile, provider, limit=1)
    assert len(selected) == 1
    assert selected[0].job_id == "match-02-ml-engineer"
    assert len(calls) == 1
    contexts = evidence_contexts(selected[0], {c.chunk_id: c for c in chunks})
    assert contexts
    for comp in selected[0].comparisons:
        for ev in comp.evidence:
            assert any(ev.quote in text for text in contexts)


def test_evidence_contexts_reject_unknown_chunk_id():
    import pytest
    from src.matching.service import analyze_jobs
    service, chunks, profile = _fixture()
    analysis = analyze_jobs(fixture_jobs(), "synthetic", service, profile=profile)[0]
    invalid = analysis.model_copy(update={"retrieved_chunk_ids": ["fake-chunk-id"]})
    with pytest.raises(ValueError, match="missing source chunk"):
        evidence_contexts(invalid, {c.chunk_id: c for c in chunks})


def test_labeled_fixture_profile_is_evidence_validated_without_llm():
    from scripts.evaluate_matching_ragas import load_evaluation_profile
    from src.matching.fixtures import fixture_profile
    from src.matching.schemas import CandidateFacts
    from src.matching.facts import validate_facts
    service, chunks, original_profile = _fixture()

    class NeverCall:
        def generate(self, *args, **kwargs):
            raise AssertionError("Fixture-backed evaluation must not call extraction LLM")

    profile = load_evaluation_profile("labeled-fixture", NeverCall(), "synthetic", chunks)
    assert profile == fixture_profile("synthetic", chunks)
    validate_facts(CandidateFacts.model_validate(profile.model_dump(exclude={"cv_id"})), chunks)
    assert profile.employment == []


def test_labeled_fixture_profile_rejects_nonidentical_stored_cv():
    import pytest
    from scripts.evaluate_matching_ragas import load_evaluation_profile
    _, chunks, _ = _fixture()
    modified = chunks[:]
    modified[0] = modified[0].model_copy(update={"text": modified[0].text + " changed"})
    with pytest.raises(ValueError, match="exact stored sample CV"):
        load_evaluation_profile("labeled-fixture", None, "synthetic", modified)


def test_extracted_profile_mode_remains_available():
    from scripts.evaluate_matching_ragas import load_evaluation_profile
    from src.matching.schemas import CandidateFacts
    _, chunks, _ = _fixture()
    calls = []

    def generate(messages, schema):
        calls.append(schema)
        return CandidateFacts()

    provider = SimpleNamespace(generate=generate)
    profile = load_evaluation_profile("extracted", provider, "synthetic", chunks)
    assert profile.cv_id == "synthetic"
    assert len(calls) >= 1
    assert not profile.skills and not profile.languages and not profile.employment
