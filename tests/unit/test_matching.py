"""All matching tests run without network, database or LLM service."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.extraction.schemas import JobPosting
from src.matching.facts import extract_candidate_profile, experience_months, validate_facts
from src.matching.fixtures import ROOT, fixture_jobs, fixture_profile
from src.matching.requirements import extract_job_language_levels, prepare_job
from src.matching.retrieval import hybrid_search
from src.matching.schemas import (CandidateFacts, CandidateProfile, EmploymentPeriod,
                                  Evidence, JobLanguageEvidence, JobLanguageExtraction,
                                  LanguageFact, NarrativeClaim, NarrativeDraft, SkillFact)
from src.matching.scoring import rank_analyses, score_match
from src.matching.service import _grounded_claims, analyze_jobs
from src.rag.embeddings import HashEmbeddings
from src.rag.ingestion import read_cv, chunk_cv
from src.rag.service import CVService
from src.rag.store import MemoryVectorStore
from scripts.evaluate_job_matching import evaluate


@pytest.fixture
def fixture_env():
    service = CVService(HashEmbeddings(), MemoryVectorStore())
    chunks = service.ingest(ROOT / "sample_cv.md", "fixture")
    return service, chunks, fixture_profile("fixture", chunks)


def test_original_job_schema_is_preserved():
    job = JobPosting(role="Engineer", skills=["Python"], language=["German"])
    assert "language_requirements" not in job.model_dump()
    prepared = prepare_job("job-1", job, {"German": "B2"})
    assert prepared.languages[0].minimum_level == "B2"
    with pytest.raises(ValueError, match="absent"):
        prepare_job("job-1", job, {"English": "C1"})


def test_skill_and_language_evidence_and_explicit_unknown(fixture_env):
    _, _, profile = fixture_env
    backend, ml, robotics, cloud, senior, security = fixture_jobs()
    score, cases = score_match(backend, profile)
    assert score.value == 100 and score.retrieval_cosine is None
    assert all(c.status == "supported" for c in cases)
    _, cases = score_match(robotics, profile)
    assert [(c.requirement, c.status) for c in cases] == [
        ("ROS 2", "supported"), ("C++", "supported"), ("German B2", "unmet")]
    _, cases = score_match(cloud, profile)
    assert next(c for c in cases if c.requirement == "Terraform").status == "unknown"
    _, cases = score_match(senior, profile)
    assert next(c for c in cases if c.kind == "experience").status == "unknown"
    assert experience_months(profile) is None
    assert not any(c.status == "unmet" for c in cases if c.kind == "skill")


def test_language_without_level_is_unknown_not_below(fixture_env):
    _, _, profile = fixture_env
    prof = profile.model_copy(deep=True)
    prof.languages = [x for x in prof.languages if x.name != "German"]
    job = fixture_jobs()[2]
    _, cases = score_match(job, prof)
    assert cases[-1].status == "unknown"
    prof.languages.append(LanguageFact(name="German", evidence=profile.languages[1].evidence))
    _, cases = score_match(job, prof)
    assert cases[-1].status == "unknown"


def test_scoring_is_reproducible_and_rejects_cosine_outside_range(fixture_env):
    _, _, profile = fixture_env
    job = fixture_jobs()[0]
    first = score_match(job, profile, 0.5)
    assert all(score_match(job, profile, 0.5) == first for _ in range(20))
    assert 0 < first[0].value <= 100
    with pytest.raises(ValueError, match="cosine"):
        score_match(job, profile, 1.0001)


def test_no_double_count_overlapping_jobs(fixture_env):
    _, chunks, profile = fixture_env
    ev = Evidence(chunk_id=chunks[0].chunk_id, quote="2020-01 to 2021-12")
    # Synthetic month-specified evidence separate from the year-only reference CV.
    periods = [EmploymentPeriod(start="2020-01", end="2021-12", evidence=ev),
               EmploymentPeriod(start="2021-01", end="2022-12", evidence=ev)]
    prof = profile.model_copy(update={"employment": periods})
    assert experience_months(prof) == 36
    job = prepare_job("job", JobPosting(role="Job", experience_years=3))
    _, cases = score_match(job, prof)
    assert cases[0].status == "supported"
    assert score_match(prepare_job("job2", JobPosting(role="Job", experience_years=4)), prof)[1][0].status == "unmet"


def test_evidence_validation_rejects_invented_quote_date_and_proficiency(fixture_env):
    _, chunks, _ = fixture_env
    source = chunks[0]
    with pytest.raises(ValueError, match="unsupported"):
        validate_facts(CandidateFacts(skills=[SkillFact(name="Rust",
            evidence=Evidence(chunk_id=source.chunk_id, quote="Rust"))]), chunks)
    language_chunk = next(c for c in chunks if "German B1" in c.text)
    with pytest.raises(ValueError, match="proficiency"):
        validate_facts(CandidateFacts(languages=[LanguageFact(name="German", level="C2",
            evidence=Evidence(chunk_id=language_chunk.chunk_id, quote="German B1"))]), chunks)
    with pytest.raises(ValueError, match="dates"):
        validate_facts(CandidateFacts(employment=[EmploymentPeriod(start="2023-01", end="2025-12",
            evidence=Evidence(chunk_id=language_chunk.chunk_id, quote="German B1"))]), chunks)


def test_provider_profile_extraction_validates_sources(fixture_env):
    _, chunks, _ = fixture_env
    py_chunk = next(c for c in chunks if "Python for data" in c.text)
    good = CandidateFacts(skills=[SkillFact(name="Python", evidence=Evidence(
        chunk_id=py_chunk.chunk_id, quote="Python for data processing and model training."))])
    provider = SimpleNamespace(generate=lambda messages, schema:good)
    result = extract_candidate_profile(provider, "fixture", chunks)
    assert result.cv_id == "fixture" and len(result.skills) == 1
    bad = CandidateFacts(skills=[SkillFact(name="Python", evidence=Evidence(chunk_id="fake", quote="Python"))])
    with pytest.raises(ValueError, match="citation"):
        extract_candidate_profile(SimpleNamespace(generate=lambda m,s:bad), "fixture", chunks)
    with pytest.raises(ValueError, match="too many"):
        extract_candidate_profile(provider, "fixture", chunks, max_chunks=1)


def test_job_language_enrichment_requires_source_quote():
    job = JobPosting(role="Developer", language=["German"])
    good = JobLanguageExtraction(requirements=[JobLanguageEvidence(name="German", level="B2", quote="German B2")])
    provider = SimpleNamespace(generate=lambda messages, schema:good)
    assert extract_job_language_levels(provider, job, "Required German B2") == {"german": "B2"}
    with pytest.raises(ValueError, match="quoted"):
        extract_job_language_levels(SimpleNamespace(generate=lambda m,s:
            JobLanguageExtraction(requirements=[JobLanguageEvidence(name="German",level="C2",quote="German B2")])),
            job, "Required German B2")


def test_hybrid_section_filter_and_rerank(fixture_env):
    service, chunks, _ = fixture_env
    result = hybrid_search(service, "fixture", "ROS 2 LoRa", section="Robotics", k=3)
    assert result and all(x.chunk.section == "Robotics" for x in result)
    assert result == hybrid_search(service, "fixture", "ROS 2 LoRa", section="Robotics", k=3)
    assert hybrid_search(service, "fixture", "Python", section="nonexistent") == []
    assert all(x.chunk.section == "Robotics" for x in service.store.search(
        "fixture", service.embeddings.embed(["ROS"])[0], service.embeddings.model,
        service.embeddings.dimensions, 5, section="Robotics"))


def test_narrative_rejects_unapproved_changes(fixture_env):
    _, _, profile = fixture_env
    job = fixture_jobs()[2]
    _, cases = score_match(job, profile)
    bad = NarrativeDraft(claims=[NarrativeClaim(kind="language", requirement="German B2",
        status="supported", text="German is fluent", evidence=cases[-1].evidence)])
    with pytest.raises(ValueError, match="unsupported"):
        _grounded_claims(SimpleNamespace(generate=lambda messages, schema:bad), job, cases)
    bad2 = NarrativeDraft(claims=[NarrativeClaim(kind="language", requirement="German B2",
        status="unmet", text="It is below B2", evidence=[Evidence(chunk_id="bad",quote="German B1")])])
    with pytest.raises(ValueError, match="citation"):
        _grounded_claims(SimpleNamespace(generate=lambda messages, schema:bad2), job, cases)


def test_analysis_requires_matching_cv_id(fixture_env):
    service, _, profile = fixture_env
    bad = profile.model_copy(update={"cv_id": "different"})
    with pytest.raises(ValueError, match="ID"):
        analyze_jobs(fixture_jobs(), "fixture", service, profile=bad)
    result = analyze_jobs(fixture_jobs(), "fixture", service, profile=profile)
    assert len(result) == 6 and result[0].score.value >= result[-1].score.value
    assert rank_analyses(list(reversed(result))) == result
    with pytest.raises(ValueError, match="unique"):
        rank_analyses([result[0], result[0]])


def test_offline_report_regression():
    result = evaluate()
    assert result["job_cases"] == 6
    assert result["labeled_requirement_checks"] == 25
    assert result["requirement_status_accuracy"] == 1.0
    assert result["score_reproducibility"]
    assert result["retrieval"]["hybrid"]["hit_at_3"] >= result["retrieval"]["vector"]["hit_at_3"]


def test_candidate_extraction_corrects_invalid_skill_once(fixture_env):
    _, chunks, _ = fixture_env
    py_chunk = next(c for c in chunks if "Python for data" in c.text)
    from src.matching.schemas import CandidateFacts, Evidence, SkillFact

    invalid = CandidateFacts(skills=[SkillFact(
        name="Python programming", evidence=Evidence(
            chunk_id=py_chunk.chunk_id,
            quote="Python for data processing and model training.",
        ),
    )])
    valid = CandidateFacts(skills=[SkillFact(
        name="Python", evidence=Evidence(
            chunk_id=py_chunk.chunk_id,
            quote="Python for data processing and model training.",
        ),
    )])
    class CorrectingProvider:
        def __init__(self):
            self.calls = 0
        def generate(self, messages, schema):
            self.calls += 1
            return invalid if self.calls == 1 else valid

    provider = CorrectingProvider()
    profile = extract_candidate_profile(provider, "fixture", chunks)
    assert provider.calls == 2
    assert [skill.name for skill in profile.skills] == ["Python"]

    class AlwaysInvalid:
        def __init__(self):
            self.calls = 0
        def generate(self, messages, schema):
            self.calls += 1
            return invalid

    never = AlwaysInvalid()
    with pytest.raises(ValueError, match="Claimed skill name"):
        extract_candidate_profile(never, "fixture", chunks)
    assert never.calls == 2


def test_candidate_extraction_reanchors_skill_to_literal_chunk(fixture_env):
    _, chunks, _ = fixture_env
    from src.matching.schemas import CandidateFacts, Evidence, SkillFact
    original = chunks[0]
    assert "PostgreSQL" not in original.text
    invalid = CandidateFacts(skills=[SkillFact(
        name="PostgreSQL", evidence=Evidence(
            chunk_id=original.chunk_id, quote=original.text[:30]
        ),
    )])

    class Provider:
        def __init__(self):
            self.calls = 0
        def generate(self, messages, schema):
            self.calls += 1
            return invalid

    provider = Provider()
    profile = extract_candidate_profile(provider, "fixture", chunks)
    assert provider.calls == 1
    assert profile.skills[0].name == "PostgreSQL"
    assert "PostgreSQL" in profile.skills[0].evidence.quote
    validate_facts(CandidateFacts(skills=profile.skills), chunks)


def test_candidate_extraction_still_rejects_unsupported_skill(fixture_env):
    _, chunks, _ = fixture_env
    from src.matching.schemas import CandidateFacts, Evidence, SkillFact
    invalid = CandidateFacts(skills=[SkillFact(
        name="NotARealSkill123", evidence=Evidence(
            chunk_id=chunks[0].chunk_id, quote=chunks[0].text[:30]
        ),
    )])

    class Provider:
        def __init__(self):
            self.calls = 0
        def generate(self, messages, schema):
            self.calls += 1
            return invalid

    provider = Provider()
    with pytest.raises(ValueError, match="Claimed skill name"):
        extract_candidate_profile(provider, "fixture", chunks)
    assert provider.calls == 2


def test_skill_reanchoring_does_not_repair_fabricated_citations(fixture_env):
    _, chunks, _ = fixture_env
    from src.matching.schemas import CandidateFacts, Evidence, SkillFact
    py_chunk = next(c for c in chunks if "Python for data" in c.text)

    class Provider:
        def __init__(self, facts):
            self.facts = facts
            self.calls = 0
        def generate(self, messages, schema):
            self.calls += 1
            return self.facts

    for evidence in (
        Evidence(chunk_id="fabricated-chunk", quote="Python"),
        Evidence(chunk_id=py_chunk.chunk_id, quote="Imaginary quote not in source"),
    ):
        provider = Provider(CandidateFacts(skills=[SkillFact(name="Python", evidence=evidence)]))
        with pytest.raises(ValueError, match="unsupported source citation"):
            extract_candidate_profile(provider, "fixture", chunks)
        assert provider.calls == 2


def test_live_like_year_only_employment_does_not_block_profile(fixture_env, caplog):
    service, chunks, _ = fixture_env
    cv_id = "fixture"
    work = next(c for c in chunks if "Research assistant" in c.text)
    programming = next(c for c in chunks if "Python for data" in c.text)
    response = CandidateFacts(
        skills=[SkillFact(name="Python", evidence=Evidence(
            chunk_id=programming.chunk_id,
            quote="Python for data processing and model training."))],
        employment=[EmploymentPeriod(start="2023-01", end="2025-12", evidence=Evidence(
            chunk_id=work.chunk_id,
            quote="Research assistant at Example Laboratory from 2023 to 2025."))],
    )

    class Provider:
        calls = 0
        def generate(self, messages, schema):
            self.calls += 1
            return response

    provider = Provider()
    profile = extract_candidate_profile(provider, cv_id, chunks)
    assert provider.calls == 1  # Do not spend another API call on guessed month precision.
    assert [skill.name for skill in profile.skills] == ["Python"]
    assert profile.employment == []
    assert experience_months(profile) is None
    job = fixture_jobs()[4]
    _, comparisons = score_match(job, profile)
    assert next(c for c in comparisons if c.kind == "experience").status == "unknown"
    assert "Excluded 1 employment period" in caplog.text


def test_month_precise_employment_is_retained(fixture_env):
    _, chunks, _ = fixture_env
    chunk = chunks[0].model_copy(update={"text": "Employed from 2023-02 to 2025-01."})
    response = CandidateFacts(employment=[EmploymentPeriod(
        start="2023-02", end="2025-01", evidence=Evidence(
            chunk_id=chunk.chunk_id, quote="Employed from 2023-02 to 2025-01."))])
    provider = SimpleNamespace(generate=lambda messages, schema: response)
    profile = extract_candidate_profile(provider, "fixture", [chunk])
    assert len(profile.employment) == 1
    assert experience_months(profile) == 24


def test_inferred_employment_with_fabricated_citation_still_fails(fixture_env):
    _, chunks, _ = fixture_env
    year_only = next(c for c in chunks if "Research assistant" in c.text)
    quote = "Research assistant at Example Laboratory from 2023 to 2025."
    for evidence in (
        Evidence(chunk_id="fake-source-id", quote=quote),
        Evidence(chunk_id=year_only.chunk_id, quote="Unrelated invented employment."),
    ):
        response = CandidateFacts(employment=[EmploymentPeriod(
            start="2023-01", end="2025-12", evidence=evidence)])
        provider = SimpleNamespace(generate=lambda messages, schema: response)
        with pytest.raises(ValueError, match="unsupported source citation"):
            extract_candidate_profile(provider, "fixture", chunks)


def test_contextual_match_explanation(fixture_env):
    from src.matching.service import render_match_explanation
    from scripts.evaluate_matching_ragas import evaluation_question
    service, _, profile = fixture_env
    analyses = analyze_jobs(fixture_jobs(), "fixture", service, profile=profile)
    ml = next(a for a in analyses if a.job_id == "match-02-ml-engineer")
    summary = render_match_explanation(ml)
    assert summary == render_match_explanation(ml)
    assert "Machine Learning Engineer" in summary
    for item in ("Python", "PyTorch", "SQL", "German B1"):
        assert item in summary
    assert "None of the assessed requirements are unmet or unknown" in summary
    assert evaluation_question(ml) == (
        "Explain the candidate's evidenced fit, gaps and unknowns for Machine Learning Engineer."
    )

    senior = next(a for a in analyses if a.job_id == "match-05-senior-platform")
    summary = render_match_explanation(senior)
    assert "Explicitly unmet requirements: German C1" in summary
    assert "Requirements with insufficient CV evidence (unknown):" in summary
    assert "5 years" in summary and "Kubernetes" in summary
    assert "Unknown does not mean" in summary
    assert "None of the assessed requirements" not in summary
