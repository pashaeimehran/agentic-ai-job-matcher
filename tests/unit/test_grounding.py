"""Deterministic unknown-vs-unmet answer check and the version-comparison harness (no network)."""
import pytest

from scripts.compare_versions import check_labels, fisher_exact_two_sided, load_cases
from src.agent.grounding import mentioned, mischaracterization_evidence, mischaracterized_unknowns
from src.agent.prompts import PROMPTS, SYSTEM_PROMPT, SYSTEM_PROMPT_V1, system_prompt

UNKNOWN = ["Kubernetes", "Terraform", "5 years", "C"]


@pytest.mark.parametrize("answer", [
    "You are lacking Kubernetes experience.",
    "For the Cloud Platform Engineer role, gaps include skills in Terraform and Kubernetes.",
    "Unmet: Kubernetes, Terraform.",
    "You don't have Kubernetes, which is a weakness.",
    "### Missing skills\n- Kubernetes\n- Terraform",
])
def test_deficit_wording_on_unknown_requirements_is_flagged(answer):
    assert mischaracterized_unknowns(answer, UNKNOWN)


@pytest.mark.parametrize("answer", [
    "Kubernetes is unknown: your CV contains no evidence either way.",
    "Your CV lacks evidence of Kubernetes and Terraform.",
    "### Unknown requirements\n- Kubernetes (not mentioned in your CV)\n- 5 years",
    "### Weaknesses\n- Kubernetes: no evidence in the CV.",
    "Supported: Python. Unmet: German C1. Unknown: Kubernetes.",
    "You are missing German C1, while Kubernetes is unknown.",
    "Consider learning C++ and Rust.",
    # Real live answers that an earlier detector revision flagged wrongly (manual audit):
    "You meet the Python requirement, but the language requirement is unmet, and there are unknowns regarding "
    "Kubernetes and your total years of experience.",
    "However, details about your experience with Terraform are missing from your CV.",
    "**Kubernetes**: Your CV lacks information related to your experience with Kubernetes.",
    "You have a solid foundation but lack evidentiary support for several skills, including Terraform.",
    "### Summary of Gaps:\n- The 5 years of experience required were not clearly documented in your CV.",
])
def test_qualified_or_unrelated_statements_are_not_flagged(answer):
    assert mischaracterized_unknowns(answer, UNKNOWN) == []


def test_heading_context_and_evidence_strings():
    evidence = mischaracterization_evidence("### Gaps\n- Terraform\n### Unknown\n- Kubernetes", UNKNOWN)
    assert evidence == [("Terraform", "### Gaps > - Terraform")]


def test_single_letter_requirement_matches_whole_word_only():
    assert mischaracterized_unknowns("You lack C experience.", ["C"]) == ["C"]
    assert mischaracterized_unknowns("You lack C++ experience.", ["C"]) == []
    assert mentioned("Kubernetes and 5 years", UNKNOWN) == ["5 years", "Kubernetes"]


def test_grounding_labels_match_deterministic_matching():
    assert check_labels(load_cases()) == []


def test_prompt_versions():
    assert set(PROMPTS) == {"v1", "v2"} and SYSTEM_PROMPT == SYSTEM_PROMPT_V1
    assert "lacking, missing, a gap" in system_prompt("v2") and "lacking, missing" not in system_prompt("v1")
    with pytest.raises(ValueError):
        system_prompt("v9")


def test_fisher_exact():
    assert fisher_exact_two_sided(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact_two_sided(0, 5, 0, 5) == 1.0
