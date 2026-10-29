"""Deterministic answer check: unknown requirements described as deficits.

calculate_match separates "unmet" (CV evidence shows the requirement is not
met) from "unknown" (the CV has no evidence either way). An answer that calls
an unknown requirement "lacking", "missing", "a gap" or "unmet" turns missing
evidence into a negative claim about the person.

The check is lexical and conservative: a sentence (or the list section it sits
in) that names an unknown requirement is flagged when it contains deficit
wording and no evidence qualifier ("unknown", "evidence", "not mentioned", ...).
It is an evaluation aid with known limits (paraphrases, negations across
sentences); it is not a semantic judge. Revision history: the qualifier list
was extended after a manual audit of flagged live answers (plural "unknowns",
adverbs such as "not clearly documented", "missing from your CV", "evidentiary");
stored reports are re-judged with scripts.compare_versions --rescore.
"""
from __future__ import annotations

import re
from typing import Iterable

from src.agent.schemas import ToolCallRecord

DEFICIT = re.compile(
    r"\b(lack(?:s|ing|ed)?|missing|miss(?:es)?|gaps?|weakness(?:es)?|unmet|not\s+met|shortfalls?|"
    r"fall(?:s)?\s+short|deficien\w*|not\s+qualified|do(?:es)?\s*n[o']t\s+(?:have|possess|meet)|"
    r"no\s+experience|need(?:s)?\s+to\s+(?:learn|acquire|gain|develop)|"
    r"areas?\s+(?:to|for)\s+(?:develop|improve)\w*)\b", re.I)
QUALIFIER = re.compile(
    r"\b(unknowns?|evidence[ds]?|eviden(?:t|tly|tiary)|mention(?:s|ed)?|unverified|unclear|"
    r"not\s+(?:\w+ly\s+)?(?:shown|stated|listed|found|documented|demonstrated|specified|present|indicated|"
    r"confirmed|clear|included)|"
    r"can(?:not|'t)\s+(?:confirm|verify|be\s+confirmed|be\s+verified)|"
    r"(?:in|from|on)\s+(?:your|the)\s+(?:cv|resume)|"
    r"(?:your|the)\s+(?:cv|resume)\s+(?:lacks|is\s+missing|does\s*n[o']t|has\s+no))\b", re.I)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")


def _is_heading(line: str) -> bool:
    text = _LIST_MARKER.sub("", line.strip()).strip()
    if line.lstrip().startswith("#"):
        return True
    bare = text.strip("*_ ").strip()
    # "**Unmet requirements:**", "Gaps:" or a bold-only label line.
    return bool(bare) and len(bare) <= 80 and (bare.endswith(":") or (text.startswith("**") and text.endswith("**")))


def unknown_requirements(records: Iterable[ToolCallRecord]) -> dict[str, set[str]]:
    """job_id -> requirements calculate_match reported as unknown in this run."""
    found: dict[str, set[str]] = {}
    for record in records:
        if record.tool_name == "calculate_match" and record.outcome.data:
            for match in record.outcome.data["matches"]:
                found.setdefault(match["job_id"], set()).update(match["unknown"])
    return found


def mischaracterization_evidence(answer: str, unknown: Iterable[str]) -> list[tuple[str, str]]:
    """(requirement, offending sentence or "heading > sentence") pairs, in answer order."""
    names = sorted({u for u in unknown if u.strip()}, key=len, reverse=True)
    patterns = {name: re.compile(rf"(?<![\w+#-]){re.escape(name)}(?![\w+#-])", re.I) for name in names}
    heading = ""
    evidence: list[tuple[str, str]] = []
    for line in answer.splitlines():
        if not line.strip():
            continue
        if _is_heading(line):
            heading = line
        heading_deficit = bool(DEFICIT.search(heading)) and not QUALIFIER.search(heading)
        for sentence in _SENTENCE.split(line):
            for name, pattern in patterns.items():
                if not pattern.search(sentence) or QUALIFIER.search(sentence):
                    continue
                if DEFICIT.search(sentence):
                    evidence.append((name, sentence.strip()))
                elif heading_deficit and heading is not line:
                    evidence.append((name, f"{heading.strip()} > {sentence.strip()}"))
    return evidence


def mischaracterized_unknowns(answer: str, unknown: Iterable[str]) -> list[str]:
    """Unknown requirements that the answer presents as deficits (sorted, unique)."""
    return sorted({name for name, _ in mischaracterization_evidence(answer, unknown)})


def mentioned(answer: str, names: Iterable[str]) -> list[str]:
    return sorted({n for n in names if re.search(rf"(?<![\w+#-]){re.escape(n)}(?![\w+#-])", answer, re.I)})
