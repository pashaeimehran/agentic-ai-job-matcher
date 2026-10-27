"""Deterministic evidence-aware matching. Scores are compatibility heuristics."""
from __future__ import annotations

from dataclasses import dataclass

from src.matching.facts import experience_months
from src.matching.schemas import (CandidateProfile, ComponentScore, JobRequirements,
                                  MatchScore, RequirementResult)

# Only explicit, narrowly reviewed synonyms. No fuzzy matching or inferred skills.
_SKILL_ALIASES = {
    "postgres": "postgresql", "postgresql": "postgresql",
    "pgvector": "pgvector", "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn", "scikit-learn": "scikit-learn",
    "ros2": "ros 2", "ros 2": "ros 2", "c plus plus": "c++",
}
_LANGUAGE_ALIASES = {"deutsch": "german", "englisch": "english",
                     "turkisch": "turkish", "allemand": "german"}
_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2", "native")
_WEIGHTS = {"skills": 0.5, "experience": 0.2, "languages": 0.2, "retrieval": 0.1}


def skill_key(value: str) -> str:
    key = " ".join(value.casefold().split())
    return _SKILL_ALIASES.get(key, key)


def language_key(value: str) -> str:
    key = " ".join(value.casefold().split())
    return _LANGUAGE_ALIASES.get(key, key)


def score_match(job: JobRequirements, profile: CandidateProfile,
                retrieval_cosine: float | None = None) -> tuple[MatchScore, list[RequirementResult]]:
    if retrieval_cosine is not None and not -1 <= retrieval_cosine <= 1:
        raise ValueError("Retrieval cosine must lie in [-1, 1]")
    skill_lookup = {}
    for fact in profile.skills:
        skill_lookup.setdefault(skill_key(fact.name), fact.evidence)
    comparisons: list[RequirementResult] = []
    scores: dict[str, float] = {}
    seen_skills = set()
    for skill in job.skills:
        key = skill_key(skill)
        if key in seen_skills:
            continue
        seen_skills.add(key)
        evidence = skill_lookup.get(key)
        comparisons.append(RequirementResult(kind="skill", requirement=skill,
            status="supported" if evidence else "unknown",
            reason="Explicitly evidenced in CV" if evidence else "Not evidenced in CV; absence is not proof of inability",
            evidence=[evidence] if evidence else []))
    skill_cases = [r for r in comparisons if r.kind == "skill"]
    if skill_cases:
        scores["skills"] = sum(r.status == "supported" for r in skill_cases) / len(skill_cases)

    months = experience_months(profile)
    if job.experience_years is not None:
        if months is None:
            experience_status, reason = "unknown", "No precise, evidenced employment boundaries"
        elif months >= job.experience_years * 12:
            experience_status, reason = "supported", f"{months} explicitly dated employment months (overlaps counted once)"
        else:
            experience_status, reason = "unmet", f"{months} explicitly dated employment months, below {job.experience_years * 12} required"
        comparisons.append(RequirementResult(kind="experience", requirement=f"{job.experience_years} years",
            status=experience_status, reason=reason,
            evidence=[x.evidence for x in profile.employment] if months is not None else []))
        scores["experience"] = {"supported": 1.0, "unmet": 0.0, "unknown": 0.5}[experience_status]

    seen_languages = set()
    for req in job.languages:
        key = language_key(req.name)
        if key in seen_languages:
            continue
        seen_languages.add(key)
        known = [f for f in profile.languages if language_key(f.name) == key]
        specified = {f.level for f in known if f.level is not None}
        required = req.minimum_level
        evidence = [f.evidence for f in known]
        if not known:
            status, reason = "unknown", "Language not evidenced in CV"
        elif required is None:
            status, reason = "supported", "Language explicitly listed; job did not specify a level"
        elif len(specified) != 1:
            status, reason = "unknown", "No unambiguous explicitly evidenced proficiency level"
        elif _LEVELS.index(next(iter(specified))) >= _LEVELS.index(required):
            status, reason = "supported", f"Explicit proficiency meets {required}"
        else:
            status, reason = "unmet", f"Explicit proficiency below {required}"
        comparisons.append(RequirementResult(kind="language",
            requirement=f"{req.name} {required}" if required else req.name,
            status=status, reason=reason, evidence=evidence))
    language_cases = [r for r in comparisons if r.kind == "language"]
    if language_cases:
        scores["languages"] = sum({"supported": 1.0, "unmet": 0.0, "unknown": 0.5}[r.status]
                                  for r in language_cases) / len(language_cases)

    # No free-floating semantic-only score for postings without explicit requirements.
    if comparisons and retrieval_cosine is not None:
        scores["retrieval"] = (retrieval_cosine + 1) / 2
    active_weight = sum(_WEIGHTS[part] for part in scores)
    components = []
    for name in ("skills", "experience", "languages", "retrieval"):
        if name in scores:
            weight = _WEIGHTS[name] / active_weight
            components.append(ComponentScore(name=name, raw=round(scores[name], 6),
                weight=round(weight, 6), weighted_contribution=round(scores[name] * weight * 100, 6)))
    value = round(sum(scores[name] * _WEIGHTS[name] for name in scores) / active_weight * 100, 2) if scores else 0.0
    known = sum(r.status != "unknown" for r in comparisons)
    return MatchScore(value=value, components=components,
        evidenced_requirements=known, total_requirements=len(comparisons),
        retrieval_cosine=retrieval_cosine), comparisons


def rank_analyses(analyses):
    """Stable, deterministic secondary sort by unique job ID."""
    ids = [a.job_id for a in analyses]
    if len(ids) != len(set(ids)):
        raise ValueError("Ranking requires unique job IDs")
    return sorted(analyses, key=lambda a: (-a.score.value, a.job_id))
