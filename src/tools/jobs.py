"""Job catalog and the search_jobs / get_job_details tools.

The catalog reuses the existing structured fixtures; it does not scrape or
call an LLM. A BA Jobsuche-backed catalog would need per-posting extraction
and can implement the same JobCatalog protocol later.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from src.data_sources.fixtures import load_fixture_jobs
from src.extraction.schemas import JobPosting
from src.matching.requirements import prepare_job
from src.matching.schemas import JobRequirements
from src.tools.errors import ToolNotFoundError
from src.tools.schemas import (GetJobDetailsArgs, JobDetailsResult, JobSummary,
                               SearchJobsArgs, SearchJobsResult)

if TYPE_CHECKING:
    from src.tools.context import ToolContext

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "fixtures"
_STOPWORDS = {"a", "an", "and", "the", "for", "in", "of", "job", "jobs", "role", "roles",
              "position", "positions", "with", "to", "or"}
_COUNTRY_ALIASES = {"germany": "germany", "deutschland": "germany", "de": "germany"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w+#.]+", text.casefold())) - _STOPWORDS


@dataclass(frozen=True)
class CatalogJob:
    job_id: str
    source: str
    posting: JobPosting
    country: str | None = None
    language_levels: dict[str, str] = field(default_factory=dict)

    def requirements(self) -> JobRequirements:
        return prepare_job(self.job_id, self.posting, self.language_levels or None)


class JobCatalog(Protocol):
    def all(self) -> list[CatalogJob]: ...
    def get(self, job_id: str) -> CatalogJob | None: ...


class FixtureJobCatalog:
    """Structured postings from the committed fixtures (all German postings).

    jobs.json holds several phrasing variants of each posting for extraction
    benchmarking; identical structured postings are collapsed to the first ID.
    """

    def __init__(self, jobs: list[CatalogJob] | None = None) -> None:
        self._jobs = jobs if jobs is not None else self._load()
        self._by_id = {job.job_id: job for job in self._jobs}
        if len(self._by_id) != len(self._jobs):
            raise ValueError("Job catalog requires unique job IDs")

    @staticmethod
    def _load() -> list[CatalogJob]:
        jobs: list[CatalogJob] = []
        seen: set[str] = set()
        for fixture in load_fixture_jobs():
            key = fixture.expected.model_dump_json()
            if key not in seen:
                seen.add(key)
                jobs.append(CatalogJob(fixture.id, "extraction_fixture", fixture.expected, "Germany"))
        rows = json.loads((FIXTURES / "matching_jobs.json").read_text(encoding="utf-8"))
        for row in rows:
            jobs.append(CatalogJob(row["id"], "matching_fixture", JobPosting.model_validate(row["posting"]),
                                   "Germany", dict(row.get("language_levels") or {})))
        return jobs

    def all(self) -> list[CatalogJob]:
        return list(self._jobs)

    def get(self, job_id: str) -> CatalogJob | None:
        return self._by_id.get(job_id)


def _location_matches(job: CatalogJob, location: str) -> bool:
    wanted = " ".join(location.casefold().split())
    country = _COUNTRY_ALIASES.get(wanted)
    if country and job.country and job.country.casefold() == country:
        return True
    return wanted in (job.posting.location or "").casefold()


def search_jobs(args: SearchJobsArgs, ctx: "ToolContext") -> SearchJobsResult:
    """Deterministic keyword ranking: role-token coverage dominates skill coverage."""
    query = _tokens(args.query)
    scored: list[tuple[float, CatalogJob]] = []
    for job in ctx.catalog.all():
        if args.location and not _location_matches(job, args.location):
            continue
        if not query:
            continue
        role_cov = len(query & _tokens(job.posting.role)) / len(query)
        skill_cov = len(query & set().union(*map(_tokens, job.posting.skills))) / len(query)
        relevance = round(role_cov + 0.5 * skill_cov, 4)
        if relevance > 0:
            scored.append((relevance, job))
    scored.sort(key=lambda item: (-item[0], item[1].job_id))
    jobs = [JobSummary(job_id=j.job_id, role=j.posting.role, location=j.posting.location,
                       seniority=j.posting.seniority, skills=j.posting.skills, relevance=r)
            for r, j in scored[:args.limit]]
    message = None if jobs else "No jobs in the catalog match this query and location."
    return SearchJobsResult(jobs=jobs, total_matches=len(scored), message=message)


def get_job_details(args: GetJobDetailsArgs, ctx: "ToolContext") -> JobDetailsResult:
    job = ctx.catalog.get(args.job_id)
    if job is None:
        raise ToolNotFoundError(f"Unknown job_id '{args.job_id}'. Use search_jobs to find valid IDs.")
    return JobDetailsResult(job_id=job.job_id, source=job.source, posting=job.posting,
                            language_levels=job.language_levels)
