"""Minimal saved-job persistence and the save_job tool.

A JSON file is enough for a single-user CLI. Writes are atomic (temp file +
replace). A database-backed store can implement SavedJobStore later.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.tools.errors import ToolNotFoundError
from src.tools.schemas import SaveJobArgs, SaveJobResult

if TYPE_CHECKING:
    from src.tools.context import ToolContext

DEFAULT_SAVED_JOBS_PATH = Path(__file__).resolve().parents[2] / "data" / "private" / "saved_jobs.json"


class SavedJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1)
    note: str | None = None
    saved_at: str


class SavedJobStore(Protocol):
    def save(self, job_id: str, note: str | None) -> tuple[SavedJob, bool]:
        """Return (record, created). Saving an existing job is idempotent."""
        ...

    def list(self) -> list[SavedJob]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemorySavedJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, SavedJob] = {}

    def save(self, job_id, note):
        if job_id in self._jobs:
            return self._jobs[job_id], False
        self._jobs[job_id] = SavedJob(job_id=job_id, note=note, saved_at=_now())
        return self._jobs[job_id], True

    def list(self):
        return list(self._jobs.values())


class JsonFileSavedJobStore:
    def __init__(self, path: str | Path = DEFAULT_SAVED_JOBS_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def list(self) -> list[SavedJob]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [SavedJob.model_validate(item) for item in payload]

    def save(self, job_id: str, note: str | None) -> tuple[SavedJob, bool]:
        with self._lock:
            jobs = self.list()
            for job in jobs:
                if job.job_id == job_id:
                    return job, False
            record = SavedJob(job_id=job_id, note=note, saved_at=_now())
            jobs.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump([j.model_dump() for j in jobs], handle, indent=2)
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            return record, True


def save_job(args: SaveJobArgs, ctx: "ToolContext") -> SaveJobResult:
    if ctx.catalog.get(args.job_id) is None:
        raise ToolNotFoundError(f"Unknown job_id '{args.job_id}'. Only catalog jobs can be saved.")
    record, created = ctx.saved_jobs.save(args.job_id, args.note)
    return SaveJobResult(job_id=record.job_id, status="saved" if created else "already_saved",
                         saved_at=record.saved_at)
