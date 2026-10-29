"""Minimal saved-job persistence and the save_job tool.

A JSON file is enough for a single-user CLI. Writes are atomic (temp file +
replace). PostgresSavedJobStore is used by the API when DATABASE_URL is set.
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


class PostgresSavedJobStore:
    """saved_jobs table (created by src.db); owner_id is fixed until authentication exists."""

    def __init__(self, dsn: str, owner_id: str = "default") -> None:
        self.dsn, self.owner_id = dsn, owner_id

    def _connect(self):
        import psycopg
        return psycopg.connect(self.dsn, connect_timeout=10)

    @staticmethod
    def _record(row) -> SavedJob:
        return SavedJob(job_id=row[0], note=row[1], saved_at=row[2].astimezone(timezone.utc).isoformat(
            timespec="seconds"))

    def list(self) -> list[SavedJob]:
        with self._connect() as conn:
            rows = conn.execute("SELECT job_id, note, saved_at FROM saved_jobs WHERE owner_id=%s "
                                "ORDER BY saved_at, job_id", (self.owner_id,)).fetchall()
        return [self._record(row) for row in rows]

    def save(self, job_id: str, note: str | None) -> tuple[SavedJob, bool]:
        with self._connect() as conn:
            created = conn.execute("INSERT INTO saved_jobs (owner_id, job_id, note) VALUES (%s, %s, %s) "
                                   "ON CONFLICT (owner_id, job_id) DO NOTHING RETURNING job_id, note, saved_at",
                                   (self.owner_id, job_id, note)).fetchone()
            if created is not None:
                return self._record(created), True
            row = conn.execute("SELECT job_id, note, saved_at FROM saved_jobs WHERE owner_id=%s AND job_id=%s",
                               (self.owner_id, job_id)).fetchone()
        return self._record(row), False


def save_job(args: SaveJobArgs, ctx: "ToolContext") -> SaveJobResult:
    if ctx.catalog.get(args.job_id) is None:
        raise ToolNotFoundError(f"Unknown job_id '{args.job_id}'. Only catalog jobs can be saved.")
    record, created = ctx.saved_jobs.save(args.job_id, args.note)
    return SaveJobResult(job_id=record.job_id, status="saved" if created else "already_saved",
                         saved_at=record.saved_at)
