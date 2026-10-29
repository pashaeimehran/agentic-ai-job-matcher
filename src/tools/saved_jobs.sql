-- Saved jobs for the single configured user (owner_id prepares for multi-user isolation).
CREATE TABLE IF NOT EXISTS saved_jobs (
    owner_id TEXT NOT NULL DEFAULT 'default',
    job_id TEXT NOT NULL,
    note TEXT,
    saved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id, job_id)
);
