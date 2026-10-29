-- Operational LLM usage and cost log. Counts, timings and labels only:
-- no prompts, messages, model outputs or CV text are stored here.
CREATE TABLE IF NOT EXISTS cost_log (
    id BIGSERIAL PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL,
    run_id TEXT,
    request_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    operation TEXT NOT NULL,
    prompt_version TEXT,
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
    cached_input_tokens INTEGER NOT NULL CHECK (cached_input_tokens >= 0),
    total_tokens INTEGER NOT NULL CHECK (total_tokens >= 0),
    estimated_cost_usd NUMERIC(14, 8),
    latency_ms DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    status TEXT NOT NULL CHECK (status IN ('ok', 'error', 'budget_exceeded')),
    cache_hit BOOLEAN NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    error_category TEXT
);
CREATE INDEX IF NOT EXISTS cost_log_created_at_idx ON cost_log (created_at);
CREATE INDEX IF NOT EXISTS cost_log_run_id_idx ON cost_log (run_id);
