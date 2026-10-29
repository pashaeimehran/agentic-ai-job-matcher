#!/usr/bin/env bash
# Smoke test a deployed (or local) API. Prints status codes, latency and mode only, never response
# bodies that could contain CV content.   Usage: smoke_test.sh https://<fqdn> [--agentic]
set -euo pipefail
BASE="${1:?usage: smoke_test.sh BASE_URL [--agentic]}"
AGENTIC="${2:-}"
BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT

check() {  # name method path [json-body] -> prints "name status seconds", fails on non-2xx
  local name="$1" method="$2" path="$3" body="${4:-}" out
  if [[ -n "$body" ]]; then
    out=$(curl -sS -o "$BODY" -w "%{http_code} %{time_total}" -X "$method" \
          -H 'Content-Type: application/json' -d "$body" --max-time 90 "$BASE$path")
  else
    out=$(curl -sS -o "$BODY" -w "%{http_code} %{time_total}" -X "$method" --max-time 90 "$BASE$path")
  fi
  echo "$name $out"
  [[ "${out%% *}" =~ ^2 ]] || { echo "FAILED: $name"; exit 1; }
}

# Scale-to-zero: the first request may wait for a cold start.
for _ in $(seq 1 30); do curl -sf --max-time 10 "$BASE/health" >/dev/null && break; sleep 5; done
check health GET /health
python3 -c 'import json, sys; print("mode", json.load(open(sys.argv[1]))["mode"])' "$BODY"
check ready GET /ready
check jobs GET "/jobs?limit=5"
check search POST /search '{"target":"jobs","query":"AI Engineer","location":"Germany"}'
check analyze POST /analyze-job '{"job_id":"match-01-python-backend"}'
python3 -c 'import json, sys; b=json.load(open(sys.argv[1])); assert b["score_source"]=="deterministic"; print("deterministic score", b["match"]["score"]["value"])' "$BODY"
if [[ "$AGENTIC" == "--agentic" ]]; then
  check chat POST /chat '{"message":"Find AI Engineer jobs in Germany that fit my CV"}'
  python3 -c 'import json, sys; b=json.load(open(sys.argv[1])); print("chat tools", [t["tool"] for t in b["tool_calls"]])' "$BODY"
fi
echo "Smoke test passed: $BASE"
