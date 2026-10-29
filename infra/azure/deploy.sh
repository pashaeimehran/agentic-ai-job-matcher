#!/usr/bin/env bash
# Infrastructure deployment (Bicep is the source of truth) and first deployment:
#   resource group -> platform (Bicep) -> Docker build + push to ACR -> Container App -> smoke test.
# Use it for new environments and infrastructure changes (SKUs, secrets, APP_MODE, resources).
# Every run re-applies every resource, including PostgreSQL. For a new application image only,
# use deploy_app.sh, which updates the Container App and nothing else.
#
# The image is built by the local Docker engine from the current working tree and pushed with a
# short-lived Entra token (az acr login --expose-token, valid ~3 h). ACR Tasks (az acr build) are
# not used: they are paused on free-credit subscriptions, and a plain build/push works everywhere.
# The registry admin user stays disabled; the Container App pulls with a managed identity.
#
# Creates cost-incurring resources (PostgreSQL Flexible B1ms, Basic ACR, Log Analytics, Container
# Apps). Review docs/deployment.md first.
#
# Requires: Azure CLI (logged in, subscription selected), Docker, python3, curl.
# Secrets:  POSTGRES_ADMIN_PASSWORD from the environment, or generated once and kept outside the
#           repository in $JMA_SECRETS_DIR/postgres-admin-password (default ~/.config/job-match-agent,
#           mode 600). Re-runs must use the same password. Nothing secret is echoed or put on a
#           command line: secure parameters go through a temporary mode-600 parameters file.
# Optional: AZ_RESOURCE_GROUP (rg-job-match-agent), AZ_LOCATION (northeurope), AZ_PREFIX (jobmatch),
#           APP_MODE (fixture|live), OPENAI_API_KEY (needed for live), IMAGE_TAG (UTC timestamp +
#           commit), LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (optional tracing),
#           DRY_RUN=1 (create the empty resource group, then only build, lint, validate and what-if).
set -euo pipefail

RG="${AZ_RESOURCE_GROUP:-rg-job-match-agent}"
LOCATION="${AZ_LOCATION:-northeurope}"
PREFIX="${AZ_PREFIX:-jobmatch}"
MODE="${APP_MODE:-fixture}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TEMPLATE="$HERE/main.bicep"
# shellcheck source=infra/azure/lib.sh
source "$HERE/lib.sh"
TAG="${IMAGE_TAG:-$(default_tag "$ROOT")}"

if [[ "$MODE" == "live" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "APP_MODE=live requires OPENAI_API_KEY" >&2; exit 1
fi

if [[ -z "${POSTGRES_ADMIN_PASSWORD:-}" ]]; then
  SECRETS_DIR="${JMA_SECRETS_DIR:-$HOME/.config/job-match-agent}"
  PW_FILE="$SECRETS_DIR/postgres-admin-password"
  if [[ ! -s "$PW_FILE" ]]; then
    mkdir -p "$SECRETS_DIR" && chmod 700 "$SECRETS_DIR"
    # 32 letters and digits with at least one of each class (Azure PostgreSQL complexity rules).
    (umask 077; python3 -c 'import secrets, string
a = string.ascii_letters + string.digits
while True:
    p = "".join(secrets.choice(a) for _ in range(32))
    if any(c.islower() for c in p) and any(c.isupper() for c in p) and any(c.isdigit() for c in p):
        print(p); break' > "$PW_FILE")
    echo "Generated a PostgreSQL admin password in $PW_FILE (keep it; re-runs need it)."
  fi
  POSTGRES_ADMIN_PASSWORD="$(cat "$PW_FILE")"
fi

# Secure parameters in a temporary file readable only by this user, removed on exit.
PARAMS="$(mktemp)"
trap 'rm -f "$PARAMS"' EXIT
chmod 600 "$PARAMS"
POSTGRES_ADMIN_PASSWORD="$POSTGRES_ADMIN_PASSWORD" python3 - "$PARAMS" <<'PY'
import json, os, sys
names = {"postgresAdminPassword": "POSTGRES_ADMIN_PASSWORD", "openAiApiKey": "OPENAI_API_KEY",
         "langfuseHost": "LANGFUSE_HOST", "langfusePublicKey": "LANGFUSE_PUBLIC_KEY",
         "langfuseSecretKey": "LANGFUSE_SECRET_KEY"}
params = {key: {"value": os.environ.get(env, "")} for key, env in names.items()}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump({"$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
               "contentVersion": "1.0.0.0", "parameters": params}, fh)
PY

az account show --query "{subscription:name, user:user.name}" -o table
echo "Resource group '$RG' in '$LOCATION' (mode: $MODE, image tag: $TAG)"

az group create --name "$RG" --location "$LOCATION" --output none

# $1 = create | validate | what-if, $2 = deployApp true|false, further arguments go to az.
deployment() {
  local extra=("${@:3}")
  [[ "$1" == "create" ]] && extra+=(--query "properties.outputs" --output json)
  az deployment group "$1" \
    --resource-group "$RG" --name "job-match-agent-app-$2" \
    --template-file "$TEMPLATE" \
    --parameters @"$PARAMS" \
    --parameters prefix="$PREFIX" location="$LOCATION" deployApp="$2" imageTag="$TAG" appMode="$MODE" \
    "${extra[@]}"
}

output() {  # $1 = deployment outputs JSON, $2 = output name
  python3 -c 'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]]["value"])' "$1" "$2"
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: Bicep build and lint, ARM validation and what-if; no resources besides the resource group."
  az bicep build --file "$TEMPLATE" --stdout > /dev/null
  az bicep lint --file "$TEMPLATE"
  deployment validate false --output none && echo "validate (platform): ok"
  deployment validate true --output none && echo "validate (platform + app): ok"
  deployment what-if true
  exit 0
fi

echo "1/4 Platform: registry, Log Analytics, Container Apps environment, PostgreSQL + pgvector"
OUTPUTS="$(deployment create false)"
REGISTRY="$(output "$OUTPUTS" registryName)"
LOGIN_SERVER="$(output "$OUTPUTS" registryLoginServer)"
IMAGE="$LOGIN_SERVER/job-match-agent:$TAG"

echo "2/4 Image: docker build + push $IMAGE"
push_image "$REGISTRY" "$LOGIN_SERVER" "$IMAGE" "$ROOT"

echo "3/4 Container App"
OUTPUTS="$(deployment create true)"
URL="$(output "$OUTPUTS" appUrl)"
echo "Deployed: $URL"

# /chat is free in fixture mode (scripted provider) and one small paid call in live mode.
echo "4/4 Smoke test"
bash "$HERE/smoke_test.sh" "$URL" --agentic
