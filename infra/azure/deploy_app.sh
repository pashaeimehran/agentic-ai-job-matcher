#!/usr/bin/env bash
# Application deployment: build the current working tree, push it to the existing registry and
# point the existing Container App at the new image. Only the Container App changes: PostgreSQL,
# the registry and the environment are not touched, so an incident in another service's
# management plane cannot block an image update. Infrastructure changes (SKUs, secrets, mode,
# new resources) go through deploy.sh and Bicep.
#
# Requires: Azure CLI (logged in, subscription selected), Docker, python3, curl, and a
# Container App created by deploy.sh.
# Optional: AZ_RESOURCE_GROUP (rg-job-match-agent), AZ_CONTAINER_APP (jobmatch-api),
#           IMAGE_TAG (UTC timestamp + commit), SKIP_SMOKE=1.
set -euo pipefail

RG="${AZ_RESOURCE_GROUP:-rg-job-match-agent}"
APP="${AZ_CONTAINER_APP:-jobmatch-api}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=infra/azure/lib.sh
source "$HERE/lib.sh"
TAG="${IMAGE_TAG:-$(default_tag "$ROOT")}"

# The registry is the one the app already pulls from (with its managed identity).
LOGIN_SERVER="$(az containerapp show -g "$RG" -n "$APP" --query "properties.configuration.registries[0].server" -o tsv)"
[[ -n "$LOGIN_SERVER" ]] || { echo "Container App '$APP' in '$RG' not found; run deploy.sh first." >&2; exit 1; }
REGISTRY="${LOGIN_SERVER%%.*}"
IMAGE="$LOGIN_SERVER/job-match-agent:$TAG"

az account show --query "{subscription:name, user:user.name}" -o table
echo "App deployment: $APP in $RG, image $IMAGE"

echo "1/3 Image: docker build + push"
push_image "$REGISTRY" "$LOGIN_SERVER" "$IMAGE" "$ROOT"

echo "2/3 Container App update (new revision)"
REVISION="$(az containerapp update -g "$RG" -n "$APP" --image "$IMAGE" --only-show-errors \
  --query properties.latestRevisionName -o tsv)"
for _ in $(seq 1 60); do
  HEALTH="$(az containerapp revision show -g "$RG" -n "$APP" --revision "$REVISION" \
    --query properties.healthState -o tsv 2>/dev/null || true)"
  [[ "$HEALTH" == "Healthy" || "$HEALTH" == "Unhealthy" ]] && break
  sleep 5
done
echo "Revision $REVISION: ${HEALTH:-unknown}"
[[ "$HEALTH" == "Healthy" ]] || { echo "Revision did not become healthy; previous revision keeps serving." >&2; exit 1; }

URL="https://$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "3/3 Smoke test $URL"
  bash "$HERE/smoke_test.sh" "$URL" --agentic
fi
