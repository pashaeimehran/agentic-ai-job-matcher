# shellcheck shell=bash
# Shared helpers for deploy.sh (infrastructure) and deploy_app.sh (application image). Sourced, not run.

# Traceable image tag without creating a Git tag: UTC time, commit, and "-dirty" for uncommitted changes.
default_tag() {  # $1 = repository root
  local stamp commit
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  commit="$(git -C "$1" rev-parse --short HEAD 2>/dev/null || true)"
  if [[ -z "$commit" ]]; then echo "$stamp"; return; fi
  if [[ -n "$(git -C "$1" status --porcelain 2>/dev/null)" ]]; then commit="$commit-dirty"; fi
  echo "$stamp-$commit"
}

# Build the image from the working tree and push it with a short-lived Entra token (~3 h).
# No ACR Tasks (az acr build) and no registry admin user.
push_image() {  # $1 = registry name, $2 = login server, $3 = full image reference, $4 = build context
  docker build --tag "$3" "$4"
  az acr login --name "$1" --expose-token --query accessToken --output tsv \
    | docker login "$2" --username 00000000-0000-0000-0000-000000000000 --password-stdin
  docker push "$3"
  docker logout "$2" >/dev/null
}
