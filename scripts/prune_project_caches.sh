#!/usr/bin/env bash
set -euo pipefail
PREFIX="$1"
KEEP_KEY="$2"
CACHE_JSON="$(gh api -H 'Accept: application/vnd.github+json' "/repos/${GITHUB_REPOSITORY}/actions/caches?per_page=100")"
echo "$CACHE_JSON" | jq -r --arg prefix "$PREFIX" --arg keep "$KEEP_KEY" \
  '.actions_caches[] | select(.key | startswith($prefix)) | select(.key != $keep) | .id' \
  | while read -r cache_id; do
      [ -z "$cache_id" ] && continue
      echo "Eski proje cache siliniyor: $cache_id"
      gh api --method DELETE -H 'Accept: application/vnd.github+json' \
        "/repos/${GITHUB_REPOSITORY}/actions/caches/${cache_id}"
    done
