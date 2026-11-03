#!/usr/bin/env bash
# Prints the id of a docker-publish run that built and tested <sha> uncached on legolas within the
# last 7 days (the runners' build-cache lifetime), or nothing. Usage: release-reuse-check.sh <repo> <sha>
set -euo pipefail
repo=$1 sha=$2
since=$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)
runs=$(gh api "repos/$repo/actions/workflows/docker-publish.yml/runs?branch=main&event=workflow_dispatch&status=success&head_sha=$sha&created=%3E%3D$since&per_page=20" --jq '.workflow_runs[].id')
for run in $runs; do
  read -r job runner < <(gh api "repos/$repo/actions/runs/$run/jobs?per_page=50" \
    --jq '.jobs[] | select(.name == "deeplearning") | "\(.id) \(.runner_name)"')
  [[ "${runner:-}" == legolas-* ]] || continue
  # The build action logs its inputs; an empty tag_name is omitted, a release build's is not.
  inputs=$(gh api "repos/$repo/actions/jobs/$job/logs" | sed -E 's/^[0-9T:.-]+Z //' \
    | awk '/^##\[group\]Run \.\/\.github\/actions\/docker_build_push/ {on=1; next} on && /^env:/ {on=0; done=1} on && !done')
  grep -qx '  use_cache: false' <<<"$inputs" || continue
  grep -q '^  tag_name:' <<<"$inputs" && continue
  echo "$run"
  exit 0
done
