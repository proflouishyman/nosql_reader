#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_file="${repo_root}/app/build_info.json"

branch="$(git -C "${repo_root}" rev-parse --abbrev-ref HEAD)"
commit_full="$(git -C "${repo_root}" rev-parse HEAD)"
commit_short="$(git -C "${repo_root}" rev-parse --short=8 HEAD)"
commit_date="$(git -C "${repo_root}" show -s --format=%cI HEAD)"
commit_subject="$(git -C "${repo_root}" show -s --format=%s HEAD | sed 's/\"/\\"/g')"

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  dirty="yes"
else
  dirty="no"
fi

generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat > "${out_file}" <<EOF
{
  "branch": "${branch}",
  "commit": "${commit_short}",
  "commit_full": "${commit_full}",
  "commit_date": "${commit_date}",
  "commit_subject": "${commit_subject}",
  "dirty": "${dirty}",
  "generated_at": "${generated_at}"
}
EOF

echo "Stamped ${out_file}"
