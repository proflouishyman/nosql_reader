#!/usr/bin/env bash
set -euo pipefail

# Stamp runtime build metadata from the current host git checkout before boot.
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/stamp_build_info.sh"

# Start the local stack after metadata is refreshed so Settings shows current branch/commit.
docker compose up
