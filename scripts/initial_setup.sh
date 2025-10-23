#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_VERSINFO:-0}" -lt 4 ]]; then
  echo "[warn] Bash 4 or newer is recommended. Current version: ${BASH_VERSION:-unknown}" >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"

ARCHIVES_DIR="${PARENT_DIR}/archives"
MONGO_DIR="${PARENT_DIR}/mongo_data"
SESSION_DIR="${PARENT_DIR}/flask_session"

cat <<SUMMARY
This script will perform the following actions:

  • Create data directories one level above the repository:
      - Archives:      ${ARCHIVES_DIR}
      - MongoDB data:  ${MONGO_DIR}
      - Flask session: ${SESSION_DIR}
  • Ensure a .env file exists in the repository root.
  • Update the .env file so the above directories are used by Docker Compose.
  • Optionally update MongoDB root credentials while keeping working defaults.

SUMMARY

read -r -p "Proceed with the setup? [y/N]: " confirm
if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
  echo "Aborting setup." >&2
  exit 1
fi

for dir in "${ARCHIVES_DIR}" "${MONGO_DIR}" "${SESSION_DIR}"; do
  mkdir -p "${dir}"
  echo "Created directory: ${dir}"
done

ENV_FILE="${REPO_ROOT}/.env"
ENV_TEMPLATE="${REPO_ROOT}/.env.example"

if [[ ! -f "${ENV_TEMPLATE}" ]]; then
  echo "Error: .env.example is missing in the repository root." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${ENV_TEMPLATE}" "${ENV_FILE}"
  echo "Created ${ENV_FILE} from .env.example"
fi

current_username=$(grep -E '^MONGO_ROOT_USERNAME=' "${ENV_FILE}" | head -n1 | cut -d'=' -f2- || true)
current_username=${current_username:-admin}

current_password=$(grep -E '^MONGO_ROOT_PASSWORD=' "${ENV_FILE}" | head -n1 | cut -d'=' -f2- || true)
current_password=${current_password:-change-me}

echo ""
read -r -p "Would you like to update the MongoDB root credentials now? [y/N]: " update_creds
if [[ "${update_creds}" =~ ^[Yy]$ ]]; then
  echo ""
  read -r -p "MongoDB root username [${current_username}]: " new_username
  if [[ -n "${new_username}" ]]; then
    current_username="${new_username}"
  fi

  read -s -p "MongoDB root password (leave blank to keep current): " new_password
  echo ""
  if [[ -n "${new_password}" ]]; then
    current_password="${new_password}"
  fi
  echo "MongoDB credentials updated."
else
  echo "Keeping existing MongoDB credentials (username: ${current_username})."
fi

export ENV_FILE ARCHIVES_DIR MONGO_DIR SESSION_DIR
export MONGO_USERNAME="${current_username}" MONGO_PASSWORD="${current_password}"
python <<'PY'
import os
import pathlib
import re
from urllib.parse import urlparse

env_path = pathlib.Path(os.environ["ENV_FILE"])
contents = env_path.read_text()

def build_mongo_uri(existing: str) -> str:
    default_uri = "mongodb://mongodb:27017/admin"
    current = existing.strip() or default_uri
    parsed = urlparse(current)
    scheme = parsed.scheme or "mongodb"
    host = parsed.hostname or "mongodb"
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or "/admin"
    query = f"?{parsed.query}" if parsed.query else ""
    return (
        f"{scheme}://{os.environ['MONGO_USERNAME']}:{os.environ['MONGO_PASSWORD']}@"
        f"{host}{port}{path}{query}"
    )

existing_uri_match = re.search(r"^APP_MONGO_URI=(.*)$", contents, re.MULTILINE)
existing_uri = existing_uri_match.group(1) if existing_uri_match else ""

updates = {
    "ARCHIVES_HOST_PATH": os.environ["ARCHIVES_DIR"],
    "MONGO_DATA_HOST_PATH": os.environ["MONGO_DIR"],
    "SESSION_HOST_PATH": os.environ["SESSION_DIR"],
    "MONGO_ROOT_USERNAME": os.environ["MONGO_USERNAME"],
    "MONGO_ROOT_PASSWORD": os.environ["MONGO_PASSWORD"],
    "APP_MONGO_URI": build_mongo_uri(existing_uri),
}

for key, value in updates.items():
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    replacement = f"{key}={value}"
    if pattern.search(contents):
        contents = pattern.sub(replacement, contents)
    else:
        if not contents.endswith("\n"):
            contents += "\n"
        contents += replacement + "\n"

env_path.write_text(contents)
PY

echo "Updated directory paths inside ${ENV_FILE}"
echo "MongoDB connection string synchronised with current credentials."

echo "Setup complete. Review ${ENV_FILE} for additional configuration values."
