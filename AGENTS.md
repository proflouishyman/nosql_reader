Whenever you make a change, append it to a logfile called AGENTS.log    If the file doesn't exist create it.

Before you finish, make sure that the mongodb connections have not been broken and that all the environmental variables are still correctly used

Do the minimal changes for the solution to the prompt.

Follow the naming conventions in the file header


Comment everything you do.

Revise the readme.md as needed

Revise the setup.md as needed

When in doubt, stop working and ask a question of the user.

always read the readme.md file when you start so you know what the file structure looks like and how the program works. 

environment & Docker Variable Guidelines

# ===========================================================
# MongoDB Connection Standards
# Created: 2025-10-24
# Purpose: Ensure consistent variable naming and prevent Codex
#          or automated tools from renaming or duplicating vars.
#
# NAMING CONVENTIONS
# -----------------------------------------------------------
# MONGO_URI    → environment variable or constant holding the
#                full Mongo connection string.
# client       → the active pymongo.MongoClient instance.
# db           → database handle, e.g. client["admin"] or client[DB_NAME].
# collection   → collection handle from db["collection_name"].
#
# RULES
# -----------------------------------------------------------
# - Do NOT create variants like mongoURI, mongo_uri, or client2.
# - Do NOT rename MONGO_URI or client.
# - Do NOT modify imports or connection structure unless explicitly instructed.
# - Use get_env() or os.environ.get() to retrieve env vars safely.
# - Handle connection errors with try/except, logging the exception only.
# ===========================================================

These rules prevent authentication errors, failed Docker builds, and misaligned configurations when Codex or other automated agents modify environment or Docker files.
1. Canonical Variable Names
Always use the following canonical naming convention for MongoDB and related services:
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=secret
APP_MONGO_URI=mongodb://admin:secret@mongodb:27017/admin
Do not use MONGO_INITDB_ROOT_USERNAME or MONGO_INITDB_ROOT_PASSWORD unless explicitly required by an upstream image.
The same variable names must appear in:
.env
docker-compose.yml
The consuming app (Flask, Node, etc.)
New services must follow the same pattern:
<SERVICE>_USERNAME, <SERVICE>_PASSWORD, <SERVICE>_URI

Never edit .env files unless adding new variables.
Do not touch the docker files.
At all cost, do not break the database, docker connections.

3. No Quotes or Inline Comments in .env
Environment variables must be defined without quotes or inline comments.
❌ Incorrect
MONGO_URI= "mongodb://admin:secret@mongodb:27017/admin" # admin db
✅ Correct
APP_MONGO_URI=mongodb://admin:secret@mongodb:27017/admin
# the @ denotes the database host; /admin selects the admin database
Docker Compose treats quotes and inline comments as literal characters, which breaks authentication strings.
3. Keep .env, docker-compose.yml, and App Configs Synchronized
Every variable in .env must have a matching reference in docker-compose.yml.
Codex agents must not introduce, rename, or delete environment variables without verifying their presence in all dependent files.
When new variables are added, update all relevant files in the same PR.
4. MongoDB Initialization Rules
MongoDB credentials are written only once during the first container startup.
If credentials change, the database volume must be wiped and reinitialized:
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
Editing .env alone will not reset the root user. The data volume must be rebuilt.
5. Validation Before Commit or Build
Before building, committing, or merging, run this check:
if [ -z "$MONGO_ROOT_USERNAME" ] || [ -z "$MONGO_ROOT_PASSWORD" ]; then
  echo "❌ Missing MongoDB credentials in environment"
  exit 1
fi
Add this to CI or pre-build hooks to prevent missing credential injection.
6. Agent Merge Behavior
Codex and other AI agents must not rename or restructure environment variables.
Before merging any PR that modifies .env or docker-compose.yml, agents must:
Compare both files for variable name parity.
Validate that no variables are quoted or commented inline.
Preserve the structure and indentation of YAML files exactly.
Agent Coding & Stability Guide (add to agents.md)
# Agent Development & Stability Guidelines

This project is designed to run reliably inside a multi-container Docker environment.
All contributors and AI agents must follow these conventions to maintain consistency and prevent misconfiguration.

---

## 1. Environment Variables
- Always use canonical MongoDB variables:
  ```bash
  MONGO_INITDB_ROOT_USERNAME
  MONGO_INITDB_ROOT_PASSWORD
  APP_MONGO_URI
Never hard-code credentials or hostnames.
Access the URI from os.environ["MONGO_URI"] — the entrypoint script ensures it’s defined.
Do not modify .env formatting (no quotes, no inline comments).
2. Database Access
All MongoDB access must go through database_setup.py:
from database_setup import get_client, get_db
Never call MongoClient() directly in route or helper files.
Ensure get_client() validates the connection before queries.
3. Application Structure
The app is initialized in app.py.
Routes must import this instance:
from app import app
Do not reinitialize Flask() elsewhere.
Avoid circular imports — always import routes after app creation.
4. Logging
Use the Flask logger (app.logger or current_app.logger) for all logs.
Do not use print() for debugging; logs are captured in Docker.
Prefer structured messages:
app.logger.info("Loaded collection: %s", collection_name)
app.logger.exception("Failed to insert document", exc_info=True)
5. Stability & Build Hygiene
Never modify Dockerfile, docker-compose.yml, or .env in generated merges unless explicitly requested.
Respect canonical variable names and paths (/app, /data/archives, /data/db).
If new dependencies are required, add them to requirements.txt and rebuild using:
docker-compose build --no-cache
Before merging, verify startup logs show:
MongoDB is up and running.
Starting Flask app...
Following these conventio
