Whenever you make a change, append it to a logfile called AGENTS.log    If the file doesn't exist create it.

Before you finish, make sure that the mongodb connections have not been broken and that all the environmental variables are still correctly used

Do the minimal changes for the solution to the prompt.

Comment everything you do.

Revise the readme.md as needed

Revise the setup.md as needed

When in doubt, stop working and ask a question of the user.

always read the readme.md file when you start so you know what the file structure looks like and how the program works. 

environment & Docker Variable Guidelines
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
2. No Quotes or Inline Comments in .env
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
