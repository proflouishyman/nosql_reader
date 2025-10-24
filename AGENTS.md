
This repository uses AI-assisted code generation and automation.
To prevent damage to configuration, databases, and environments, all agents and contributors must strictly follow these rules.

------------------------------------------------------------
GENERAL WORKFLOW RULES
------------------------------------------------------------
1. Change Logging
   - Whenever you make a change, append a summary to a file called AGENTS.log.
   - If the file does not exist, create it automatically.
   - Each entry must include:
     [YYYY-MM-DD HH:MM] <filename(s)> edited — <short description>

2. Connection Integrity Check
   - Before finishing any task, verify that:
       • All MongoDB connections still work.
       • All environment variables (.env, docker-compose.yml, etc.) are present and correctly referenced.

3. Minimal Edits Only
   - Perform the smallest change necessary to satisfy the prompt.
   - Do not refactor or “improve” unrelated code.
   - Follow all naming conventions in the file header exactly.

4. Comment Every Change
   - Every modification must include a short inline comment explaining what was changed and why.

5. Documentation Updates
   - If your change affects configuration, setup steps, or usage:
       • Revise README.md as needed.
       • Revise setup.md as needed.
   - Do not delete or rewrite unrelated sections.

6. When in Doubt, Stop
   - If uncertain about the intent, stop immediately and ask the user for clarification.

7. Startup Discipline
   - Always read README.md before beginning work.
   - Understand the file structure and the purpose of each major module before making changes.

------------------------------------------------------------
ENVIRONMENT & DOCKER VARIABLE GUIDELINES
------------------------------------------------------------
Canonical MongoDB Variable Names:
   MONGO_ROOT_USERNAME=admin
   MONGO_ROOT_PASSWORD=secret
   APP_MONGO_URI=mongodb://admin:secret@mongodb:27017/admin

   • Do not use MONGO_INITDB_ROOT_USERNAME or MONGO_INITDB_ROOT_PASSWORD unless required by a base image.
   • The same variables must appear in:
       - .env
       - docker-compose.yml
       - The consuming application (Flask, Node, etc.)

New Services:
   <SERVICE>_USERNAME
   <SERVICE>_PASSWORD
   <SERVICE>_URI

Never edit .env files unless adding new variables.
Do not touch Docker files.
At all cost, do not break database or Docker connections.

------------------------------------------------------------
ADDITIONAL RULES
------------------------------------------------------------
• Environment variables must not contain quotes or inline comments.
• Ensure .env, docker-compose.yml, and app configuration files stay synchronized.
• Codex agents must not introduce, rename, or delete environment variables without verifying parity across all dependent files.
• Before commit or build, validate Mongo credentials exist:
    if [ -z "$MONGO_ROOT_USERNAME" ] || [ -z "$MONGO_ROOT_PASSWORD" ]; then
        echo "❌ Missing MongoDB credentials in environment"
        exit 1
    fi
• Never rename or restructure environment variables.
• Before merging any PR that modifies .env or docker-compose.yml:
    - Compare both files for variable name parity.
    - Validate that no variables are quoted or commented inline.
    - Preserve YAML indentation and structure exactly.


path = "/mnt/data/AGENT_GUIDELINES.txt"
with open(path, "w") as f:
    f.write(content)
