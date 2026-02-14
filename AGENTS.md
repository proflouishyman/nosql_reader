
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
   - Record when ingestion defaults or selector behaviour changes so operators understand why the UI shifted. <!-- Added reminder to log ingestion-facing updates. -->

2. Connection Integrity Check
   - Before finishing any task, verify that:
       • All MongoDB connections still work (use the existing quick_check.sh helper when available). <!-- Clarified how to perform the connection check. -->
       • All environment variables (.env, docker-compose.yml, etc.) are present and correctly referenced.
       • Ingestion defaults still honour `HISTORIAN_AGENT_MODEL_PROVIDER` and `HISTORIAN_AGENT_MODEL`. <!-- Added ingestion alignment requirement. -->

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
       • Update ingestion design docs when the selector workflow, defaults, or error handling change. <!-- Added doc sync note for ingestion features. -->
   - Do not delete or rewrite unrelated sections.

6. When in Doubt, Stop
   - If uncertain about the intent, stop immediately and ask the user for clarification.

7. Startup Discipline
   - Always read README.md before beginning work.
   - Understand the file structure and the purpose of each major module before making changes.

8. Overnight Runs
   - When the user asks for an overnight run, complete any required fixes first, then launch a stable run and keep it running unless explicitly told to stop. <!-- Added to ensure overnight time is actually used once fixes land. -->

9. Long-Run Discipline
   - For runs expected to take >30 minutes, write to a dedicated log file, tail it periodically, and report progress at regular intervals. <!-- Added to prevent silent long runs. -->

10. Evidence-First Principle
    - If evidence is missing, label output as a gap; do not infer or fabricate. <!-- Added to enforce historian standards. -->

11. False-Positive Bias
    - Prefer false negatives over false positives; if unclear, exclude the claim from synthesis. <!-- Added to align with historian expectations. -->

12. Model Comparison Protocol
    - When comparing models, keep inputs identical and report a small rubric (length, evidence density, coherence). <!-- Added for repeatable evals. -->

13. Output Artifacts
    - Always write outputs to a timestamped path and report that path in the response. <!-- Added for traceability. -->

14. Commit Cadence
    - For iterative experiments, commit after each material improvement (not every small tweak). <!-- Added to balance traceability and noise. -->

15. “Don’t Stall” Rule
    - If a run fails twice, add logging and reduce scope to isolate the failing stage before retrying. <!-- Added to keep progress moving. -->

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
• Maintain Ollama/OpenAI selector defaults so the UI bootstraps from .env values and displays ingestion errors both on the page and in server logs. <!-- Added rule connecting UI expectations from the latest change. -->
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
