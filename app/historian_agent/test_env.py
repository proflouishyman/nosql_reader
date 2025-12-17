import os

# =========================================================
# CONFIGURATION: Update these values as needed
# =========================================================
ENV_VARS = {
    "HISTORIAN_AGENT_ENABLED": "1",
    "HISTORIAN_AGENT_MODEL_PROVIDER": "ollama",
    "HISTORIAN_AGENT_MODEL": "gpt-oss:20b",
    "HISTORIAN_AGENT_TEMPERATURE": "0.2",
    "HISTORIAN_AGENT_CONTEXT_K": "10",
    "HISTORIAN_AGENT_CONTEXT_FIELDS": "title,content",
    "HISTORIAN_AGENT_SUMMARY_FIELD": "content",
    "HISTORIAN_AGENT_FALLBACK": "1",
    "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
    "OPENAI_API_KEY": "" # Add your key here
}

ENV_FILE_PATH = ".env"

def sync_env():
    # Read existing content
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as f:
            lines = f.readlines()
    else:
        lines = []

    # Track which keys from our dictionary were found in the file
    keys_found = set()
    new_content = []

    # Update existing lines
    for line in lines:
        stripped = line.strip()
        
        # Check if line is a variable (contains '=' and doesn't start with '#')
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=")[0]
            if key in ENV_VARS:
                new_content.append(f"{key}={ENV_VARS[key]}\n")
                keys_found.add(key)
                continue
        
        new_content.append(line)

    # Append any variables that weren't in the file yet
    for key, value in ENV_VARS.items():
        if key not in keys_found:
            # Add a newline if appending to a non-empty file
            if new_content and not new_content[-1].endswith('\n'):
                new_content.append('\n')
            new_content.append(f"{key}={value}\n")

    # Save the updated file
    with open(ENV_FILE_PATH, 'w') as f:
        f.writelines(new_content)

    print(f"✅ Environment variables synced successfully to {ENV_FILE_PATH}")

if __name__ == "__main__":
    sync_env()