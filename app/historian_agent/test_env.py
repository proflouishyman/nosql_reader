import os

# ============================================================================
# MASTER CONFIGURATION: Centralized values for the entire pipeline
# ============================================================================
ENV_VARS = {
    # 1. MongoDB Configuration
    "MONGO_INITDB_ROOT_USERNAME": "admin",
    "MONGO_INITDB_ROOT_PASSWORD": "secret",
    "MONGO_URI": "mongodb://admin:secret@mongodb:27017/admin",
    "DB_NAME": "railroad_documents",
    "CHUNKS_COLLECTION": "document_chunks",
    "DOCS_COLLECTION": "documents",

    # 2. Ollama & LLM Master Settings
    "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
    "LLM_MODEL": "gpt-oss:20b",
    "LLM_TEMPERATURE": "0.2",
    "LLM_MAX_TOKENS": "10000",
    "MAX_CONTEXT_TOKENS": "100000",
    "HISTORIAN_AGENT_MODEL": "gpt-oss:20b",
    "HISTORIAN_AGENT_TEMPERATURE": "0.2",

    # 3. RAG & Retrieval Parameters
    "HISTORIAN_AGENT_TOP_K": "50",
    "VECTOR_WEIGHT": "0.7",
    "KEYWORD_WEIGHT": "0.3",
    "HISTORIAN_AGENT_HYBRID_ALPHA": "0.5",
    "HISTORIAN_AGENT_EMBEDDING_PROVIDER": "ollama",
    "HISTORIAN_AGENT_EMBEDDING_MODEL": "qwen3-embedding:0.6b",
    "HISTORIAN_AGENT_VECTOR_STORE": "chroma",
    "CHROMA_PERSIST_DIRECTORY": "/data/chroma_db/persist",

    # 4. Reranking & Adversarial Pipeline
    "CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "CROSS_ENCODER_WEIGHT": "0.85",
    "TEMPORAL_WEIGHT": "0.10",
    "ENTITY_WEIGHT": "0.05",
    "FINAL_TOP_K": "10",
    "CRITIQUE_TEMPERATURE": "0.1",

    # 5. Flask & System Paths
    "SECRET_KEY": "super-secret-key",
    "FLASK_ENV": "development",
    "ARCHIVES_PATH": "/data/archives/",
    "RUN_BOOTSTRAP": "0",
    "OPENAI_API_KEY": "", # Add your key here if needed

    # How many initial chunks to pull from the DB
    "HISTORIAN_AGENT_TOP_K": "100",

    # How many FULL documents to retrieve in Tier 2 expansion (Small-to-Big)
    "PARENT_RETRIEVAL_CAP": "15",


    #6. Additional settings can be added here as needed
    "DEBUG_MODE": "1"  # Set to "0" for clean research reports
}

ENV_FILE_PATH = ".env"

def sync_env():
    """
    Syncs the ENV_VARS dictionary to the .env file.
    Existing variables are updated; new ones are appended.
    Comments and order are preserved where possible.
    """
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as f:
            lines = f.readlines()
    else:
        lines = []

    keys_found = set()
    new_content = []

    # Update existing lines
    for line in lines:
        stripped = line.strip()
        
        # Check if line is a variable (contains '=' and doesn't start with '#')
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=")[0].strip()
            if key in ENV_VARS:
                new_content.append(f"{key}={ENV_VARS[key]}\n")
                keys_found.add(key)
                continue
        
        new_content.append(line)

    # Append any missing variables with a category header
    missing_keys = [k for k in ENV_VARS.keys() if k not in keys_found]
    if missing_keys:
        if new_content and not new_content[-1].endswith('\n'):
            new_content.append('\n')
        new_content.append("\n# =========================================================\n")
        new_content.append("# NEWLY SYNCED VARIABLES\n")
        new_content.append("# =========================================================\n")
        for key in missing_keys:
            new_content.append(f"{key}={ENV_VARS[key]}\n")

    # Save the updated file
    with open(ENV_FILE_PATH, 'w') as f:
        f.writelines(new_content)

    print(f"✅ Successfully synced {len(ENV_VARS)} variables to {ENV_FILE_PATH}")

if __name__ == "__main__":
    sync_env()