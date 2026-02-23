# First-Time Setup Guide

This document walks through setting up the Historical Document Reader on both macOS and Windows PCs. It covers installing prerequisites, preparing MongoDB data directories, editing environment variables, and configuring the UI JSON used by the application.

---

## 1. Overview of the stack

The project is designed to run through Docker Compose. Two containers start together:

- **Flask web app** – serves the search UI, Historian Agent, and ingestion endpoints.
- **MongoDB** – stores ingested documents, term statistics, and conversation history.

Persistent data lives on the host file system so you can rebuild containers without losing state. The `.env` file controls environment variables, while `app/config.json` governs UI styling and layout.

Environment policy for this repository:

- Use one runtime env file only: `/Users/louishyman/coding/nosql/nosql_reader_cleanup/.env`.
- Do not create additional `.env` files in subfolders.
- Example variables are documented in `/Users/louishyman/coding/nosql/nosql_reader_cleanup/docs/ENV_EXAMPLES.md`.

---

## 2. macOS setup

### 2.1 Install prerequisites

1. **Homebrew (recommended):**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   If you prefer not to install Homebrew, you can download the required apps manually.

2. **Docker Desktop for Mac:**
   - Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
   - After installation, launch Docker Desktop and ensure it reports "Docker is running".

3. **Git:**
   ```bash
   brew install git
   ```
   macOS often ships with Git, but Homebrew guarantees an up-to-date version.

4. **Python 3.10+ (optional but useful for helper scripts):**
   ```bash
   brew install python@3.11
   ```

### 2.2 Clone the repository

```bash
git clone https://github.com/proflouishyman/nosql_reader.git
cd nosql_reader
```

### 2.3 Run the bootstrap script (creates host directories)

The repository ships with `scripts/initial_setup.sh`, a helper that prepares the recommended folder structure and updates your `.env` file.

```bash
./scripts/initial_setup.sh
```

> **Note:** The script requires Bash. In PowerShell, run `bash ./scripts/initial_setup.sh`.

By default the script places persistent data one directory above the cloned repo:

- `../mongo_data`
- `../flask_session`
- `../archives`

You will be prompted before anything is created, and you can relocate the folders later by editing `.env`. After preparing the
directories the script offers to update the MongoDB root username and password. Press **Enter** to keep the working defaults
(`admin` / `change-me`) or supply your own values—the script rewrites both `MONGO_ROOT_*` entries and the `APP_MONGO_URI`
connection string to match.

### 2.4 Configure environment variables

1. Edit the root `.env` with your preferred editor (`code .env`, `nano .env`, etc.) and update:
   - `SECRET_KEY` – random string used by Flask sessions.
   - `MONGO_ROOT_USERNAME` / `MONGO_ROOT_PASSWORD` – credentials for MongoDB. The defaults (`admin` / `change-me`) are ready for
     local development, and the setup script can regenerate them if you ever change the values.
   - `ARCHIVES_HOST_PATH`, `MONGO_DATA_HOST_PATH`, `SESSION_HOST_PATH` – absolute or relative paths on your Mac.
   - Historian Agent settings (`HISTORIAN_AGENT_*`, `OLLAMA_BASE_URL`, `OPENAI_API_KEY` if using OpenAI).

> **Tip:** Use absolute paths if the project folder is synced via iCloud or a network drive; Docker on macOS handles them more reliably.

### 2.5 Configure the UI JSON

The UI pulls typography, colour, spacing, and layout options from `app/config.json`. To customise:

1. Open `app/config.json` in a JSON-aware editor.
2. Modify values under the `fonts`, `sizes`, `colors`, and `spacing` keys. Each key has both a flat `"key[value]"` entry (legacy support) and a nested object version. Keep the nested objects in sync if you change a value.
3. Save the file. The Flask app reloads the JSON on each request, so no rebuild is required to see cosmetic changes.

### 2.6 Start the stack

```bash
docker compose up --build
```

The Flask app becomes available at [http://localhost:5001](http://localhost:5001). MongoDB listens on port 27017. Use `docker compose down` to stop containers and `docker compose down -v` to remove volumes (Mongo data will persist because it is mapped to `MONGO_DATA_HOST_PATH`).

Runtime utility status:

- Utility scripts were consolidated.
- Canonical diagnostics/test entrypoint is `app/util/e2e_cli_suite.py`.

### 2.7 Populate the archives directory (optional first run)

- Place JSON or JSONL source files inside the folder referenced by `ARCHIVES_HOST_PATH`.
- Store related media (images, PDFs, etc.) alongside the JSON with matching relative paths. For example: `archives/newspapers/1901-05-12.jsonl` and `archives/newspapers/images/page_001.png`.
- When the app runs, you can trigger ingestion via `docker compose exec app /app/bootstrap_data.sh` or manually run `python app/data_processing.py` inside the container.

---

## 3. Windows PC setup

Windows users can run the project with Docker Desktop (recommended) or inside Windows Subsystem for Linux (WSL2). The instructions below use Docker Desktop with WSL2 integration enabled.

### 3.1 Install prerequisites

1. **Enable WSL2 (Windows 10 2004+ / Windows 11):**
   - Open PowerShell as Administrator and run:
     ```powershell
     wsl --install
     ```
   - Reboot when prompted. During installation you may be asked to choose a Linux distribution (Ubuntu is the default).

2. **Docker Desktop for Windows:**
   - Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
   - During setup, ensure **Use WSL 2 based engine** is checked.
   - After installation, start Docker Desktop and wait for it to report that Docker is running.

3. **Git for Windows:**
   - Download from [https://git-scm.com/download/win](https://git-scm.com/download/win) and install with default options (including "Git from the command line and also from 3rd-party software").

4. **Optional: Windows Terminal and VS Code** for a better editing experience.

### 3.2 Clone the repository

You can clone inside your Windows user directory or inside your WSL filesystem. If you work in WSL (recommended for performance), open an Ubuntu terminal and run:

```bash
cd ~
git clone https://github.com/proflouishyman/nosql_reader.git
cd nosql_reader
```

When cloning inside WSL, the path will look like `/home/<user>/nosql_reader`. Docker Desktop can access these files through its WSL integration.

### 3.3 Prepare host directories

From PowerShell, Command Prompt, or your WSL shell, run the bootstrap script once:

```bash
./scripts/initial_setup.sh
```

> **Tip:** In pure PowerShell, invoke the script with `bash ./scripts/initial_setup.sh`.

It confirms the action and creates the default directories one level up (e.g., `..\mongo_data`). You can safely re-run it if you change machines—the script only creates missing folders and keeps existing `.env` values in sync. When prompted you may enter new MongoDB credentials or press **Enter** to keep the ready-to-use defaults (`admin` / `change-me`). To store data elsewhere, edit the paths in `.env` after the script completes.

### 3.4 Configure environment variables

1. Edit the root `.env` with your preferred editor:
   - VS Code: `code .env`
   - Notepad: `notepad .env`
   - WSL editor: `nano .env`
2. Update the same fields described in the macOS section (credentials, host paths, Historian Agent settings). When using Windows paths, either:
   - Use absolute Windows paths (e.g., `D:/HistoricalReader/mongo_data`). Docker will translate them.
   - Or keep the project inside WSL and use Linux-style paths (e.g., `/home/<user>/nosql_reader/mongo_data`).

### 3.5 Configure the UI JSON

Edit `app/config.json` just like on macOS. VS Code with the Remote - WSL extension is ideal if you cloned inside WSL; otherwise any JSON-aware editor works. Maintain both the flat and nested keys for each setting.

### 3.6 Start the stack

From the project root, run:

```bash
docker compose up --build
```

The web UI will be at [http://localhost:5001](http://localhost:5001). To stop the stack, press `Ctrl+C` or run `docker compose down` in a separate terminal. Container logs appear in your terminal—watch for messages indicating that the Flask app is running and MongoDB has initialised.

### 3.7 Add archives and media

- Copy JSON/JSONL files into the directory referenced by `ARCHIVES_HOST_PATH`.
- Maintain the same relative layout between JSON files and any media assets. For instance, if your JSON record references `images/frame_01.tif`, place that file at `ARCHIVES_HOST_PATH/images/frame_01.tif`.
- Trigger ingestion by running `docker compose exec app /app/bootstrap_data.sh` once the containers are up.

> **Note for Windows:** Avoid storing archives on network drives mounted via UNC paths (`\\server\share`). Docker volume mounts are faster and more reliable on local NTFS or within the WSL ext4 filesystem.

---

## 4. Verifying MongoDB connectivity

After the containers start, you can confirm MongoDB is reachable:

```bash
docker compose exec app python -c "
from database_setup import get_client, get_db
client = get_client()
db = get_db(client)
print('connected to', db.name)
"
```

A successful run prints the database name. If it fails, double-check credentials in `.env` and ensure the `mongodb` service container is healthy (`docker compose ps`).

You can also run the integrated smoke suite:

```bash
docker compose exec app python /app/util/e2e_cli_suite.py
```

---

## 5. Next steps

1. Visit [http://localhost:5001](http://localhost:5001) and explore the search UI and Historian Agent.
2. Upload archive files through the **Archive Files** page or place them directly in the host archive directory.
3. Customise the interface further by expanding `app/config.json` (e.g., add new colour keys) and adjusting templates under `app/templates/`.
4. Review `readme.md` for advanced operations like backups (`backup_db.py`, `restore_db.py`), bootstrapping scripts, and troubleshooting tips.

With these steps complete, the Historical Document Reader should be ready for use on both Mac and PC.
