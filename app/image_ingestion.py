"""Image-to-JSON ingestion pipeline utilities."""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

import data_processing
from database_setup import (
    get_client,
    get_db,
    update_field_structure,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.2-vision:11b"
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_PROMPT = """Perform OCR on the document to extract all text. Then, analyze the text to provide a summary of the content. Structure the extracted data into a STRICT JSON format. Use double quotes for strings. Remove trailing commas. Ensure all braces and brackets are correctly closed. Correctly structure nested elements. In the JSON arrays, each element should be separated by a comma. The document may contain multiple types of bureaucratic forms with printed or handwritten information. Each line of tabular information, if available, should be accurately categorized and linked to relevant information. Use null values for any categories where no relevant information is found. If any other information is necessary, include it in the "other" category. Return the extracted information as a JSON object with the following structure:
{
    "ocr_text": "string",
    "summary": "string",
    "sections": [
        {
            "section_name": "string",
            "fields": [
                {
                    "field_name": "string",
                    "value": "string",
                    "linked_information": {
                        "personal_information": {
                            "name": "string",
                            "date_of_birth": "string",
                            "social_security_no": "string",
                            "employee_no": "string",
                            "id_no": "string",
                            "rp_certificate_no": "string"
                        },
                        "named_entities": [
                            {
                                "entity": "string",
                                "type": "string"
                            }
                        ],
                        "dates": [
                            {
                                "date": "string",
                                "category": "string"
                            }
                        ],
                        "monetary_amounts": [
                            {
                                "amount": "string",
                                "category": "string"
                            }
                        ],
                        "relationships": [
                            {
                                "entity1": "string",
                                "relationship": "string",
                                "entity2": "string"
                            }
                        ],
                        "medical_information": [
                            {
                                "info": "string",
                                "category": "string"
                            }
                        ],
                        "family_information": [
                            {
                                "name": "string",
                                "relationship": "string",
                                "event": "string"
                            }
                        ],
                        "employment_history": [
                            {
                                "date": "string",
                                "position_description": "string",
                                "division": "string",
                                "department": "string",
                                "location": "string",
                                "rate_of_pay": "string",
                                "remarks": "string"
                            }
                        ],
                        "employment_events": [
                            {
                                "date": "string",
                                "location": "string",
                                "incident": "string",
                                "action_taken": "string",
                                "reference_numbers": [
                                    "string"
                                ]
                            }
                        ],
                        "geographical_information": [
                            "string"
                        ],
                        "document_specific_information": [
                            "string"
                        ],
                        "other": [
                            "string"
                        ],
                        "metadata": {
                            "document_type": "string",
                            "period": "string",
                            "context": "string"
                        },
                        "sentiment": "string"
                    }
                }
            ]
        }
    ]
}
"""

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
DEFAULT_OLLAMA_BASE_URL = os.environ.get("HISTORIAN_AGENT_OLLAMA_BASE_URL") or os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OPENAI_KEY_FILE_ENV = "OPENAI_API_KEY_FILE"
FALLBACK_KEY_PATH = Path.home() / ".config" / "nosql_reader" / "openai_api_key.txt"
DATA_MOUNT_ROOT = Path("/mnt")  # change: Dynamic bind mounts are projected into this directory tree inside the container.


@dataclass
class ModelConfig:
    """Configuration for a model invocation."""

    provider: str
    model: str
    prompt: str
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4000


@dataclass
class IngestionSummary:
    images_total: int = 0
    generated: int = 0
    skipped_existing: int = 0
    queued_existing: int = 0
    failed: int = 0
    ingested: int = 0
    updated: int = 0
    ingest_failures: int = 0
    errors: List[Dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def as_dict(self) -> Dict[str, object]:
        return {
            "images_total": self.images_total,
            "generated": self.generated,
            "skipped_existing": self.skipped_existing,
            "queued_existing": self.queued_existing,
            "failed": self.failed,
            "ingested": self.ingested,
            "updated": self.updated,
            "ingest_failures": self.ingest_failures,
            "errors": self.errors,
        }


class IngestionError(RuntimeError):
    """Raised when the ingestion pipeline encounters a fatal error."""

def ollama_models(base_url: Optional[str] = None) -> List[str]:
    """Return a list of available Ollama models from the local runtime."""

    endpoint = (base_url or DEFAULT_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/") + OLLAMA_TAGS_ENDPOINT
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [item.get("name") for item in models if item.get("name")]
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.warning("Failed to list Ollama models: %s", exc)
        return []


def _image_to_base64(path: Path) -> Tuple[str, str]:
    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "application/octet-stream")
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return encoded, mime


def _call_ollama(image_path: Path, config: ModelConfig) -> str:
    base_url = (config.base_url or DEFAULT_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    url = f"{base_url}{OLLAMA_CHAT_ENDPOINT}"
    encoded, mime = _image_to_base64(image_path)
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Process this document and return JSON only."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": encoded,
                        },
                    },
                ],
            },
        ],
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_ctx": 8192,
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message") or {}
    content = message.get("content")
    if not content:
        raise IngestionError("Ollama response did not contain any content.")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return str(content)

def _call_openai(image_path: Path, config: ModelConfig, api_key: str) -> str:
    from openai import OpenAI  # Imported lazily to keep optional dependency optional

    client = OpenAI(api_key=api_key)
    encoded, mime = _image_to_base64(image_path)
    image_url = f"data:{mime};base64,{encoded}"
    completion = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": config.prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Process this document and respond with JSON only."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    return completion.choices[0].message.content or ""


def _serialise_json(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = data_processing.clean_json(text)
        return json.loads(cleaned)


def _json_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".json")


def _archives_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _ensure_api_key_path() -> Path:
    configured = os.environ.get(OPENAI_KEY_FILE_ENV)
    if configured:
        return Path(configured).expanduser()
    return FALLBACK_KEY_PATH


def read_api_key() -> Optional[str]:
    path = _ensure_api_key_path()
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def write_api_key(value: str) -> Path:
    path = _ensure_api_key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip(), encoding="utf-8")
    return path


def _document_exists(db: Database, relative_path: str) -> bool:
    return db["documents"].find_one({"relative_path": relative_path}) is not None

def _ingest_json_documents(db: Database, json_paths: Iterable[Path], root: Path, summary: IngestionSummary) -> None:
    collection: Collection = db["documents"]
    unique_paths = []
    seen: set[Path] = set()
    for path in json_paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)

    if not unique_paths:
        return

    data_processing.root_directory = str(root)

    for json_path in unique_paths:
        try:
            json_data, error = data_processing.load_and_validate_json_file(str(json_path))
            if error or not json_data:
                summary.ingest_failures += 1
                summary.errors.append({
                    "path": str(json_path),
                    "error": error or "Unknown validation error",
                })
                continue

            relative_path = json_data.get("relative_path") or _archives_relative(json_path, root)
            json_data["relative_path"] = relative_path

            existing = collection.find_one({"relative_path": relative_path})
            if existing:
                collection.replace_one({"_id": existing["_id"]}, json_data)
                summary.updated += 1
            else:
                try:
                    collection.insert_one(json_data)
                    summary.ingested += 1
                except DuplicateKeyError:
                    collection.replace_one({"file_hash": json_data.get("file_hash")}, json_data, upsert=True)
                    summary.updated += 1

            update_field_structure(db, json_data)
        except Exception as exc:  # pragma: no cover - interacts with external db
            LOGGER.exception("Failed to ingest JSON %s", json_path)
            summary.ingest_failures += 1
            summary.errors.append({"path": str(json_path), "error": str(exc)})

def process_directory(
    directory: Path,
    config: ModelConfig,
    reprocess_existing: bool = False,
    api_key: Optional[str] = None,
) -> IngestionSummary:
    """Process images under ``directory`` and ingest generated JSON documents."""

    if not directory.exists() or not directory.is_dir():
        raise IngestionError(f"Directory does not exist: {directory}")

    summary = IngestionSummary()
    images: List[Path] = [
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    summary.images_total = len(images)

    if not images:
        return summary

    json_targets: List[Path] = []

    client = get_client()
    db = get_db(client)

    data_processing.root_directory = str(directory)
    data_processing.db = db  # type: ignore[attr-defined]

    for image_path in images:
        json_path = _json_path_for_image(image_path)
        relative_path = _archives_relative(json_path, directory)
        if json_path.exists() and not reprocess_existing:
            if _document_exists(db, relative_path):
                summary.skipped_existing += 1
                continue
            json_targets.append(json_path)
            summary.queued_existing += 1
            continue

        try:
            if config.provider == "ollama":
                output_text = _call_ollama(image_path, config)
            elif config.provider == "openai":
                if not api_key:
                    raise IngestionError("OpenAI API key is required for ChatGPT ingestion.")
                output_text = _call_openai(image_path, config, api_key)
            else:
                raise IngestionError(f"Unsupported provider: {config.provider}")

            payload = _serialise_json(output_text)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            json_targets.append(json_path)
            summary.generated += 1
        except Exception as exc:  # pragma: no cover - depends on external services
            LOGGER.exception("Failed to process image %s", image_path)
            summary.failed += 1
            summary.errors.append({"path": str(image_path), "error": str(exc)})

    _ingest_json_documents(db, json_targets, directory, summary)
    return summary


def expand_directory(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def provider_from_string(raw: Optional[str]) -> str:
    normalised = (raw or DEFAULT_PROVIDER).strip().lower()
    if normalised not in {"ollama", "openai"}:
        return DEFAULT_PROVIDER
    return normalised


def ensure_api_key(value: Optional[str]) -> Optional[str]:
    if value:
        write_api_key(value)
        return value.strip()
    return read_api_key()


def discover_all_mounts() -> List[Path]:
    """Return available bind mounts created by the dynamic mount helper."""

    mounts: List[Path] = []  # change: Collect valid directories so callers can iterate safely.
    if not DATA_MOUNT_ROOT.exists():
        return mounts  # change: Avoid raising when the helper has not created any mounts yet.
    for candidate in sorted(DATA_MOUNT_ROOT.iterdir()):
        if candidate.is_dir():
            mounts.append(candidate)  # change: Only expose directories that ingestion can traverse.
    return mounts


def run_ingestion_over_mounts(
    config: ModelConfig,
    reprocess_existing: bool = False,
    api_key: Optional[str] = None,
) -> IngestionSummary:
    """Process every mounted directory under ``/mnt`` sequentially."""

    combined = IngestionSummary()  # change: Aggregate per-mount results for a consolidated report.
    for mount_path in discover_all_mounts():
        try:
            summary = process_directory(
                mount_path,
                config,
                reprocess_existing=reprocess_existing,
                api_key=api_key,
            )
        except IngestionError as exc:
            LOGGER.error("Skipping mount %s due to ingestion error: %s", mount_path, exc)  # change: Log mount-level failures for troubleshooting.
            combined.errors.append({"path": str(mount_path), "error": str(exc)})  # change: Surface mount-level failures to callers.
            combined.ingest_failures += 1  # change: Track mounts we could not process.
            continue
        _merge_summaries(combined, summary)
    return combined


def _merge_summaries(target: IngestionSummary, addition: IngestionSummary) -> None:
    """Merge two ingestion summaries in place."""

    target.images_total += addition.images_total  # change: Aggregate the counts across mounts.
    target.generated += addition.generated  # change: Carry over generated counts from each mount.
    target.skipped_existing += addition.skipped_existing  # change: Combine skip totals for accurate reporting.
    target.queued_existing += addition.queued_existing  # change: Include queued existing JSON counts.
    target.failed += addition.failed  # change: Capture image-processing failures across mounts.
    target.ingested += addition.ingested  # change: Sum database insert counts.
    target.updated += addition.updated  # change: Aggregate update counts for previously ingested docs.
    target.ingest_failures += addition.ingest_failures  # change: Track JSON ingest failures from each run.
    if addition.errors:
        target.errors.extend(addition.errors)  # change: Preserve detailed error context for operators.
