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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Added helper to normalise the ingestion provider pulled from the environment.
def _env_provider_default() -> str:
    candidate = os.environ.get("HISTORIAN_AGENT_MODEL_PROVIDER", "ollama").strip().lower()  # Updated to respect .env default provider while keeping Ollama as the fallback.
    return candidate if candidate in {"ollama", "openai"} else "ollama"


DEFAULT_PROVIDER = _env_provider_default()  # Updated to source the default provider from .env so UI and backend stay aligned.
_SHARED_MODEL_DEFAULT = os.environ.get("HISTORIAN_AGENT_MODEL", "").strip()  # Added shared model lookup so both providers can reuse the same .env value.
DEFAULT_OLLAMA_MODEL = _SHARED_MODEL_DEFAULT or os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2-vision:latest")  # Updated Ollama default to honour the historian model override when present.
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", _SHARED_MODEL_DEFAULT or "gpt-4o-mini")  # Updated OpenAI default to fall back to historian model when provider switches.
_PROMPT_FALLBACK = """Perform OCR on the document to extract all text. Then, analyze the text to provide a summary of the content. Structure the extracted data into a STRICT JSON format. Use double quotes for strings. Remove trailing commas. Ensure all braces and brackets are correctly closed. Correctly structure nested elements. In the JSON arrays, each element should be separated by a comma. The document may contain multiple types of bureaucratic forms with printed or handwritten information. Each line of tabular information, if available, should be accurately categorized and linked to relevant information. Use null values for any categories where no relevant information is found. If any other information is necessary, include it in the "other" category. Return the extracted information as a JSON object with the following structure:
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
"""  # Added fallback constant so environment overrides can reuse the legacy prompt text.
DEFAULT_PROMPT = os.environ.get("HISTORIAN_AGENT_PROMPT") or _PROMPT_FALLBACK  # Updated prompt default to allow .env customisation without losing the existing template.

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
DEFAULT_OLLAMA_BASE_URL = os.environ.get("HISTORIAN_AGENT_OLLAMA_BASE_URL") or os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OPENAI_KEY_FILE_ENV = "OPENAI_API_KEY_FILE"
FALLBACK_KEY_PATH = Path.home() / ".config" / "nosql_reader" / "openai_api_key.txt"


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
    """
    Call Ollama API with detailed verbose logging.
    
    According to Ollama API docs, the correct format is:
    {
      "model": "llama3.2-vision",
      "messages": [
        {
          "role": "user",
          "content": "text prompt",
          "images": ["base64_string"]  # At message level, not nested
        }
      ]
    }
    """
    LOGGER.info("="*60)
    LOGGER.info(f"STARTING OLLAMA PROCESSING")
    LOGGER.info(f"  Image: {image_path.name}")
    LOGGER.info(f"  Full path: {image_path}")
    LOGGER.info(f"  Model: {config.model}")
    LOGGER.info(f"  Provider: {config.provider}")
    
    # Build URL
    base_url = (config.base_url or DEFAULT_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    url = f"{base_url}{OLLAMA_CHAT_ENDPOINT}"
    LOGGER.info(f"  Target URL: {url}")
    
    # Check if file exists and is readable
    if not image_path.exists():
        LOGGER.error(f"  ❌ FILE NOT FOUND: {image_path}")
        raise IngestionError(f"Image file not found: {image_path}")
    
    file_size = image_path.stat().st_size
    LOGGER.info(f"  ✅ File exists, size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Encode image
    LOGGER.info(f"  Encoding image to base64...")
    try:
        encoded, mime = _image_to_base64(image_path)
        encoded_size = len(encoded)
        LOGGER.info(f"  ✅ Image encoded: {encoded_size:,} characters")
        LOGGER.info(f"  MIME type: {mime}")
    except Exception as e:
        LOGGER.error(f"  ❌ ENCODING FAILED: {e}")
        raise IngestionError(f"Failed to encode image: {e}")
    
    # Build payload - CORRECT FORMAT per Ollama docs
    LOGGER.info(f"  Building API payload...")
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.prompt},
            {
                "role": "user",
                "content": "Process this document and return JSON only.",
                "images": [encoded]  # Images at message level per Ollama API docs
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": config.temperature,
            "num_ctx": 8192,
        },
    }
    
    LOGGER.info(f"  ✅ Payload built:")
    LOGGER.info(f"    - System prompt length: {len(config.prompt)} chars")
    LOGGER.info(f"    - User message: 'Process this document and return JSON only.'")
    LOGGER.info(f"    - Images array: 1 image ({encoded_size:,} chars)")
    LOGGER.info(f"    - Stream: False")
    LOGGER.info(f"    - Temperature: {config.temperature}")
    
    # Make API call
    LOGGER.info(f"  📡 Sending request to Ollama...")
    LOGGER.info(f"  (This may take 30-120 seconds for vision models)")
    
    try:
        import time
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=120)
        
        elapsed = time.time() - start_time
        LOGGER.info(f"  ⏱️  Request completed in {elapsed:.2f} seconds")
        LOGGER.info(f"  HTTP Status: {response.status_code}")
        
        # Log response details
        if response.status_code != 200:
            LOGGER.error(f"  ❌ HTTP ERROR: {response.status_code}")
            LOGGER.error(f"  Response body: {response.text[:500]}")
            response.raise_for_status()
        
        # Parse response
        LOGGER.info(f"  ✅ Got 200 OK response")
        result = response.json()
        
        # Log response structure
        LOGGER.info(f"  Response keys: {list(result.keys())}")
        
        message = result.get("message") or {}
        content = message.get("content")
        
        if not content:
            LOGGER.error(f"  ❌ NO CONTENT in response")
            LOGGER.error(f"  Full response: {result}")
            raise IngestionError("Ollama response did not contain any content.")
        
        content_length = len(str(content))
        LOGGER.info(f"  ✅ Got content: {content_length:,} characters")
        LOGGER.info(f"  Content preview: {str(content)[:200]}...")
        
        # Return content
        if isinstance(content, list):
            final = "".join(part.get("text", "") for part in content)
            LOGGER.info(f"  Content was list, joined to {len(final)} chars")
            return final
        
        LOGGER.info(f"  ✅ PROCESSING COMPLETE for {image_path.name}")
        LOGGER.info("="*60)
        return str(content)
        
    except requests.exceptions.Timeout:
        LOGGER.error(f"  ❌ REQUEST TIMEOUT (>120 seconds)")
        LOGGER.error(f"  Image: {image_path.name}")
        LOGGER.error(f"  This usually means:")
        LOGGER.error(f"    1. Model is still loading")
        LOGGER.error(f"    2. Image is too large")
        LOGGER.error(f"    3. System resources exhausted")
        raise IngestionError(f"Ollama request timed out for {image_path.name}")
        
    except requests.exceptions.ConnectionError as e:
        LOGGER.error(f"  ❌ CONNECTION ERROR")
        LOGGER.error(f"  Cannot reach: {url}")
        LOGGER.error(f"  Error: {e}")
        raise IngestionError(f"Cannot connect to Ollama at {url}")
        
    except requests.exceptions.HTTPError as e:
        LOGGER.error(f"  ❌ HTTP ERROR: {e}")
        LOGGER.error(f"  Status code: {response.status_code}")
        LOGGER.error(f"  Response: {response.text[:500]}")
        raise IngestionError(f"Ollama API error: {e}")
        
    except Exception as e:
        LOGGER.error(f"  ❌ UNEXPECTED ERROR: {type(e).__name__}")
        LOGGER.error(f"  Message: {e}")
        LOGGER.error(f"  Image: {image_path.name}")
        raise IngestionError(f"Unexpected error processing {image_path.name}: {e}")


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
    """Process images under directory with detailed verbose logging."""
    
    LOGGER.info("")
    LOGGER.info("🚀 " + "="*70)
    LOGGER.info("🚀 STARTING IMAGE INGESTION BATCH")
    LOGGER.info("🚀 " + "="*70)
    LOGGER.info(f"  Directory: {directory}")
    LOGGER.info(f"  Model: {config.model} ({config.provider})")
    LOGGER.info(f"  Reprocess existing: {reprocess_existing}")
    
    if not directory.exists() or not directory.is_dir():
        LOGGER.error(f"  ❌ Directory does not exist: {directory}")
        raise IngestionError(f"Directory does not exist: {directory}")
    
    LOGGER.info(f"  ✅ Directory exists")
    
    # Initialize summary
    summary = IngestionSummary()
    
    # Find all images
    LOGGER.info(f"  📁 Scanning for images...")
    images: List[Path] = [
        path for path in directory.rglob("*") 
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    summary.images_total = len(images)
    
    LOGGER.info(f"  ✅ Found {summary.images_total} images")
    if not images:
        LOGGER.warning(f"  ⚠️  No images found in {directory}")
        return summary
    
    # Log breakdown by extension
    from collections import Counter
    extensions = Counter(p.suffix.lower() for p in images)
    LOGGER.info(f"  📊 Image breakdown:")
    for ext, count in extensions.most_common():
        LOGGER.info(f"     {ext}: {count} files")
    
    json_targets: List[Path] = []
    
    # Initialize DB connection
    client = get_client()
    db = get_db(client)
    data_processing.root_directory = str(directory)
    data_processing.db = db
    
    LOGGER.info("")
    LOGGER.info("📝 Processing images...")
    LOGGER.info("-" * 70)
    
    # Process each image
    for idx, image_path in enumerate(images, 1):
        LOGGER.info("")
        LOGGER.info(f"[{idx}/{summary.images_total}] {image_path.name}")
        
        json_path = _json_path_for_image(image_path)
        relative_path = _archives_relative(json_path, directory)
        
        # Check if JSON already exists
        if json_path.exists():
            LOGGER.info(f"  📄 JSON file exists: {json_path.name}")
            
            if not reprocess_existing:
                # Check if in database
                if _document_exists(db, relative_path):
                    LOGGER.info(f"  ⏭️  Already in database, skipping")
                    summary.skipped_existing += 1
                    continue
                else:
                    LOGGER.info(f"  📥 Not in database, queuing for ingestion")
                    json_targets.append(json_path)
                    summary.queued_existing += 1
                    continue
            else:
                LOGGER.info(f"  🔄 Reprocessing (reprocess_existing=True)")
        else:
            LOGGER.info(f"  🆕 No JSON file, processing from scratch")
        
        # Process the image
        try:
            LOGGER.info(f"  🤖 Calling AI model...")
            
            if config.provider == "ollama":
                output_text = _call_ollama(image_path, config)
            elif config.provider == "openai":
                if not api_key:
                    raise IngestionError("OpenAI API key is required for ChatGPT ingestion.")
                output_text = _call_openai(image_path, config, api_key)
            else:
                raise IngestionError(f"Unsupported provider: {config.provider}")
            
            LOGGER.info(f"  ✅ AI processing complete")
            LOGGER.info(f"  💾 Writing JSON file...")
            
            # Parse and validate JSON
            payload = _serialise_json(output_text)
            LOGGER.info(f"  ✅ JSON validated")
            
            # Write JSON file
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
            LOGGER.info(f"  ✅ JSON saved: {json_path.name}")
            
            json_targets.append(json_path)
            summary.generated += 1
            
            LOGGER.info(f"  ✅ SUCCESS")
            
        except Exception as exc:
            LOGGER.exception(f"  ❌ FAILED: {exc}")
            summary.failed += 1
            summary.errors.append({
                "path": str(image_path), 
                "error": str(exc)
            })
    
    # Summary so far
    LOGGER.info("")
    LOGGER.info("="*70)
    LOGGER.info("📊 PROCESSING SUMMARY")
    LOGGER.info("="*70)
    LOGGER.info(f"  Total images: {summary.images_total}")
    LOGGER.info(f"  ✅ Generated new JSON: {summary.generated}")
    LOGGER.info(f"  📥 Queued existing JSON: {summary.queued_existing}")
    LOGGER.info(f"  ⏭️  Skipped (already in DB): {summary.skipped_existing}")
    LOGGER.info(f"  ❌ Failed: {summary.failed}")
    
    # Ingest JSON files into MongoDB
    LOGGER.info("")
    LOGGER.info("💾 Ingesting JSON into MongoDB...")
    LOGGER.info(f"  Files to ingest: {len(json_targets)}")
    
    if json_targets:
        _ingest_json_documents(db, json_targets, directory, summary)
        LOGGER.info(f"  ✅ Ingested: {summary.ingested}")
        LOGGER.info(f"  🔄 Updated: {summary.updated}")
        LOGGER.info(f"  ❌ Ingest failures: {summary.ingest_failures}")
    else:
        LOGGER.info(f"  ⏭️  Nothing to ingest")
    
    LOGGER.info("")
    LOGGER.info("="*70)
    LOGGER.info("🎉 BATCH COMPLETE")
    LOGGER.info("="*70)
    LOGGER.info(f"  Total: {summary.images_total} images")
    LOGGER.info(f"  Generated: {summary.generated}")
    LOGGER.info(f"  Ingested: {summary.ingested}")
    LOGGER.info(f"  Failed: {summary.failed}")
    LOGGER.info("="*70)
    LOGGER.info("")
    
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
