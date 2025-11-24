"""Image-to-JSON ingestion pipeline utilities.
This cleaned-up version:

Moves all constants to the top - Including DEFAULT_OLLAMA_STAGE2_MODEL, file suffixes, and configuration values
Removes duplicate process_directory - Keeps the better two-stage implementation
Fixes missing _call_ollama() - Replaced with proper stage-specific functions
Implements proper reprocessing logic:
When reprocess_existing=False: Reuses existing OCR files
When reprocess_existing=True: Regenerates both OCR and JSON files
Separates Ollama and OpenAI processing into clear helper functions
Fixes temporary file cleanup - Consistent cleanup in finally blocks
Updates streaming function to support two-stage Ollama processing
Adds proper error handling with specific error messages for each stage
Maintains OCR files for debugging and avoiding repeated expensive calls
The architecture is now clear:

Ollama: Two-stage (Vision OCR → Text structuring)
OpenAI: Single-stage (Direct to JSON)
All configuration values are at the top for easy modification





"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image, ImageOps
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

import data_processing
from database_setup import (
    get_client,
    get_db,
    update_field_structure,
)

# ============================================================
# HELPER FOR PROVIDER DEFAULT (needed before constants)
# ============================================================

def _env_provider_default() -> str:
    candidate = os.environ.get("HISTORIAN_AGENT_MODEL_PROVIDER", "ollama").strip().lower()
    return candidate if candidate in {"ollama", "openai"} else "ollama"


# ============================================================
# CONSTANTS - All configuration at the top for easy updating
# ============================================================

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
OCR_FILE_SUFFIX = ".ocr.txt"
JSON_FILE_SUFFIX = ".json"

# Ollama configuration
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
DEFAULT_OLLAMA_BASE_URL = os.environ.get("HISTORIAN_AGENT_OLLAMA_BASE_URL") or os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
DEFAULT_OLLAMA_STAGE2_MODEL = "llama3.1:8b"  # Model for JSON structuring
OLLAMA_STAGE2_CONTEXT_SIZE = 8192

# OpenAI configuration
OPENAI_KEY_FILE_ENV = "OPENAI_API_KEY_FILE"
FALLBACK_KEY_PATH = Path.home() / ".config" / "nosql_reader" / "openai_api_key.txt"

# Model defaults
DEFAULT_PROVIDER = _env_provider_default()
_SHARED_MODEL_DEFAULT = os.environ.get("HISTORIAN_AGENT_MODEL", "").strip()
DEFAULT_OLLAMA_MODEL = _SHARED_MODEL_DEFAULT or os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2-vision:latest")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", _SHARED_MODEL_DEFAULT or "gpt-4o-mini")

# Image preprocessing
MAX_IMAGE_SIDE = 1600
JPEG_QUALITY = 85

# Validation thresholds
MIN_OCR_TEXT_LENGTH = 50

# Default prompt
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
"""
DEFAULT_PROMPT = os.environ.get("HISTORIAN_AGENT_PROMPT") or _PROMPT_FALLBACK


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _env_provider_default() -> str:
    candidate = os.environ.get("HISTORIAN_AGENT_MODEL_PROVIDER", "ollama").strip().lower()
    return candidate if candidate in {"ollama", "openai"} else "ollama"


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


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ollama_models(base_url: Optional[str] = None) -> List[str]:
    """Return a list of available Ollama models from the local runtime."""
    endpoint = (base_url or DEFAULT_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/") + OLLAMA_TAGS_ENDPOINT
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [item.get("name") for item in models if item.get("name")]
    except Exception as exc:
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


def _preprocess_image(image_path: Path, max_side: int = MAX_IMAGE_SIDE) -> Path:
    """Resize, normalise, and compress images pre-ingestion."""
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            enhanced_img = ImageOps.autocontrast(rgb_img)
            enhanced_img.thumbnail((max_side, max_side))

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                temp_path = Path(handle.name)
            enhanced_img.save(temp_path, format="JPEG", optimize=True, quality=JPEG_QUALITY)
            return temp_path
    except Exception as exc:
        LOGGER.warning("Image preprocessing failed, using original asset: %s", exc)
        return image_path


def _ocr_path_for_image(image_path: Path) -> Path:
    """Get intermediate OCR text file path."""
    return image_path.with_suffix(image_path.suffix + OCR_FILE_SUFFIX)


def _json_path_for_image(image_path: Path) -> Path:
    """Get JSON output file path."""
    return image_path.with_suffix(image_path.suffix + JSON_FILE_SUFFIX)


def _archives_relative(path: Path, root: Path) -> str:
    """Get relative path for database storage."""
    return str(path.relative_to(root))


def _serialise_json(text: str) -> Dict[str, object]:
    """Parse and clean JSON text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = data_processing.clean_json(text)
        return json.loads(cleaned)


def _document_exists(db: Database, relative_path: str) -> bool:
    """Check if document already exists in database."""
    return db["documents"].find_one({"relative_path": relative_path}) is not None


# ============================================================
# OLLAMA FUNCTIONS
# ============================================================

def _call_ollama_stage1_ocr(image_path: Path, config: ModelConfig) -> str:
    """Stage 1: Vision model extracts text only."""
    import ollama
    
    LOGGER.info(f"  📸 STAGE 1 (OCR): {image_path.name}")
    
    processed_path = _preprocess_image(image_path)
    cleanup_path = processed_path if processed_path != image_path else None
    
    try:
        base_url = config.base_url or DEFAULT_OLLAMA_BASE_URL or "http://localhost:11434"
        client = ollama.Client(host=base_url)
        
        import time
        start = time.time()
        
        response = client.chat(
            model=config.model,
            messages=[{
                'role': 'user',
                'content': """Extract ALL visible text from this document. 
Read every word, number, date, and handwritten note you can see.
Maintain the original layout and structure as much as possible.
Include form labels, field values, stamps, signatures - everything readable.
Return only the extracted text, no commentary.""",
                'images': [str(processed_path)]
            }]
        )
        
        elapsed = time.time() - start
        ocr_text = response['message']['content']
        
        LOGGER.info(f"    ✅ OCR complete in {elapsed:.2f}s ({len(ocr_text):,} chars)")
        return ocr_text
        
    except Exception as exc:
        LOGGER.error(f"Ollama stage 1 OCR failed: {exc}")
        raise IngestionError(f"Vision model OCR failed: {exc}")
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)


def _call_ollama_stage2_structure(ocr_text: str, config: ModelConfig, image_name: str) -> str:
    """Stage 2: Text model structures OCR into JSON."""
    import ollama
    
    LOGGER.info(f"  🔧 STAGE 2 (JSON): {image_name}")
    
    base_url = config.base_url or DEFAULT_OLLAMA_BASE_URL or "http://localhost:11434"
    client = ollama.Client(host=base_url)
    
    structuring_prompt = f"""{config.prompt}

Raw OCR text extracted from document:
{ocr_text}

Based on the above text, create a properly structured JSON response following the schema provided.
Return ONLY valid JSON, no markdown code blocks, no explanation."""

    import time
    start = time.time()
    
    try:
        response = client.chat(
            model=DEFAULT_OLLAMA_STAGE2_MODEL,
            messages=[{
                'role': 'user',
                'content': structuring_prompt
            }],
            options={
                'temperature': 0.0,
                'num_ctx': OLLAMA_STAGE2_CONTEXT_SIZE,
            }
        )
        
        elapsed = time.time() - start
        json_content = response['message']['content']
        
        LOGGER.info(f"    ✅ Structuring complete in {elapsed:.2f}s ({len(json_content):,} chars)")
        return json_content
        
    except Exception as exc:
        LOGGER.error(f"Ollama stage 2 structuring failed: {exc}")
        raise IngestionError(f"JSON structuring failed: {exc}")


# ============================================================
# OPENAI FUNCTIONS
# ============================================================

def _call_openai(image_path: Path, config: ModelConfig, api_key: str) -> str:
    """Single-stage OpenAI processing: image to structured JSON."""
    from openai import OpenAI
    
    LOGGER.info(f"  🤖 OpenAI processing: {image_path.name}")
    
    client = OpenAI(api_key=api_key)
    processed_path = _preprocess_image(image_path)
    cleanup_path = processed_path if processed_path != image_path else None
    
    try:
        encoded, mime = _image_to_base64(processed_path)
        image_url = f"data:{mime};base64,{encoded}"
        
        import time
        start = time.time()
        
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
        
        elapsed = time.time() - start
        result = completion.choices[0].message.content or ""
        
        LOGGER.info(f"    ✅ OpenAI complete in {elapsed:.2f}s ({len(result):,} chars)")
        return result
        
    except Exception as exc:
        LOGGER.error(f"OpenAI processing failed: {exc}")
        raise IngestionError(f"OpenAI API call failed: {exc}")
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)

# ============================================================
# API KEY MANAGEMENT
# ============================================================

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


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def _ingest_json_documents(db: Database, json_paths: Iterable[Path], root: Path, summary: IngestionSummary) -> None:
    """Ingest JSON documents into MongoDB."""
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
        except Exception as exc:
            LOGGER.exception("Failed to ingest JSON %s", json_path)
            summary.ingest_failures += 1
            summary.errors.append({"path": str(json_path), "error": str(exc)})


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_directory(
    directory: Path,
    config: ModelConfig,
    reprocess_existing: bool = False,
    api_key: Optional[str] = None,
) -> IngestionSummary:
    """Process images with batched two-stage pipeline for Ollama, single-stage for OpenAI."""
    
    LOGGER.info("")
    LOGGER.info("🚀 " + "="*70)
    LOGGER.info("🚀 STARTING IMAGE INGESTION BATCH")
    LOGGER.info("🚀 " + "="*70)
    LOGGER.info(f"  Directory: {directory}")
    LOGGER.info(f"  Provider: {config.provider}")
    LOGGER.info(f"  Model: {config.model}")
    if config.provider == "ollama":
        LOGGER.info(f"  Stage 2 Model: {DEFAULT_OLLAMA_STAGE2_MODEL}")
    LOGGER.info(f"  Reprocess existing: {reprocess_existing}")
    
    if not directory.exists() or not directory.is_dir():
        LOGGER.error(f"  ❌ Directory does not exist: {directory}")
        raise IngestionError(f"Directory does not exist: {directory}")
    
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
        LOGGER.warning(f"  ⚠️  No images found")
        return summary
    
    # Initialize DB
    client = get_client()
    db = get_db(client)
    data_processing.root_directory = str(directory)
    data_processing.db = db
    
    # Process based on provider
    if config.provider == "ollama":
        _process_ollama_two_stage(images, config, db, directory, reprocess_existing, summary)
    elif config.provider == "openai":
        if not api_key:
            raise IngestionError("OpenAI API key required")
        _process_openai_single_stage(images, config, api_key, db, directory, reprocess_existing, summary)
    else:
        raise IngestionError(f"Unsupported provider: {config.provider}")
    
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("🎉 BATCH COMPLETE")
    LOGGER.info("=" * 70)
    LOGGER.info(f"  Total: {summary.images_total} images")
    LOGGER.info(f"  Generated: {summary.generated}")
    LOGGER.info(f"  Queued: {summary.queued_existing}")
    LOGGER.info(f"  Skipped: {summary.skipped_existing}")
    LOGGER.info(f"  Ingested: {summary.ingested}")
    LOGGER.info(f"  Failed: {summary.failed}")
    LOGGER.info("=" * 70)
    
    return summary


def _process_ollama_two_stage(
    images: List[Path],
    config: ModelConfig,
    db: Database,
    directory: Path,
    reprocess_existing: bool,
    summary: IngestionSummary
) -> None:
    """Process images using Ollama's two-stage pipeline."""
    
    # ============================================================
    # STAGE 1: OCR - Process all images to text
    # ============================================================
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("📸 STAGE 1: BATCH OCR EXTRACTION (Ollama)")
    LOGGER.info("=" * 70)
    
    ocr_needed: List[Path] = []
    
    for image_path in images:
        ocr_path = _ocr_path_for_image(image_path)
        json_path = _json_path_for_image(image_path)
        
        # Skip if final JSON exists and we're not reprocessing
        if not reprocess_existing and json_path.exists():
            relative_path = _archives_relative(json_path, directory)
            if _document_exists(db, relative_path):
                LOGGER.info(f"  ⏭️  Skip: {image_path.name} (already in DB)")
                summary.skipped_existing += 1
                continue
        
        # Check if we need OCR
        if reprocess_existing or not ocr_path.exists():
            ocr_needed.append(image_path)
        else:
            LOGGER.info(f"  📄 Reuse OCR: {image_path.name}")
    
    LOGGER.info(f"  Need OCR for {len(ocr_needed)} images")
    
    # Run OCR on needed images
    for idx, image_path in enumerate(ocr_needed, 1):
        LOGGER.info(f"  [{idx}/{len(ocr_needed)}] {image_path.name}")
        
        try:
            ocr_text = _call_ollama_stage1_ocr(image_path, config)
            
            # Validate OCR text
            if len(ocr_text.strip()) < MIN_OCR_TEXT_LENGTH:
                LOGGER.warning(f"    ⚠️  OCR text too short ({len(ocr_text)} chars), skipping")
                summary.failed += 1
                continue
            
            # Write intermediate OCR file
            ocr_path = _ocr_path_for_image(image_path)
            ocr_path.write_text(ocr_text, encoding="utf-8")
            LOGGER.info(f"    💾 Saved OCR to {ocr_path.name}")
            
        except Exception as exc:
            LOGGER.exception(f"  ❌ OCR FAILED: {exc}")
            summary.failed += 1
            summary.errors.append({"path": str(image_path), "error": str(exc)})
    
    # ============================================================
    # STAGE 2: JSON Structuring - Process all OCR texts
    # ============================================================
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("🔧 STAGE 2: BATCH JSON STRUCTURING (Ollama)")
    LOGGER.info("=" * 70)
    
    # Find all OCR files that need structuring
    json_targets: List[Path] = []
    
    for image_path in images:
        ocr_path = _ocr_path_for_image(image_path)
        json_path = _json_path_for_image(image_path)
        
        # Skip if no OCR file
        if not ocr_path.exists():
            continue
        
        # Skip if JSON exists and we're not reprocessing
        if not reprocess_existing and json_path.exists():
            relative_path = _archives_relative(json_path, directory)
            if _document_exists(db, relative_path):
                continue
            else:
                # JSON exists but not in DB - queue for ingestion
                json_targets.append(json_path)
                summary.queued_existing += 1
                continue
        
        # Need to structure this one
        LOGGER.info(f"  Processing: {image_path.name}")
        
        try:
            # Read OCR text
            ocr_text = ocr_path.read_text(encoding="utf-8")
            
            if len(ocr_text.strip()) < MIN_OCR_TEXT_LENGTH:
                LOGGER.warning(f"    ⚠️  OCR text too short, skipping")
                summary.failed += 1
                continue
            
            # Structure with text model
            json_content = _call_ollama_stage2_structure(ocr_text, config, image_path.name)
            
            # Validate and write JSON
            payload = _serialise_json(json_content)
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            LOGGER.info(f"    ✅ JSON saved: {json_path.name}")
            
            json_targets.append(json_path)
            summary.generated += 1
            
        except Exception as exc:
            LOGGER.exception(f"  ❌ STRUCTURING FAILED: {exc}")
            summary.failed += 1
            summary.errors.append({"path": str(image_path), "error": str(exc)})
    
    # ============================================================
    # STAGE 3: MongoDB Ingestion
    # ============================================================
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("💾 STAGE 3: MONGODB INGESTION")
    LOGGER.info("=" * 70)
    LOGGER.info(f"  Files to ingest: {len(json_targets)}")
    
    if json_targets:
        _ingest_json_documents(db, json_targets, directory, summary)
        LOGGER.info(f"  ✅ Ingested: {summary.ingested}")
        LOGGER.info(f"  🔄 Updated: {summary.updated}")
        LOGGER.info(f"  ❌ Failures: {summary.ingest_failures}")


def _process_openai_single_stage(
    images: List[Path],
    config: ModelConfig,
    api_key: str,
    db: Database,
    directory: Path,
    reprocess_existing: bool,
    summary: IngestionSummary
) -> None:
    """Process images using OpenAI's single-stage pipeline."""
    
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("🤖 OPENAI SINGLE-STAGE PROCESSING")
    LOGGER.info("=" * 70)
    
    json_targets: List[Path] = []
    
    for idx, image_path in enumerate(images, 1):
        LOGGER.info(f"  [{idx}/{summary.images_total}] {image_path.name}")
        
        json_path = _json_path_for_image(image_path)
        relative_path = _archives_relative(json_path, directory)
        
        # Check if JSON already exists
        if json_path.exists():
            if not reprocess_existing:
                if _document_exists(db, relative_path):
                    LOGGER.info(f"    ⏭️  Already in database, skipping")
                    summary.skipped_existing += 1
                    continue
                else:
                    LOGGER.info(f"    📥 Not in database, queuing for ingestion")
                    json_targets.append(json_path)
                    summary.queued_existing += 1
                    continue
            else:
                LOGGER.info(f"    🔄 Reprocessing (reprocess_existing=True)")
        
        # Process the image
        try:
            output_text = _call_openai(image_path, config, api_key)
            
            # Parse and validate JSON
            payload = _serialise_json(output_text)
            
            # Write JSON file
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
            LOGGER.info(f"    ✅ JSON saved: {json_path.name}")
            
            json_targets.append(json_path)
            summary.generated += 1
            
        except Exception as exc:
            LOGGER.exception(f"  ❌ FAILED: {exc}")
            summary.failed += 1
            summary.errors.append({
                "path": str(image_path), #BREAK
                "error": str(exc)
            })
    
    # Ingest JSON files into MongoDB
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("💾 MONGODB INGESTION")
    LOGGER.info("=" * 70)
    LOGGER.info(f"  Files to ingest: {len(json_targets)}")
    
    if json_targets:
        _ingest_json_documents(db, json_targets, directory, summary)
        LOGGER.info(f"  ✅ Ingested: {summary.ingested}")
        LOGGER.info(f"  🔄 Updated: {summary.updated}")
        LOGGER.info(f"  ❌ Failures: {summary.ingest_failures}")


# ============================================================
# STREAMING FUNCTION (for real-time progress updates)
# ============================================================

def process_directory_streaming(
    directory: Path,
    config: ModelConfig,
    reprocess_existing: bool = False,
    api_key: Optional[str] = None,
):
    """Process directory and yield progress updates for SSE streaming."""

    if not directory.exists() or not directory.is_dir():
        raise IngestionError(f"Directory does not exist: {directory}")

    images: List[Path] = [
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    yield {
        'type': 'scan_start',
        'total_images': len(images),
        'directory': str(directory)
    }

    client = get_client()
    db = get_db(client)
    data_processing.root_directory = str(directory)
    data_processing.db = db

    processed = 0
    skipped = 0
    errors = 0

    for idx, image_path in enumerate(images, 1):
        yield {
            'type': 'image_start',
            'image': image_path.name,
            'index': idx,
            'total': len(images)
        }

        json_path = _json_path_for_image(image_path)
        relative_path = _archives_relative(json_path, directory)
        ocr_path = _ocr_path_for_image(image_path)

        # Check if we should skip
        if not reprocess_existing and _document_exists(db, relative_path):
            skipped += 1
            yield {
                'type': 'image_skip',
                'image': image_path.name,
                'reason': 'Already in database'
            }
            continue

        try:
            if config.provider == 'ollama':
                # Two-stage processing for Ollama
                
                # Stage 1: OCR
                if reprocess_existing or not ocr_path.exists():
                    yield {
                        'type': 'image_processing',
                        'image': image_path.name,
                        'message': 'Stage 1: Extracting text with vision model...'
                    }
                    
                    ocr_text = _call_ollama_stage1_ocr(image_path, config)
                    
                    if len(ocr_text.strip()) < MIN_OCR_TEXT_LENGTH:
                        raise IngestionError(f"OCR text too short ({len(ocr_text)} chars)")
                    
                    ocr_path.write_text(ocr_text, encoding="utf-8")
                else:
                    yield {
                        'type': 'image_info',
                        'image': image_path.name,
                        'message': 'Using existing OCR text'
                    }
                    ocr_text = ocr_path.read_text(encoding="utf-8")
                
                # Stage 2: JSON structuring
                yield {
                    'type': 'image_processing',
                    'image': image_path.name,
                    'message': f'Stage 2: Structuring with {DEFAULT_OLLAMA_STAGE2_MODEL}...'
                }
                
                json_content = _call_ollama_stage2_structure(ocr_text, config, image_path.name)
                payload = _serialise_json(json_content)
                
            elif config.provider == 'openai':
                # Single-stage processing for OpenAI
                if not api_key:
                    raise IngestionError('OpenAI API key is required')
                
                yield {
                    'type': 'image_processing',
                    'image': image_path.name,
                    'message': 'Processing with OpenAI...'
                }
                
                output_text = _call_openai(image_path, config, api_key)
                payload = _serialise_json(output_text)
                
            else:
                raise IngestionError(f"Unknown provider: {config.provider}")

            # Write JSON file
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            # Ingest into database
            temp_summary = IngestionSummary()
            _ingest_json_documents(db, [json_path], directory, temp_summary)
            processed += 1

            yield {
                'type': 'image_complete',
                'image': image_path.name,
                'processed': processed,
                'skipped': skipped,
                'errors': errors
            }

        except Exception as exc:
            errors += 1
            LOGGER.exception('Failed to process %s', image_path)
            yield {
                'type': 'image_error',
                'image': image_path.name,
                'error': str(exc),
                'processed': processed,
                'skipped': skipped,
                'errors': errors
            }

    yield {
        'type': 'scan_complete',
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'total': len(images)
    }


# ============================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# ============================================================

def expand_directory(path_str: str) -> Path:
    """Expand and resolve directory path."""
    return Path(path_str).expanduser().resolve()


def provider_from_string(raw: Optional[str]) -> str:
    """Normalize provider string."""
    normalised = (raw or DEFAULT_PROVIDER).strip().lower()
    if normalised not in {"ollama", "openai"}:
        return DEFAULT_PROVIDER
    return normalised


def ensure_api_key(value: Optional[str]) -> Optional[str]:
    """Ensure API key is available."""
    if value:
        write_api_key(value)
        return value.strip()
    return read_api_key()
                