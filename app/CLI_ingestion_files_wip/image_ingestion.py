"""Image-to-JSON ingestion pipeline utilities."""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile  # Added to support temporary files for image preprocessing so large originals stay untouched.
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image, ImageOps  # Added to downscale and normalise images ahead of model ingestion to avoid GPU timeouts.
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


def _preprocess_image(image_path: Path, max_side: int = 1600) -> Path:  # Added helper to resize, normalise, and compress images pre-ingestion.
    try:  # Added guard so we gracefully fall back to the original image if preprocessing fails.
        with Image.open(image_path) as img:  # Added safe image open to ensure we can manipulate various colour profiles.
            rgb_img = img.convert("RGB")  # Added conversion to RGB to unify colour space expected by downstream models.
            enhanced_img = ImageOps.autocontrast(rgb_img)  # Added auto-contrast to boost OCR clarity before ingestion.
            enhanced_img.thumbnail((max_side, max_side))  # Added thumbnail resize so the longest side stays within GPU-friendly bounds.

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:  # Added temp file to hold the optimised JPEG for encoding.
                temp_path = Path(handle.name)  # Added Path wrapper for the temporary file to stay consistent with existing helpers.
            enhanced_img.save(temp_path, format="JPEG", optimize=True, quality=85)  # Added optimised save to cut VRAM spikes while preserving quality.
            return temp_path  # Added explicit return of the new optimised image so callers can encode the smaller asset.
    except Exception as exc:  # Added broad exception to avoid blocking ingestion if Pillow encounters an unsupported format.
        LOGGER.warning("Image preprocessing failed, using original asset: %s", exc)  # Added log to highlight when preprocessing is skipped.
        return image_path  # Added fallback to original image to maintain ingestion continuity on errors.



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
    
    response = client.chat(
        model='llama3.1:8b',
        messages=[{
            'role': 'user',
            'content': structuring_prompt
        }],
        options={
            'temperature': 0.0,
            'num_ctx': 8192,
        }
    )
    
    elapsed = time.time() - start
    json_content = response['message']['content']
    
    LOGGER.info(f"    ✅ Structuring complete in {elapsed:.2f}s ({len(json_content):,} chars)")
    return json_content


def _ocr_path_for_image(image_path: Path) -> Path:
    """Get intermediate OCR text file path."""
    return image_path.with_suffix(image_path.suffix + ".ocr.txt")


def process_directory(
    directory: Path,
    config: ModelConfig,
    reprocess_existing: bool = False,
    api_key: Optional[str] = None,
) -> IngestionSummary:
    """Process images with batched two-stage pipeline."""
    
    LOGGER.info("")
    LOGGER.info("🚀 " + "="*70)
    LOGGER.info("🚀 STARTING BATCHED TWO-STAGE IMAGE INGESTION")
    LOGGER.info("🚀 " + "="*70)
    LOGGER.info(f"  Directory: {directory}")
    LOGGER.info(f"  Model: {config.model} ({config.provider})")
    LOGGER.info(f"  Stage 1: Vision OCR | Stage 2: llama3.1:8b structuring")
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
    
    # ============================================================
    # STAGE 1: OCR - Process all images to text
    # ============================================================
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("📸 STAGE 1: BATCH OCR EXTRACTION")
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
        
        # Skip if OCR already done and we're not reprocessing
        if not reprocess_existing and ocr_path.exists():
            LOGGER.info(f"  📄 Reuse OCR: {image_path.name}")
            continue
        
        ocr_needed.append(image_path)
    
    LOGGER.info(f"  Need OCR for {len(ocr_needed)} images")
    
    # Run OCR on needed images
    for idx, image_path in enumerate(ocr_needed, 1):
        LOGGER.info(f"  [{idx}/{len(ocr_needed)}] {image_path.name}")
        
        try:
            if config.provider == "ollama":
                ocr_text = _call_ollama_stage1_ocr(image_path, config)
            elif config.provider == "openai":
                if not api_key:
                    raise IngestionError("OpenAI API key required")
                ocr_text = _call_openai(image_path, config, api_key)
            else:
                raise IngestionError(f"Unsupported provider: {config.provider}")
            
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
    LOGGER.info("🔧 STAGE 2: BATCH JSON STRUCTURING")
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
            
            if len(ocr_text.strip()) < 50:
                LOGGER.warning(f"    ⚠️  OCR text too short, skipping")
                summary.failed += 1
                continue
            
            # Structure with text model
            if config.provider == "ollama":
                json_content = _call_ollama_stage2_structure(ocr_text, config, image_path.name)
            elif config.provider == "openai":
                # For OpenAI, use the OCR text directly (already structured)
                json_content = ocr_text
            else:
                raise IngestionError(f"Unsupported provider: {config.provider}")
            
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


def _call_openai(image_path: Path, config: ModelConfig, api_key: str) -> str:
    from openai import OpenAI  # Imported lazily to keep optional dependency optional

    client = OpenAI(api_key=api_key)
    processed_path = _preprocess_image(image_path)  # Added preprocessing before OpenAI calls to mirror the Ollama protection.
    cleanup_path = processed_path if processed_path != image_path else None  # Added cleanup tracker for temporary optimised files.
    encoded, mime = _image_to_base64(processed_path)  # Added encoding of the optimised asset to reduce payload size for OpenAI.
    if cleanup_path and cleanup_path.exists():  # Added immediate cleanup since OpenAI payload is already prepared.
        cleanup_path.unlink(missing_ok=True)  # Added removal of the temporary JPEG once base64 encoding finishes.
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



def process_directory_streaming(
    directory: Path,
    config: ModelConfig,
    reprocess_existing: bool = False,
    api_key: Optional[str] = None,
):
    """Process directory and yield progress updates for SSE streaming."""

    if not directory.exists() or not directory.is_dir():
        raise IngestionError(f"Directory does not exist: {directory}")  # Added guard so streaming callers get an immediate failure.

    images: List[Path] = [
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]  # Added filtered listing so SSE mirrors the batch ingestion scope.

    yield {
        'type': 'scan_start',
        'total_images': len(images),
        'directory': str(directory)
    }  # Added initial payload so the UI knows how many files will be processed.

    client = get_client()  # Added DB bootstrap to reuse existing ingestion helpers.
    db = get_db(client)  # Added DB handle for duplicate detection and inserts.
    data_processing.root_directory = str(directory)  # Aligned data_processing helpers with the active directory.
    data_processing.db = db  # Stored DB reference for downstream validators expecting it.

    processed = 0  # Added counters so we can report totals incrementally.
    skipped = 0
    errors = 0

    for idx, image_path in enumerate(images, 1):
        yield {
            'type': 'image_start',
            'image': image_path.name,
            'index': idx,
            'total': len(images)
        }  # Added image start event to drive the live progress list.

        json_path = _json_path_for_image(image_path)  # Reused helper so file naming matches batch ingestion.
        relative_path = _archives_relative(json_path, directory)  # Added relative path reuse for DB lookups.

        if not reprocess_existing and _document_exists(db, relative_path):
            skipped += 1
            yield {
                'type': 'image_skip',
                'image': image_path.name,
                'reason': 'Already in database'
            }  # Added skip event so operators see which files were ignored.
            continue

        try:
            if json_path.exists():
                yield {
                    'type': 'image_info',
                    'image': image_path.name,
                    'message': 'Loading from JSON file'
                }  # Added info event when an existing JSON payload is reused.
            else:
                yield {
                    'type': 'image_processing',
                    'image': image_path.name,
                    'message': f'Calling {config.provider} model...'
                }  # Added processing event to indicate a model invocation is in flight.

                if config.provider == 'ollama':
                    output_text = _call_ollama(image_path, config)  # Reused provider-specific ingestion path.
                elif config.provider == 'openai':
                    if not api_key:
                        raise IngestionError('OpenAI API key is required for ChatGPT ingestion.')  # Added explicit API key guard to prevent silent failures.
                    output_text = _call_openai(image_path, config, api_key)  # Reused OpenAI ingestion path for parity.
                else:
                    raise IngestionError(f"Unknown provider: {config.provider}")  # Added defensive branch for unsupported providers.

                payload = _serialise_json(output_text)  # Added JSON normalisation so downstream ingestion receives valid data.
                json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )  # Added write step so subsequent ingestion steps can reuse the stored JSON.

            temp_summary = IngestionSummary()  # Added throwaway summary so existing ingest helper can run without refactor.
            _ingest_json_documents(db, [json_path], directory, temp_summary)  # Reused ingestion helper to keep DB operations consistent.
            processed += 1

            yield {
                'type': 'image_complete',
                'image': image_path.name,
                'processed': processed,
                'skipped': skipped,
                'errors': errors
            }  # Added completion event summarising running totals for the UI.

        except Exception as exc:
            errors += 1
            LOGGER.exception('Failed to process %s', image_path)  # Added logging to preserve existing troubleshooting signals.
            yield {
                'type': 'image_error',
                'image': image_path.name,
                'error': str(exc),
                'processed': processed,
                'skipped': skipped,
                'errors': errors
            }  # Added error event so failures appear inline during the stream.

    yield {
        'type': 'scan_complete',
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'total': len(images)
    }  # Added final summary event so the frontend can re-enable controls.

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
