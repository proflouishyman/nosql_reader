# File: routes.py
# Path: routes.py

from flask import (
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    flash,
    session,
    abort,
    Response,
    stream_with_context,  # Added for SSE streaming responses.
    send_file,
    Flask,
    has_request_context,
)
from functools import wraps
from main import app, cache
from database_setup import (
    get_client,
    get_db,
    get_collections,
    find_document_by_id,
    update_document,
    delete_document,
    get_field_structure
)
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import math
import json
import re
import logging
import time
from datetime import datetime, timedelta
import random
import csv
from io import StringIO, BytesIO
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple
import os
import uuid
import pymongo
from pathlib import Path
import zipfile

from historian_agent import (
    HistorianAgentConfig,
    HistorianAgentError,
    get_agent,
    reset_agent,
)

import image_ingestion
from util.mounts import get_mounted_paths, short_tree  # Added to support read-only ingestion mounts view.
from network import init_app as init_network
from network.config import NetworkConfig
from network.network_utils import get_network_documents_for_viewer

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a console handler (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create a file handler to log to routes.log
log_file_path = os.path.join(os.path.dirname(__file__), 'routes.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  # Set the level for the file handler
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Initialize database connection and collections
client = get_client()
db = get_db(client)
documents, unique_terms_collection, field_structure_collection = get_collections(db)
init_network(app)

# Hashed password (generate this using generate_password_hash('your_actual_password'))
ADMIN_PASSWORD_HASH = 'pbkdf2:sha256:260000$uxZ1Fkjt9WQCHwuN$ca37dfb41ebc26b19daf24885ebcd09f607cab85f92dcab13625627fd9ee902a'

# Login attempt tracking
MAX_ATTEMPTS = 5
LOCKOUT_TIME = 15 * 60  # 15 minutes in seconds
login_attempts = {}

HISTORIAN_HISTORY_TIMEOUT = 3600
HISTORIAN_HISTORY_MAX_TURNS = 20

ARCHIVE_ALLOWED_EXTENSIONS = {'.json', '.jsonl'}
ARCHIVE_LIST_LIMIT = 200


"""
Flask Routes for Three RAG Approaches
======================================

Add these routes to routes.py to support three different RAG query methods:
1. Basic Query Handler - Direct RAG with hybrid retrieval
2. Adversarial RAG - One-shot with monitoring
3. Tiered Agent - Confidence-based escalation with Tier 1/Tier 2

All routes maintain the same session/history pattern as existing historian_agent_query
"""
# ============================================================================
# AGENT HANDLER IMPORTS
# ============================================================================

from historian_agent.adversarial_rag import AdversarialRAGHandler
from historian_agent.iterative_adversarial_agent import build_agent_from_env
from historian_agent.rag_query_handler import RAGQueryHandler
from historian_agent.corpus_explorer import CorpusExplorer

# Global instances (initialized lazily)
_rag_handler = None
_adversarial_handler = None
_tiered_agent = None
_tier0_explorer = None
_last_exploration_report = None

def get_rag_handler():
    """Lazy initialization of RAGQueryHandler"""
    global _rag_handler
    if _rag_handler is None:
        _rag_handler = RAGQueryHandler()
    return _rag_handler

def get_adversarial_handler():
    """Lazy initialization of AdversarialRAGHandler"""
    global _adversarial_handler
    if _adversarial_handler is None:
        _adversarial_handler = AdversarialRAGHandler()
    return _adversarial_handler

def get_tiered_agent():
    """Lazy initialization of TieredHistorianAgent (via factory)."""
    global _tiered_agent
    if _tiered_agent is None:
        _tiered_agent = build_agent_from_env()
    return _tiered_agent


def get_tier0_explorer():
    """Lazy initialization of Tier 0 CorpusExplorer."""
    global _tier0_explorer
    if _tier0_explorer is None:
        _tier0_explorer = CorpusExplorer()
    return _tier0_explorer

# ============================================================================
# SHARED RAG HELPER FUNCTIONS
# ============================================================================

def sources_list_to_dict(sources: List[Dict[str, str]], search_id: str) -> Dict[str, Dict[str, str]]:
    """
    Convert internal list format to API dict format at boundary.
    
    This is the ONLY place where we convert from list to dict.
    
    Args:
        sources: List of source dicts [{"label": "Source 1", "id": "...", ...}]
        search_id: Search ID for building document URLs
        
    Returns:
        Dict formatted for API response
    """
    return {
        source["label"]: {
            "id": source["id"],
            "display_name": source["display_name"],
            "url": f"/document/{source['id']}?search_id={search_id}"
        }
        for source in sources
    }


def process_rag_query(
    method: str,
    question: str,
    conversation_id: str,
    history: List[Dict]
) -> Dict[str, Any]:
    """
    Unified RAG query processor for all three methods.
    
    Handles:
        - Dispatching to appropriate handler
        - Converting sources list to dict (API boundary)
        - Creating and caching search_id
        - Updating conversation history
        - Formatting response
    
    Args:
        method: 'basic', 'adversarial', or 'tiered'
        question: User's question
        conversation_id: Session ID
        history: Chat history
        
    Returns:
        Standardized API response dict ready for jsonify()
    """
    # Dispatch to appropriate handler
    if method == 'basic':
        handler = get_rag_handler()
        answer, metrics = handler.process_query(question, context="", label="BASIC_RAG")
        sources_list = metrics.get('sources', [])
        
    elif method == 'adversarial':
        handler = get_adversarial_handler()
        answer, latency, sources_list = handler.process_query(question)
        metrics = {
            'total_time': latency,
            'tokens': 0,
            'doc_count': len(sources_list)
        }
        
    elif method == 'tiered':
        agent = get_tiered_agent()
        result = agent.run(question)
        answer = result.get('answer', '')
        sources_list = result.get('sources', [])
        metrics = result.get('metrics', {})
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create search_id and cache ordered document IDs
    search_id = str(uuid.uuid4())
    ordered_ids = [source["id"] for source in sources_list]
    cache.set(f'search_{search_id}', ordered_ids, timeout=3600)
    
    # Convert sources list to dict format (API BOUNDARY)
    sources_dict = sources_list_to_dict(sources_list, search_id)
    
    # Update history
    history.append({'role': 'user', 'content': question})
    history.append({'role': 'assistant', 'content': answer, 'sources': sources_dict})
    if len(history) > HISTORIAN_HISTORY_MAX_TURNS * 2:
        history = history[-HISTORIAN_HISTORY_MAX_TURNS * 2:]
    
    # Return standardized response
    return {
        'conversation_id': conversation_id,
        'answer': answer,
        'sources': sources_dict,
        'search_id': search_id,
        'metrics': metrics,
        'history': history,
        'method': method
    }


def _format_tier0_summary(report: Dict[str, Any], focus_note: str) -> str:
    """Build a concise Tier 0 summary for the Historian Agent UI."""
    # Added helper to format Tier 0 exploration output for chat display.
    metadata = report.get("exploration_metadata", {})
    strategy = metadata.get("strategy", "unknown")
    documents_read = metadata.get("documents_read", 0)
    batches_processed = metadata.get("batches_processed", 0)
    duration_seconds = metadata.get("duration_seconds", 0) or 0
    research_lens = metadata.get("research_lens") or []
    notebook_path = report.get("notebook_path")
    questions = report.get("questions") or []

    summary_lines = [
        "Tier 0 corpus exploration complete.",
        f"Strategy: {strategy}. Documents read: {documents_read}. Batches: {batches_processed}. Duration: {float(duration_seconds):.1f}s.",
    ]
    if focus_note:
        summary_lines.append(f"Focus note: {focus_note}")
    if research_lens:
        summary_lines.append(f"Research lens: {', '.join([str(item) for item in research_lens])}")  # Added to confirm historian-specified priorities were applied to the run.
    if questions:
        summary_lines.append("Top research questions:")
        for item in questions[:5]:
            question_text = item.get("question") if isinstance(item, dict) else str(item)
            if question_text:
                summary_lines.append(f"- {question_text}")
    if notebook_path:
        summary_lines.append(f"Notebook saved to `{notebook_path}`.")

    return "\n".join(summary_lines)

# ============================================================================
# UPDATED: Basic RAG Query Handler (Standardized)
# ============================================================================
@app.route('/historian-agent/query-basic', methods=['POST'])
def historian_agent_query_basic():
    """Basic RAG query using direct hybrid retrieval."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    conversation_id = payload.get('conversation_id') or str(uuid.uuid4())
    refresh_requested = _is_truthy(payload.get('refresh'))
    
    history_key = _historian_history_cache_key(conversation_id)
    
    if refresh_requested:
        cache.delete(history_key)
        if not question:
            return jsonify({
                'conversation_id': str(uuid.uuid4()),
                'answer': '',
                'sources': {},
                'search_id': '',
                'metrics': {},
                'history': [],
            })
    
    if not question:
        return jsonify({'error': 'A question is required.'}), 400
    
    history = cache.get(history_key) or []
    
    try:
        result = process_rag_query('basic', question, conversation_id, history)
        cache.set(history_key, result['history'], timeout=HISTORIAN_HISTORY_TIMEOUT)
        return jsonify(result)
        
    except Exception as exc:
        app.logger.exception('Basic RAG query failed')
        return jsonify({'error': f'Basic RAG query failed: {str(exc)}'}), 500

@app.route('/historian-agent/query-adversarial', methods=['POST'])
def historian_agent_query_adversarial():
    """Adversarial RAG query with enhanced monitoring."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    conversation_id = payload.get('conversation_id') or str(uuid.uuid4())
    refresh_requested = _is_truthy(payload.get('refresh'))
    
    history_key = _historian_history_cache_key(conversation_id)
    
    if refresh_requested:
        cache.delete(history_key)
        if not question:
            return jsonify({
                'conversation_id': str(uuid.uuid4()),
                'answer': '',
                'sources': {},
                'search_id': '',
                'metrics': {},
                'history': [],
            })
    
    if not question:
        return jsonify({'error': 'A question is required.'}), 400
    
    history = cache.get(history_key) or []
    
    try:
        result = process_rag_query('adversarial', question, conversation_id, history)
        cache.set(history_key, result['history'], timeout=HISTORIAN_HISTORY_TIMEOUT)
        return jsonify(result)
        
    except Exception as exc:
        app.logger.exception('Adversarial RAG query failed')
        return jsonify({'error': f'Adversarial RAG query failed: {str(exc)}'}), 500


@app.route('/historian-agent/query-tiered', methods=['POST'])
def historian_agent_query_tiered():
    """Advanced tiered RAG with confidence-based escalation."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    conversation_id = payload.get('conversation_id') or str(uuid.uuid4())
    refresh_requested = _is_truthy(payload.get('refresh'))
    
    history_key = _historian_history_cache_key(conversation_id)
    
    if refresh_requested:
        cache.delete(history_key)
        if not question:
            return jsonify({
                'conversation_id': str(uuid.uuid4()),
                'answer': '',
                'sources': {},
                'search_id': '',
                'metrics': {},
                'history': [],
            })
    
    if not question:
        return jsonify({'error': 'A question is required.'}), 400
    
    history = cache.get(history_key) or []
    
    try:
        result = process_rag_query('tiered', question, conversation_id, history)
        cache.set(history_key, result['history'], timeout=HISTORIAN_HISTORY_TIMEOUT)
        return jsonify(result)
        
    except Exception as exc:
        app.logger.exception('Tiered agent query failed')
        return jsonify({'error': f'Tiered agent query failed: {str(exc)}'}), 500


@app.route('/historian-agent/query-tier0', methods=['POST'])
def historian_agent_query_tier0():
    """Tier 0 corpus exploration triggered from the Historian Agent UI."""
    # Added Tier 0 query handler to serve the new dropdown option in the Historian Agent UI.
    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    conversation_id = payload.get('conversation_id') or str(uuid.uuid4())
    refresh_requested = _is_truthy(payload.get('refresh'))

    history_key = _historian_history_cache_key(conversation_id)

    if refresh_requested:
        cache.delete(history_key)
        if not question:
            return jsonify({
                'conversation_id': str(uuid.uuid4()),
                'answer': '',
                'sources': {},
                'search_id': '',
                'metrics': {},
                'history': [],
            })

    if not question:
        return jsonify({'error': 'A question is required.'}), 400

    history = cache.get(history_key) or []

    try:
        strategy = payload.get('strategy')
        total_budget = payload.get('total_budget')
        year_range = payload.get('year_range')
        save_notebook = payload.get('save_notebook')
        research_lens = payload.get('research_lens', payload.get('focus_areas'))  # Added optional lens aliases so callers can bias Tier 0 toward historian-defined intersections.

        if isinstance(year_range, list) and len(year_range) == 2:
            try:
                year_range = (int(year_range[0]), int(year_range[1]))
            except Exception:
                year_range = None
        elif not isinstance(year_range, tuple):
            year_range = None

        explorer = get_tier0_explorer()
        report = explorer.explore(
            strategy=strategy,
            total_budget=total_budget,
            year_range=year_range,
            save_notebook=save_notebook,
            research_lens=research_lens,
        )

        global _last_exploration_report
        _last_exploration_report = report

        answer = _format_tier0_summary(report, question)
        metadata = report.get("exploration_metadata", {})
        metrics = {
            'total_time': float(metadata.get('duration_seconds') or 0),
            'doc_count': metadata.get('documents_read', 0),
            'batches_processed': metadata.get('batches_processed', 0),
            'questions': len(report.get('questions') or []),
            'patterns': len(report.get('patterns') or []),
            'entities': len(report.get('entities') or []),
            'contradictions': len(report.get('contradictions') or []),
        }  # Added Tier 0 metrics so the UI can surface exploration stats.

        search_id = ''  # Tier 0 does not return document sources to cache.
        sources_dict: Dict[str, Dict[str, str]] = {}

        history.append({'role': 'user', 'content': question})
        history.append({'role': 'assistant', 'content': answer, 'sources': sources_dict})
        if len(history) > HISTORIAN_HISTORY_MAX_TURNS * 2:
            history = history[-HISTORIAN_HISTORY_MAX_TURNS * 2:]

        result = {
            'conversation_id': conversation_id,
            'answer': answer,
            'sources': sources_dict,
            'search_id': search_id,
            'metrics': metrics,
            'history': history,
            'method': 'tier0',
        }
        cache.set(history_key, result['history'], timeout=HISTORIAN_HISTORY_TIMEOUT)
        return jsonify(result)

    except Exception as exc:
        app.logger.exception('Tier 0 query failed')
        return jsonify({'error': f'Tier 0 query failed: {str(exc)}'}), 500


@app.route('/historian-agent/reset-rag', methods=['POST'])
def reset_rag_handlers():
    """Reset and reinitialize all RAG handlers."""
    global _rag_handler, _adversarial_handler, _tiered_agent
    
    try:
        if _rag_handler: _rag_handler.close()
        if _adversarial_handler: _adversarial_handler.close()
        if _tiered_agent and hasattr(_tiered_agent, 'handler'): _tiered_agent.handler.close()
        
        _rag_handler = None
        _adversarial_handler = None
        _tiered_agent = None
        
        return jsonify({'message': 'All RAG handlers reset successfully', 'status': 'success'})
    except Exception as exc:
        app.logger.exception('Failed to reset RAG handlers')
        return jsonify({'error': f'Reset failed: {str(exc)}', 'status': 'error'}), 500


# ============================================================================
# TIER 0 CORPUS EXPLORATION ENDPOINTS
# ============================================================================

@app.route('/api/rag/explore_corpus', methods=['POST'])
def explore_corpus_endpoint():
    """Run Tier 0 corpus exploration (synchronous)."""
    payload = request.get_json(silent=True) or {}

    strategy = payload.get('strategy')
    total_budget = payload.get('total_budget')
    year_range = payload.get('year_range')
    save_notebook = payload.get('save_notebook')
    research_lens = payload.get('research_lens', payload.get('focus_areas'))  # Added optional lens aliases for API clients that prepopulate historian interests.

    if isinstance(year_range, list) and len(year_range) == 2:
        try:
            year_range = (int(year_range[0]), int(year_range[1]))
        except Exception:
            year_range = None
    elif not isinstance(year_range, tuple):
        year_range = None

    explorer = get_tier0_explorer()
    report = explorer.explore(
        strategy=strategy,
        total_budget=total_budget,
        year_range=year_range,
        save_notebook=save_notebook,
        research_lens=research_lens,
    )

    global _last_exploration_report
    _last_exploration_report = report

    return jsonify(report)


@app.route('/api/rag/exploration_report', methods=['GET'])
def exploration_report_endpoint():
    """Return the most recent Tier 0 exploration report, if available."""
    if _last_exploration_report is None:
        return jsonify({'error': 'No exploration report available'}), 404

    return jsonify(_last_exploration_report)

def _archives_root() -> Path:
    """Return the archive root directory, creating it if necessary."""

    root_path = Path(
        os.environ.get('ARCHIVES_PATH', os.path.join(app.root_path, 'archives'))
    ).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def _normalise_subdirectory(raw_subdir: Optional[str]) -> Path:
    """Sanitise a user-supplied subdirectory string for archive uploads."""

    if not raw_subdir:
        return Path()
    cleaned = Path(raw_subdir.strip().replace('\\', '/'))
    safe_parts = [
        secure_filename(part)
        for part in cleaned.parts
        if part not in {'', '.', '..'}
    ]
    return Path(*[part for part in safe_parts if part])


def _allowed_archive(filename: str) -> bool:
    """Return True if the provided filename has an allowed extension."""

    suffix = Path(filename).suffix.lower()
    return suffix in ARCHIVE_ALLOWED_EXTENSIONS


def _list_archives() -> Tuple[List[str], bool]:
    """Return archive files relative to the root and whether results were truncated."""

    root_path = _archives_root()
    files: List[str] = []
    for directory, _subdirs, filenames in os.walk(root_path):
        for name in sorted(filenames):
            relative = Path(directory, name).relative_to(root_path)
            files.append(str(relative))
            if len(files) > ARCHIVE_LIST_LIMIT:
                return files[:ARCHIVE_LIST_LIMIT], True
    return files, False


def _ingestion_default_config() -> Tuple[image_ingestion.ModelConfig, Optional[str]]:
    """Build a ``ModelConfig`` using the stored defaults for automated scans."""

    # Added helper so new mount-based endpoints reuse the existing ingestion defaults.
    provider = image_ingestion.DEFAULT_PROVIDER
    if provider == 'ollama':
        config = image_ingestion.ModelConfig(
            provider=provider,
            model=image_ingestion.DEFAULT_OLLAMA_MODEL,
            prompt=image_ingestion.DEFAULT_PROMPT,
            base_url=image_ingestion.DEFAULT_OLLAMA_BASE_URL,
        )
        api_key: Optional[str] = None
    else:
        api_key = image_ingestion.read_api_key()
        config = image_ingestion.ModelConfig(
            provider=provider,
            model=image_ingestion.DEFAULT_OPENAI_MODEL,
            prompt=image_ingestion.DEFAULT_PROMPT,
            base_url=None,
        )

    return config, api_key


def _blank_ingestion_totals() -> Dict[str, object]:
    """Return an accumulator dict for per-mount ingestion results."""

    # Added explicit structure so responses stay consistent between scan and rebuild endpoints.
    return {
        'images_total': 0,
        'generated': 0,
        'skipped_existing': 0,
        'queued_existing': 0,
        'failed': 0,
        'ingested': 0,
        'updated': 0,
        'ingest_failures': 0,
        'errors': [],
    }


def _merge_ingestion_totals(target: Dict[str, object], summary: Dict[str, object]) -> None:
    """Update ``target`` with numeric values from ``summary``."""

    # Added helper to share aggregation logic between the new scan and rebuild routes.
    for key in ['images_total', 'generated', 'skipped_existing', 'queued_existing', 'failed', 'ingested', 'updated', 'ingest_failures']:
        target[key] = int(target.get(key, 0)) + int(summary.get(key, 0))
    errors = list(target.get('errors', []))
    errors.extend(summary.get('errors', []) or [])
    target['errors'] = errors


# Added payload parser so both ingestion endpoints can accept client-supplied configuration.
def _ingestion_config_from_payload(payload: Dict[str, object]) -> Tuple[image_ingestion.ModelConfig, Optional[str], Optional[Tuple[Response, int]]]:
    provider = image_ingestion.provider_from_string(payload.get('provider', image_ingestion.DEFAULT_PROVIDER))  # Added provider normalisation for robustness.
    prompt = str(payload.get('prompt') or image_ingestion.DEFAULT_PROMPT)  # Added prompt fallback so old clients still work.
    base_url = payload.get('ollama_base_url') or payload.get('base_url') or image_ingestion.DEFAULT_OLLAMA_BASE_URL  # Added base URL merging so both keys are accepted.

    model_value = str(payload.get('model') or '').strip()  # Added local variable so we can reuse the submitted model.
    if provider == 'ollama':
        model = model_value or image_ingestion.DEFAULT_OLLAMA_MODEL  # Added fallback so Ollama still works without explicit selection.
    else:
        candidate = payload.get('openai_model') or model_value or image_ingestion.DEFAULT_OPENAI_MODEL  # Added OpenAI model precedence to favour provider-specific field.
        model = str(candidate).strip() or image_ingestion.DEFAULT_OPENAI_MODEL  # Added guard so whitespace does not blank the model.

    api_key: Optional[str] = None
    if provider == 'openai':
        api_key = image_ingestion.ensure_api_key(payload.get('api_key'))  # Added API key resolver so keys can be supplied or reused.
        if not api_key:
            error_body = jsonify({'status': 'error', 'message': 'Configure an OpenAI API key before running ingestion.', 'code': 'missing_api_key'})  # Added friendly error response when the key is absent.
            return image_ingestion.ModelConfig(provider=provider, model=model, prompt=prompt, base_url=None), None, (error_body, 400)  # Added early return so endpoints can stop processing.

    config = image_ingestion.ModelConfig(  # Added config assembly to share between scan and rebuild flows.
        provider=provider,
        model=model,
        prompt=prompt,
        base_url=(base_url or None) if provider == 'ollama' else None,
    )
    return config, api_key, None  # Added tuple return so callers get both config and optional API key.


def _process_archive_uploads(
    uploads: List[Any],
    subdirectory: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Save uploaded archive files and return (saved, rejected) lists."""

    archive_root = _archives_root()
    target_subdir = _normalise_subdirectory(subdirectory)
    saved_files: List[str] = []
    rejected_files: List[str] = []

    for file_storage in uploads:
        if not file_storage or not getattr(file_storage, 'filename', None):
            continue

        original_name = file_storage.filename
        filename = secure_filename(original_name)

        if not filename or not _allowed_archive(filename):
            rejected_files.append(original_name)
            continue

        destination = archive_root / target_subdir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_storage.save(destination)
        saved_files.append(str(destination.relative_to(archive_root)))

    return saved_files, rejected_files


@app.route('/data-files', methods=['GET', 'POST'])
def data_file_manager():
    """Allow users to upload additional archive files for data processing."""

    archive_root = _archives_root()

    if request.method == 'POST':
        uploads = request.files.getlist('archives')
        saved_files, rejected_files = _process_archive_uploads(
            uploads,
            request.form.get('target_subdir'),
        )

        if saved_files:
            display_subset = ', '.join(saved_files[:3])
            if len(saved_files) > 3:
                display_subset += ', …'
            flash(
                f"Uploaded {len(saved_files)} file(s): {display_subset}",
                'success',
            )
        else:
            flash(
                'No files were uploaded. Please choose JSON files with supported extensions (.json, .jsonl).',
                'warning',
            )

        if rejected_files:
            flash(f"Skipped unsupported files: {', '.join(rejected_files)}", 'warning')

        return redirect(url_for('data_file_manager'))

    archive_files, truncated = _list_archives()
    return render_template(
        'data_files.html',
        archive_root=str(archive_root),
        archive_files=archive_files,
        archive_list_truncated=truncated,
    )


@app.route('/data-files/download-all', methods=['GET'])
def download_all_archives():
    """Bundle the entire archive directory into a ZIP for download."""

    archive_root = _archives_root()
    files_to_include = [p for p in archive_root.rglob('*') if p.is_file()]

    if not files_to_include:
        flash('No archive files available to download yet.')
        return redirect(url_for('data_file_manager'))

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in files_to_include:
            arcname = file_path.relative_to(archive_root)
            bundle.write(file_path, arcname=str(arcname))

    zip_buffer.seek(0)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f'archives_{timestamp}.zip'

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename,
    )


def _historian_session_overrides() -> dict:
    """Return session-scoped Historian Agent overrides if available."""

    if not has_request_context():
        return {}
    stored = session.get('historian_agent_overrides')
    if isinstance(stored, dict):
        return stored.copy()
    return {}


def _historian_agent_overrides(include_session: bool = True) -> dict:
    """Combine config defaults with optional UI/session overrides."""

    settings = app.config.get('UI_CONFIG', {}).get('historian_agent', {})
    overrides: Dict[str, Any] = {}
    if isinstance(settings, dict):
        allowed = {
            'enabled',
            'model_provider',
            'model_name',
            'temperature',
            'max_context_documents',
            'system_prompt',
            'context_fields',
            'summary_field',
            'allow_general_fallback',
            'ollama_base_url',
            'openai_api_key',
            'use_vector_retrieval',  # Added to let persisted UI settings toggle RAG mode without env edits.
            'embedding_provider',  # Added to persist embedding backend choice from the settings form.
            'embedding_model',  # Added so the selected embedding model survives reloads.
            'chunk_size',  # Added to mirror UI chunk settings back into the agent config.
            'chunk_overlap',  # Added to persist overlap changes for downstream retrievers.
            'vector_store_type',  # Added to surface vector store selection through session overrides.
            'chroma_persist_directory',  # Added so custom Chroma paths can be reused across sessions.
            'hybrid_alpha',  # Added to keep hybrid weighting stable when users tweak sliders.
        }
        overrides.update({key: value for key, value in settings.items() if key in allowed})
    if include_session:
        overrides.update(_historian_session_overrides())
    return overrides


def _serialise_agent_config(config: HistorianAgentConfig) -> Dict[str, Any]:
    """Convert the agent configuration into JSON-safe primitives."""

    return {
        'enabled': config.enabled,
        'model_provider': config.model_provider,
        'model_name': config.model_name,
        'temperature': config.temperature,
        'max_context_documents': config.max_context_documents,
        'system_prompt': config.system_prompt,
        'context_fields': list(config.context_fields),
        'summary_field': config.summary_field,
        'allow_general_fallback': config.allow_general_fallback,
        'ollama_base_url': config.ollama_base_url or '',
        'openai_api_key_present': bool(config.openai_api_key),
        'use_vector_retrieval': config.use_vector_retrieval,  # Added to expose the RAG toggle to the UI payload.
        'embedding_provider': config.embedding_provider,  # Added so embedding provider dropdown hydrates correctly.
        'embedding_model': config.embedding_model,  # Added to prefill the embedding model input.
        'chunk_size': config.chunk_size,  # Added so chunk sizing controls reflect current config.
        'chunk_overlap': config.chunk_overlap,  # Added to keep overlap slider in sync with backend.
        'vector_store_type': config.vector_store_type,  # Added to mirror the selected vector store implementation.
        'chroma_persist_directory': config.chroma_persist_directory or '',  # Added to show custom Chroma path when set.
        'hybrid_alpha': config.hybrid_alpha,  # Added so hybrid weighting slider reports the active value.
    }


HISTORIAN_PROVIDER_OPTIONS = [
    {'value': 'ollama', 'label': 'Ollama (local)'},
    {'value': 'openai', 'label': 'OpenAI (cloud)'},
]


def _evaluate_agent_error(
    overrides: Optional[Dict[str, Any]] = None,
    config: Optional[HistorianAgentConfig] = None,
) -> Optional[str]:
    """Return a user-friendly error string if the Historian Agent is unavailable."""

    current_overrides = overrides if overrides is not None else _historian_agent_overrides()
    if config is None:
        config = HistorianAgentConfig.from_env(current_overrides)
    if not config.enabled:
        return None
    try:
        get_agent(documents, current_overrides)
    except HistorianAgentError as exc:
        return str(exc)
    except Exception:  # pragma: no cover - defensive logging
        app.logger.exception('Historian agent configuration check failed')
        return 'Historian agent is currently unavailable.'
    return None


def _historian_history_cache_key(conversation_id: str) -> str:
    """Return the cache key used to persist chat history for a conversation."""

    return f"historian_agent_history_{conversation_id}"


def _is_truthy(value) -> bool:
    """Coerce form inputs and JSON payload values into booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return False

def is_locked_out(ip):
    if ip in login_attempts:
        attempts, last_attempt_time = login_attempts[ip]
        if attempts >= MAX_ATTEMPTS:
            if datetime.now() - last_attempt_time < timedelta(seconds=LOCKOUT_TIME):
                return True
            else:
                login_attempts[ip] = (0, datetime.now())
    return False

def update_login_attempts(ip, success):
    if ip in login_attempts:
        attempts, _ = login_attempts[ip]
        if success:
            login_attempts[ip] = (0, datetime.now())
        else:
            login_attempts[ip] = (attempts + 1, datetime.now())
    else:
        login_attempts[ip] = (0, datetime.now()) if success else (1, datetime.now())

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
# @login_required
def index():
    app.logger.info('Rendering home page')
    return render_template('home.html')


@app.route('/search')
# @login_required
def search_database():
    app.logger.info('Rendering search interface')
    num_search_fields = 3
    field_structure = get_field_structure(db)
    return render_template(
        'search.html',
        num_search_fields=num_search_fields,
        field_structure=field_structure,
    )

@app.route('/api/dropdown-fields', methods=['GET'])
def get_dropdown_fields():
    """
    Return fields that should render as dropdowns (2-10 unique values).
    Frontend fetches this to dynamically create dropdown menus.
    """
    try:
        # For now, return hardcoded dropdown fields
        # In the future, this could be generated dynamically from the database
        dropdown_fields = {
            "collection": {
                "values": ["Microfilm Digitization", "Relief Record Scans"]
            },
            "archive_structure.physical_box": {
                "values": [
                    "Blue Box 128", 
                    "Blue Box 129", 
                    "Relief Department Box 4", 
                    "Relief Department Box 5", 
                    "Relief Department Box 6"
                ]
            }
        }
        
        logger.debug(f"Returning dropdown fields: {dropdown_fields}")
        return jsonify({"dropdown_fields": dropdown_fields})
        
    except Exception as e:
        logger.error(f"Error fetching dropdown fields: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to fetch dropdown fields",
            "dropdown_fields": {}
        }), 500

@app.route('/historian-agent')
def historian_agent_page():
    """Render the Historian Agent interface with the current configuration."""

    refresh = _is_truthy(request.args.get('refresh'))
    if refresh:
        reset_agent()
    overrides = _historian_agent_overrides()
    agent_config = HistorianAgentConfig.from_env(overrides)
    agent_error = _evaluate_agent_error(overrides=overrides, config=agent_config)

    return render_template(
        'historian_agent.html',
        agent_enabled=agent_config.enabled and agent_error is None,
        agent_config_payload=_serialise_agent_config(agent_config),
        provider_options=HISTORIAN_PROVIDER_OPTIONS,
        agent_error=agent_error,
    )


def _parse_agent_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise posted Historian Agent configuration values."""

    overrides: Dict[str, Any] = {}
    if 'enabled' in payload:
        overrides['enabled'] = _is_truthy(payload['enabled'])
    if 'model_provider' in payload:
        provider = str(payload['model_provider']).strip().lower()
        allowed = {option['value'] for option in HISTORIAN_PROVIDER_OPTIONS}
        if provider and provider not in allowed:
            raise ValueError('Unsupported model provider selected.')
        if provider:
            overrides['model_provider'] = provider
    if 'model_name' in payload:
        model_name = str(payload['model_name']).strip()
        if model_name:
            overrides['model_name'] = model_name
        else:
            overrides['model_name'] = ''
    if 'temperature' in payload and payload['temperature'] != '':
        try:
            temperature = float(payload['temperature'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Temperature must be a number.') from exc
        overrides['temperature'] = temperature
    if 'max_context_documents' in payload and payload['max_context_documents'] != '':
        try:
            max_docs = int(payload['max_context_documents'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Context documents must be a whole number.') from exc
        if max_docs < 1:
            raise ValueError('Context documents must be at least 1.')
        overrides['max_context_documents'] = max_docs
    if 'system_prompt' in payload:
        overrides['system_prompt'] = str(payload['system_prompt'])
    if 'context_fields' in payload:
        raw_fields = payload['context_fields']
        items: List[str] = []
        if isinstance(raw_fields, str):
            for value in re.split(r'[\n,]+', raw_fields):
                value = value.strip()
                if value:
                    items.append(value)
        elif isinstance(raw_fields, list):
            for value in raw_fields:
                if not isinstance(value, str):
                    continue
                value = value.strip()
                if value:
                    items.append(value)
        overrides['context_fields'] = items if items else None
    if 'summary_field' in payload:
        overrides['summary_field'] = str(payload['summary_field']).strip()
    if 'allow_general_fallback' in payload:
        overrides['allow_general_fallback'] = _is_truthy(payload['allow_general_fallback'])
    if 'ollama_base_url' in payload:
        base_url = str(payload['ollama_base_url']).strip()
        overrides['ollama_base_url'] = base_url or None
    if 'openai_api_key' in payload:
        api_key = str(payload['openai_api_key']).strip()
        overrides['openai_api_key'] = api_key or None
    if 'use_vector_retrieval' in payload:
        overrides['use_vector_retrieval'] = _is_truthy(payload['use_vector_retrieval'])  # Added to accept the UI RAG toggle.
    if 'embedding_provider' in payload:
        provider = str(payload['embedding_provider']).strip().lower()
        if provider and provider not in {'local', 'huggingface', 'openai'}:
            raise ValueError('Unsupported embedding provider selected.')  # Added guard to prevent invalid providers from breaking RAG init.
        overrides['embedding_provider'] = provider or None  # Added to persist supported provider choices.
    if 'embedding_model' in payload:
        model = str(payload['embedding_model']).strip()
        overrides['embedding_model'] = model  # Added to carry embedding model selection from the form.
    if 'chunk_size' in payload and payload['chunk_size'] != '':
        try:
            chunk_size = int(payload['chunk_size'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Chunk size must be a whole number.') from exc  # Added validation to keep chunk sizing numeric.
        if chunk_size < 1:
            raise ValueError('Chunk size must be at least 1.')  # Added lower bound to avoid empty chunks.
        overrides['chunk_size'] = chunk_size  # Added to save validated chunk size.
    if 'chunk_overlap' in payload and payload['chunk_overlap'] != '':
        try:
            chunk_overlap = int(payload['chunk_overlap'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Chunk overlap must be a whole number.') from exc  # Added validation to keep overlap numeric.
        if chunk_overlap < 0:
            raise ValueError('Chunk overlap cannot be negative.')  # Added guard so overlap stays sane.
        overrides['chunk_overlap'] = chunk_overlap  # Added to persist validated overlap.
    if 'vector_store_type' in payload:
        store_type = str(payload['vector_store_type']).strip().lower()
        if store_type and store_type not in {'chroma'}:
            raise ValueError('Unsupported vector store selected.')  # Added validation to match backend factory support.
        overrides['vector_store_type'] = store_type or None  # Added to save the chosen vector store type.
    if 'chroma_persist_directory' in payload:
        directory = str(payload['chroma_persist_directory']).strip()
        overrides['chroma_persist_directory'] = directory or None  # Added to pass custom persistence path through the config.
    if 'hybrid_alpha' in payload and payload['hybrid_alpha'] != '':
        try:
            hybrid_alpha = float(payload['hybrid_alpha'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Hybrid alpha must be a number.') from exc  # Added validation to keep weighting numeric.
        if not 0 <= hybrid_alpha <= 1:
            raise ValueError('Hybrid alpha must be between 0 and 1.')  # Added bound to align with retriever expectations.
        overrides['hybrid_alpha'] = hybrid_alpha  # Added to store validated weighting.
    return overrides


@app.route('/historian-agent/config', methods=['GET', 'POST'])
def historian_agent_config():
    """Return or mutate the active Historian Agent configuration."""

    if request.method == 'GET':
        config = HistorianAgentConfig.from_env(_historian_agent_overrides())
        return jsonify(
            {
                'config': _serialise_agent_config(config),
                'provider_options': HISTORIAN_PROVIDER_OPTIONS,
                'agent_error': _evaluate_agent_error(config=config),
            }
        )

    payload = request.get_json(silent=True) or {}
    if _is_truthy(payload.get('reset')):
        session.pop('historian_agent_overrides', None)
        session.modified = True
        reset_agent()
        config = HistorianAgentConfig.from_env(_historian_agent_overrides())
        return jsonify(
            {
                'config': _serialise_agent_config(config),
                'message': 'Historian Agent configuration reset to defaults.',
                'agent_error': _evaluate_agent_error(config=config),
            }
        )

    try:
        overrides = _parse_agent_config_payload(payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    session_overrides = _historian_session_overrides()
    for key, value in overrides.items():
        if value in (None, ''):
            session_overrides.pop(key, None)
        else:
            session_overrides[key] = value
    session['historian_agent_overrides'] = session_overrides
    session.modified = True
    reset_agent()
    config = HistorianAgentConfig.from_env(_historian_agent_overrides())
    return jsonify(
        {
            'config': _serialise_agent_config(config),
            'message': 'Historian Agent configuration updated.',
            'agent_error': _evaluate_agent_error(config=config),
        }
    )


@app.route('/historian-agent/query', methods=['POST'])
def historian_agent_query():
    """Execute a Historian Agent turn and persist the resulting chat history."""

    payload = request.get_json(silent=True) or {}
    question = (payload.get('question') or '').strip()
    conversation_id = payload.get('conversation_id') or str(uuid.uuid4())
    refresh_requested = _is_truthy(payload.get('refresh'))
    history_key = _historian_history_cache_key(conversation_id)
    if refresh_requested:
        reset_agent()
        cache.delete(history_key)
        if not question:
            return jsonify({
                'conversation_id': str(uuid.uuid4()),
                'answer': '',
                'sources': [],
                'history': [],
            })

    if not question:
        return jsonify({'error': 'A question is required.'}), 400

    history = cache.get(history_key) or []

    try:
        agent = get_agent(documents, _historian_agent_overrides())
        result = agent.invoke(question, chat_history=history)
    except HistorianAgentError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        app.logger.exception('Historian agent query failed')
        return jsonify({'error': 'Historian agent failed to generate a response.'}), 500

    history.append({'role': 'user', 'content': question})
    history.append({'role': 'assistant', 'content': result['answer']})
    if len(history) > HISTORIAN_HISTORY_MAX_TURNS * 2:
        history = history[-HISTORIAN_HISTORY_MAX_TURNS * 2:]
    cache.set(history_key, history, timeout=HISTORIAN_HISTORY_TIMEOUT)

    response_payload = {
        'conversation_id': conversation_id,
        'answer': result['answer'],
        'sources': result['sources'],
        'history': history,
    }
    return jsonify(response_payload)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        ip = request.remote_addr

        if is_locked_out(ip):
            flash('Too many failed attempts. Please try again later.')
            return render_template('login.html')

        # Verify CAPTCHA
        user_captcha = request.form.get('captcha')
        correct_captcha = request.form.get('captcha_answer')
        if user_captcha != correct_captcha:
            flash('Incorrect CAPTCHA')
            return redirect(url_for('login'))

        if check_password_hash(ADMIN_PASSWORD_HASH, request.form['password']):
            session['logged_in'] = True
            update_login_attempts(ip, success=True)
            flash('You were successfully logged in')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            update_login_attempts(ip, success=False)
            time.sleep(2)  # Add a delay after failed attempt
            flash('Invalid password')

    # Generate CAPTCHA for GET requests
    captcha_num1 = random.randint(1, 10)
    captcha_num2 = random.randint(1, 10)
    captcha_answer = str(captcha_num1 + captcha_num2)

    return render_template('login.html', captcha_num1=captcha_num1, captcha_num2=captcha_num2, captcha_answer=captcha_answer)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('index'))


# === NEW FUNCTIONS ADDED TO routes.py START ===

def load_dropdown_fields_from_config():
    """Load dropdown configuration from existing config.json."""
    try:
        # Load the main config file
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract dropdown fields from the config
        dropdown_fields = config.get('dropdown_fields', {})
        return dropdown_fields
        
    except Exception as e:
        # Fallback to hardcoded values if config loading fails
        print(f"Warning: Could not load dropdown config - {e}")
        return {
            "collection": {
                "values": ["Microfilm Digitization", "Relief Record Scans"],
                "min_values": 2,
                "max_values": 10,
                "count": 2
            },
            "archive_structure.physical_box": {
                "values": [
                    "Blue Box 128", 
                    "Blue Box 129", 
                    "Relief Department Box 4", 
                    "Relief Department Box 5", 
                    "Relief Department Box 6"
                ],
                "min_values": 2,
                "max_values": 10,
                "count": 7
            }
        }


# In routes.py - Add to your existing file (around line ~2035 or end of file)

@app.route('/api/searchable-fields')
def get_searchable_fields():
    """Return all fields that should appear in search dropdown."""
    # This returns ALL fields for selection, but we'll show which ones can use dropdowns
    return jsonify({
        "fields": ["collection", "archive_structure.physical_box"]  # Fields that qualify for dropdowns
    })






@app.route('/search', methods=['POST'])
# @login_required
def search():
    try:
        data = request.get_json()
        logger.debug(f"Received search request: {data}")

        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 50))

        query = build_query(data)
        logger.debug(f"Constructed MongoDB query: {query}")

        total_count = documents.count_documents(query)
        search_results = list(documents.find(query).skip((page - 1) * per_page).limit(per_page))

        for doc in search_results:
            doc['_id'] = str(doc['_id'])

        total_pages = math.ceil(total_count / per_page) if per_page else 1

        # Generate unique search ID
        search_id = str(uuid.uuid4())
        # Store the ordered list of document IDs
        ordered_ids = [doc['_id'] for doc in search_results]
        cache.set(f'search_{search_id}', ordered_ids, timeout=3600)  # Expires in 1 hour

        logger.debug(f"Search ID: {search_id}, Found {total_count} documents.")

        return jsonify({
            "search_id": search_id,
            "documents": search_results,
            "total_count": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page
        })

    except Exception as e:
        logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

def build_query(data):
    query = {}
    criteria_list = []

    logger.debug(f"Building query from search data: {data}")

    for i in range(1, 4):
        field = data.get(f'field{i}')
        search_term = data.get(f'searchTerm{i}')
        operator = data.get(f'operator{i}')

        if field and search_term:
            condition = {}
            if operator == 'NOT':
                condition[field] = {'$not': {'$regex': re.escape(search_term), '$options': 'i'}}
            else:
                condition[field] = {'$regex': re.escape(search_term), '$options': 'i'}
            
            criteria_list.append((operator, condition))
            logger.debug(f"Processed field {field} with search term '{search_term}' and operator '{operator}'")

    if criteria_list:
        and_conditions = []
        or_conditions = []

        for operator, condition in criteria_list:
            if operator == 'AND' or operator == 'NOT':
                and_conditions.append(condition)
            elif operator == 'OR':
                or_conditions.append(condition)

        if and_conditions:
            query['$and'] = and_conditions

        if or_conditions:
            if '$or' not in query:
                query['$or'] = or_conditions
            else:
                query['$or'].extend(or_conditions)

    logger.debug(f"Final query: {query}")
    return query


def _normalize_entity_type(raw_type: Any) -> str:
    """Normalize entity category labels for consistent grouping."""
    if raw_type is None:
        return "UNKNOWN"
    normalized = str(raw_type).strip()
    if not normalized:
        return "UNKNOWN"
    normalized = normalized.upper().replace("-", "_").replace(" ", "_")

    type_aliases = {
        "COMPANY": "ORG",
        "ORGANIZATION": "ORG",
        "ORGANISATION": "ORG",
        "LOCATION": "GPE",
        "PLACE": "GPE",
        "PERSON_NAME": "PERSON",
        "EMPLOYEEID": "EMPLOYEE_ID",
    }
    return type_aliases.get(normalized, normalized)


def _extract_entity_text(raw_value: Any) -> Optional[str]:
    """Extract a displayable entity string from scalar or mapping values."""
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return text or None

    if isinstance(raw_value, (int, float)):
        return str(raw_value)

    if isinstance(raw_value, dict):
        for key in ("text", "term", "entity", "name", "value"):
            value = raw_value.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
            elif isinstance(value, (int, float)):
                return str(value)

    return None


def build_document_ner_groups(document: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Build categorized NER groups for document detail display.
    Supports both legacy extracted_entities payloads and newer entities/entity_refs fields.
    """
    entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
    seen_by_type: Dict[str, Dict[str, int]] = {}
    empty_tokens = {"", "N/A", "NONE", "NULL"}

    def add_entity(
        raw_type: Any,
        raw_value: Any,
        entity_id: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> None:
        entity_type = _normalize_entity_type(raw_type)
        entity_text = _extract_entity_text(raw_value)
        if not entity_text:
            return

        if entity_text.strip().upper() in empty_tokens:
            return

        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
            seen_by_type[entity_type] = {}

        key = entity_text.casefold()
        if key in seen_by_type[entity_type]:
            # Prefer resolved records with entity_id over unresolved duplicates.
            existing_idx = seen_by_type[entity_type][key]
            existing = entities_by_type[entity_type][existing_idx]
            if entity_id and not existing.get("entity_id"):
                existing["entity_id"] = entity_id
                existing["canonical_name"] = canonical_name or entity_text
            return

        entities_by_type[entity_type].append(
            {
                "text": entity_text,
                "entity_id": entity_id,
                "canonical_name": canonical_name or entity_text,
                "type": entity_type,
            }
        )
        seen_by_type[entity_type][key] = len(entities_by_type[entity_type]) - 1

    def collect_named_entities(node: Any) -> None:
        """Recursively collect embedded named_entities payloads from section-linked metadata."""
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "named_entities" and isinstance(value, list):
                    for entity in value:
                        if isinstance(entity, dict):
                            add_entity(
                                entity.get("type") or entity.get("entity_type") or entity.get("label"),
                                entity.get("entity") or entity.get("text") or entity.get("name") or entity,
                            )
                        else:
                            add_entity("UNKNOWN", entity)
                else:
                    collect_named_entities(value)
        elif isinstance(node, list):
            for item in node:
                collect_named_entities(item)

    entities_payload = document.get("entities")
    if isinstance(entities_payload, dict):
        for entity_type, values in entities_payload.items():
            if isinstance(values, list):
                for value in values:
                    add_entity(entity_type, value)
            else:
                add_entity(entity_type, values)
    elif isinstance(entities_payload, list):
        for entity in entities_payload:
            if isinstance(entity, dict):
                entity_type = entity.get("type") or entity.get("entity_type") or entity.get("label")
                add_entity(entity_type, entity)
            else:
                add_entity("UNKNOWN", entity)

    extracted_payload = document.get("extracted_entities")
    if isinstance(extracted_payload, dict):
        for entity_type, values in extracted_payload.items():
            if isinstance(values, list):
                for value in values:
                    add_entity(entity_type, value)
            else:
                add_entity(entity_type, values)
    elif isinstance(extracted_payload, list):
        for entity in extracted_payload:
            if isinstance(entity, dict):
                entity_type = entity.get("type") or entity.get("entity_type") or entity.get("label")
                add_entity(entity_type, entity)
            else:
                add_entity("UNKNOWN", entity)

    # Include named_entities captured inside structured linked_information blocks.
    collect_named_entities(document.get("sections"))

    entity_refs = document.get("entity_refs")
    if isinstance(entity_refs, list) and entity_refs:
        ref_ids: List[ObjectId] = []
        for entity_ref in entity_refs:
            if isinstance(entity_ref, ObjectId):
                ref_ids.append(entity_ref)
            elif isinstance(entity_ref, str) and ObjectId.is_valid(entity_ref):
                ref_ids.append(ObjectId(entity_ref))

        if ref_ids:
            try:
                linked_docs = db["linked_entities"].find(
                    {"_id": {"$in": ref_ids}},
                    {
                        "type": 1,
                        "entity_type": 1,
                        "label": 1,
                        "term": 1,
                        "text": 1,
                        "name": 1,
                        "canonical_name": 1,
                        "aliases": 1,
                        "variants": 1,
                    },
                )
                for linked_doc in linked_docs:
                    entity_type = (
                        linked_doc.get("type")
                        or linked_doc.get("entity_type")
                        or linked_doc.get("label")
                        or "UNKNOWN"
                    )
                    linked_entity_id = str(linked_doc["_id"])
                    canonical_name = (
                        linked_doc.get("canonical_name")
                        or linked_doc.get("term")
                        or linked_doc.get("text")
                        or linked_doc.get("name")
                    )
                    add_entity(
                        entity_type,
                        canonical_name,
                        entity_id=linked_entity_id,
                        canonical_name=canonical_name,
                    )

                    aliases = linked_doc.get("aliases")
                    if isinstance(aliases, list):
                        for alias in aliases:
                            add_entity(
                                entity_type,
                                alias,
                                entity_id=linked_entity_id,
                                canonical_name=canonical_name,
                            )

                    variants = linked_doc.get("variants")
                    if isinstance(variants, list):
                        for variant in variants:
                            add_entity(
                                entity_type,
                                variant,
                                entity_id=linked_entity_id,
                                canonical_name=canonical_name,
                            )
            except Exception:
                logger.exception("Failed to load linked entity_refs for document detail view.")

    for entity_values in entities_by_type.values():
        entity_values.sort(key=lambda value: value["text"].casefold())

    if not entities_by_type:
        return []

    preferred_order = [
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "DATE",
        "TIME",
        "MONEY",
        "PERCENT",
        "OCCUPATION",
        "EMPLOYEE_ID",
    ]
    ordered_types = [entity_type for entity_type in preferred_order if entity_type in entities_by_type]
    ordered_types.extend(sorted(entity_type for entity_type in entities_by_type if entity_type not in preferred_order))

    return [
        (entity_type.replace("_", " ").title(), entities_by_type[entity_type])
        for entity_type in ordered_types
    ]


@app.route('/network-analysis')
def network_analysis_page():
    """Standalone network explorer page."""
    return render_template('network-analysis.html')


NETWORK_VIEWER_CONTEXT_TIMEOUT = 3600
NETWORK_VIEWER_MAX_DOCUMENTS = 2000
NETWORK_VIEWER_MAX_DOCS_PER_EDGE = 250


def _parse_network_type_filter(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _parse_network_int_arg(name: str, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    raw_value = request.args.get(name, default)
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        parsed_value = default
    parsed_value = max(minimum, parsed_value)
    if maximum is not None:
        parsed_value = min(parsed_value, maximum)
    return parsed_value


def _parse_network_bool_arg(name: str, default: bool) -> bool:
    raw_value = request.args.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_network_type_pair(raw_value: str) -> Tuple[Optional[str], Optional[str]]:
    raw = (raw_value or "").strip()
    if not raw:
        return None, None

    separator = "|" if "|" in raw else ","
    parts = [part.strip() for part in raw.split(separator) if part.strip()]
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _parse_network_rank_mode(default: str = "most_connected") -> str:
    allowed = {
        "most_connected",
        "most_cross_type",
        "rare_but_strong",
        "low_frequency_pairs",
    }
    raw_mode = request.args.get("rank_mode", default)
    mode = str(raw_mode or default).strip().lower()
    return mode if mode in allowed else default


def _parse_network_pair_preset(default: str = "cross_type_only") -> str:
    allowed = {"cross_type_only", "all_pairs", "within_type_only"}
    raw_preset = request.args.get("pair_preset", default)
    preset = str(raw_preset or default).strip().lower()
    return preset if preset in allowed else default


def _build_network_viewer_context() -> Dict[str, Any]:
    config = NetworkConfig.from_env()
    entity_id = request.args.get("entity", "").strip() or None
    document_term = request.args.get("document_term", request.args.get("entity_search", "")).strip()
    type_filter = _parse_network_type_filter(request.args.get("type_filter", "").strip())
    source_type, target_type = _parse_network_type_pair(request.args.get("type_pair", ""))
    min_weight = _parse_network_int_arg("min_weight", config.default_min_weight, minimum=1, maximum=9999)
    edge_limit = _parse_network_int_arg("limit", config.default_limit, minimum=1, maximum=5000)
    strict_type_filter = _parse_network_bool_arg("strict_type_filter", True)
    rank_mode = _parse_network_rank_mode()
    pair_preset = _parse_network_pair_preset()

    context = get_network_documents_for_viewer(
        db,
        entity_id=entity_id,
        type_filter=type_filter or None,
        source_type=source_type,
        target_type=target_type,
        min_weight=min_weight,
        limit=edge_limit,
        strict_type_filter=strict_type_filter,
        pair_preset=pair_preset,
        rank_mode=rank_mode,
        document_term=document_term,
        max_documents=NETWORK_VIEWER_MAX_DOCUMENTS,
        max_docs_per_edge=NETWORK_VIEWER_MAX_DOCS_PER_EDGE,
    )

    network_query = request.query_string.decode("utf-8")
    analysis_url = url_for("network_analysis_page")
    if network_query:
        analysis_url = f"{analysis_url}?{network_query}"

    context["filters"] = {
        "entity": entity_id,
        "type_filter": type_filter,
        "min_weight": min_weight,
        "limit": edge_limit,
        "strict_type_filter": strict_type_filter,
        "type_pair": [source_type, target_type] if source_type and target_type else [],
        "pair_preset": pair_preset,
        "rank_mode": rank_mode,
        "document_term": document_term,
    }
    context["analysis_url"] = analysis_url
    context["generated_at"] = datetime.utcnow().isoformat()
    return context


def _hydrate_network_documents_from_ids(ordered_ids: List[str]) -> List[Dict[str, Any]]:
    if not ordered_ids:
        return []

    valid_object_ids = [ObjectId(doc_id) for doc_id in ordered_ids if ObjectId.is_valid(doc_id)]
    filename_by_id: Dict[str, str] = {}
    if valid_object_ids:
        for doc in documents.find({"_id": {"$in": valid_object_ids}}, {"filename": 1}):
            filename_by_id[str(doc["_id"])] = doc.get("filename", "Unknown")

    rows = []
    for doc_id in ordered_ids:
        rows.append(
            {
                "document_id": doc_id,
                "filename": filename_by_id.get(doc_id, "Unknown"),
                "matched_entity_count": None,
                "matched_edge_count": None,
                "network_score": None,
            }
        )
    return rows


@app.route('/network/viewer-launch')
def network_viewer_launch():
    """
    Build a network-derived document list and launch list-driven document viewer.

    The resulting list is cached using the same `search_<id>` key contract as
    Search Database so document detail prev/next navigation works unchanged.
    """
    config = NetworkConfig.from_env()
    if not config.enabled:
        flash("Network analysis is disabled.", "warning")
        return redirect(url_for("network_analysis_page"))

    context = _build_network_viewer_context()
    documents_for_viewer = context.get("documents", [])

    search_id = str(uuid.uuid4())
    ordered_ids = [entry["document_id"] for entry in documents_for_viewer if entry.get("document_id")]
    cache.set(f"search_{search_id}", ordered_ids, timeout=NETWORK_VIEWER_CONTEXT_TIMEOUT)
    cache.set(f"network_search_{search_id}", context, timeout=NETWORK_VIEWER_CONTEXT_TIMEOUT)

    if not ordered_ids:
        flash("No documents matched the current network filters.", "info")

    return redirect(url_for("network_viewer_results", search_id=search_id))


@app.route('/network/viewer-results')
def network_viewer_results():
    """Render network-generated document list for next/previous document navigation."""
    search_id = request.args.get("search_id", "").strip()
    if not search_id:
        flash("Missing network viewer context.", "warning")
        return redirect(url_for("network_analysis_page"))

    context = cache.get(f"network_search_{search_id}") or {}
    documents_list = context.get("documents", [])

    if not documents_list:
        ordered_ids = cache.get(f"search_{search_id}") or []
        if ordered_ids:
            documents_list = _hydrate_network_documents_from_ids(ordered_ids)

    page = _parse_network_int_arg("page", 1, minimum=1, maximum=100000)
    per_page = _parse_network_int_arg("per_page", 50, minimum=10, maximum=200)

    total_count = len(documents_list)
    total_pages = max(1, math.ceil(total_count / per_page)) if per_page else 1
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_documents = documents_list[start_idx:end_idx]

    viewer_meta = {
        "mode": context.get("mode", "global"),
        "entity": context.get("entity"),
        "filters": context.get("filters", {}),
        "analysis_url": context.get("analysis_url", url_for("network_analysis_page")),
        "generated_at": context.get("generated_at"),
        "stats": context.get("stats", {}),
        "expired": not bool(context),
    }

    return render_template(
        "network-document-list.html",
        search_id=search_id,
        documents=page_documents,
        total_count=total_count,
        current_page=page,
        total_pages=total_pages,
        per_page=per_page,
        viewer_meta=viewer_meta,
    )


@app.route('/document/<string:doc_id>')
# @login_required
def document_detail(doc_id):
    
    
    # Hard-coded SHOW_EMPTY variable
    SHOW_EMPTY = False  # Set to True to show empty fields, False to hide them

    # Function to clean the document data
    def clean_data(data):
        empty_values = [None, '', 'N/A', 'null', [], {}, 'None']
        if isinstance(data, dict):
            return {
                k: clean_data(v)
                for k, v in data.items()
                if v not in empty_values and clean_data(v) not in empty_values
            }
        elif isinstance(data, list):
            return [
                clean_data(item)
                for item in data
                if item not in empty_values and clean_data(item) not in empty_values
            ]
        else:
            return data
    
    
    
    
    search_id = request.args.get('search_id', '')
    origin = request.args.get('origin', '').strip().lower()

    try:
        # Fetch the document by ID
        document = find_document_by_id(db, doc_id)
        if not document:
            abort(404)

        document['_id'] = str(document['_id'])

        # Log the document information for debugging
        logger.debug(f"Retrieved document for ID {doc_id}: {document}")


        # Decide whether to clean the document based on SHOW_EMPTY
        if SHOW_EMPTY:
            document = document
        else:
            # Clean the document to remove empty fields
            document = clean_data(document)
        
        # Strip common file extensions for display
        # Strip all common file extensions from right to left
        display_filename = document.get('filename', 'Untitled Document')
        extensions = ['.json', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pdf']

        # Keep stripping extensions until none are found
        while True:
            stripped = False
            for ext in extensions:
                if display_filename.lower().endswith(ext):
                    display_filename = display_filename[:-len(ext)]
                    stripped = True
                    break  # Start over to check for more extensions
            if not stripped:
                break  # No more extensions found

        # Retrieve ordered IDs from cache when a valid search context is present.
        ordered_ids = cache.get(f'search_{search_id}') if search_id else None

        if ordered_ids and doc_id in ordered_ids:
            current_index = ordered_ids.index(doc_id)
            prev_id = ordered_ids[current_index - 1] if current_index > 0 else None
            next_id = ordered_ids[current_index + 1] if current_index < len(ordered_ids) - 1 else None
        else:
            # Allow direct links and stale search IDs to render the document page.
            prev_id = None
            next_id = None

        if origin == "network":
            if search_id:
                back_to_list_url = url_for("network_viewer_results", search_id=search_id)
                back_to_list_label = "Network Results"
            else:
                back_to_list_url = url_for("network_analysis_page")
                back_to_list_label = "Network"
        else:
            back_to_list_url = f"{url_for('search_database')}?return_to_search=true"
            back_to_list_label = "Search"

        prev_url = "#"
        next_url = "#"
        if prev_id:
            prev_params = {"doc_id": prev_id, "search_id": search_id}
            if origin:
                prev_params["origin"] = origin
            prev_url = url_for("document_detail", **prev_params)
        if next_id:
            next_params = {"doc_id": next_id, "search_id": search_id}
            if origin:
                next_params["origin"] = origin
            next_url = url_for("document_detail", **next_params)

        # Get the relative path from the document
        relative_path = document.get('relative_path')  # This should contain the relative path to the JSON file

        if relative_path:
            # Construct the image path by removing the '.json' extension
            image_path = relative_path.replace('.json', '')  # 'rolls/rolls/tray_1_roll_5_page3303_img1.png'
            logger.debug(f"Document ID: {doc_id}, Image path: {image_path}")

            # Check if the image file exists
            archives_root = os.environ.get("ARCHIVES_PATH", "/data/archives")
            absolute_image_path = os.path.join(archives_root, image_path)
            image_exists = os.path.exists(absolute_image_path)

            if not image_exists:
                logger.warning(f"Image not found at: {absolute_image_path}. The absolute path was constructed from relative path: {relative_path}")
        else:
            # Log an error if relative_path is None or not found
            logger.error(f"Error: No relative_path found for document ID: {doc_id}. Document content: {document}")
            image_exists = False
            image_path = None

        ner_groups = build_document_ner_groups(document)

        # Render the template with all required variables
        return render_template(
            'document-detail.html',
            document=document,
            display_filename=display_filename,
            prev_id=prev_id,
            next_id=next_id,
            prev_url=prev_url,
            next_url=next_url,
            search_id=search_id,
            origin=origin,
            back_to_list_url=back_to_list_url,
            back_to_list_label=back_to_list_label,
            image_path=image_path,  # Pass the constructed image path
            image_exists=image_exists,  # Pass the flag indicating if the image exists
            ner_groups=ner_groups
        )
    except Exception as e:
        logger.error(f"Error in document_detail: {str(e)}", exc_info=True)
        abort(500)


#more attempts to fix image serving
@app.route('/images/<path:filename>')
def serve_image(filename):
    archive_root = _archives_root()  # Pull archive root from configuration for consistent serving
    image_path = archive_root / filename  # Build absolute path using configured archive root
    logger.debug(f"Serving image from: {image_path}")
    
    if image_path.exists():
        return send_file(image_path)
    else:
        logger.warning(f"Image not found at: {image_path}. also tried {archive_root / filename}")
        abort(404)


def get_top_unique_terms(db, field, term_type, query='', limit=1000, skip=0):
    """
    Retrieve top unique terms based on the field, term type, and optional search query.

    :param db: Database instance
    :param field: The field to filter terms by (e.g., 'title', 'description')
    :param term_type: The type of term ('word' or 'phrase')
    :param query: Optional search query string to filter terms
    :param limit: Number of top terms to retrieve
    :param skip: Number of records to skip for pagination
    :return: List of dictionaries with term and count
    """
    unique_terms_collection = db['unique_terms']
    
    try:
        # Base MongoDB query
        mongo_query = {"field": field, "type": term_type}
        
        # If a search query is provided, add a regex filter for the 'term' field
        if query:
            # Escape special regex characters to prevent injection attacks
            escaped_query = re.escape(query)
            # Case-insensitive search for terms containing the query substring
            mongo_query['term'] = {"$regex": f".*{escaped_query}.*", "$options": "i"}
        
        start_time = time.time()
        
        # Execute the query with sorting, skipping, and limiting for pagination
        cursor = unique_terms_collection.find(
            mongo_query,
            {"_id": 0, "term": 1, "frequency": 1}
        ).sort("frequency", pymongo.DESCENDING).skip(skip).limit(limit)
        
        terms_list = []
        for doc in cursor:
            key = 'word' if term_type == 'word' else 'phrase'
            terms_list.append({key: doc['term'], 'count': doc['frequency']})
        
        duration = time.time() - start_time
        logger.info(f"Retrieved top {len(terms_list)} {term_type}s in {duration:.4f} seconds for field '{field}' with query '{query}'.")
        
        return terms_list
    except Exception as e:
        logger.error(f"Error retrieving unique terms: {e}")
        return []

def get_unique_terms_count(db, field, term_type, query=''):
    """
    Get the count of unique terms based on the field, term type, and optional search query.

    :param db: Database instance
    :param field: The field to filter terms by
    :param term_type: The type of term ('word' or 'phrase')
    :param query: Optional search query string to filter terms
    :return: Integer count of unique terms
    """
    unique_terms_collection = db['unique_terms']
    
    try:
        # Base MongoDB query
        mongo_query = {"field": field, "type": term_type}
        
        # If a search query is provided, add a regex filter for the 'term' field
        if query:
            # Escape special regex characters to prevent injection attacks
            escaped_query = re.escape(query)
            # Case-insensitive search for terms containing the query substring
            mongo_query['term'] = {"$regex": f".*{escaped_query}.*", "$options": "i"}
        
        # Count the number of unique terms matching the query
        count = unique_terms_collection.count_documents(mongo_query)
        logger.info(f"Counted {count} unique {term_type}s for field '{field}' with query '{query}'.")
        return count
    except Exception as e:
        logger.error(f"Error counting unique terms: {e}")
        return 0



@app.route('/search-terms', methods=['GET'])
def search_terms():
    client = get_client()  # Initialize your MongoDB client
    db = get_db(client)    # Get the database instance

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Handle AJAX request
        field = request.args.get('field')
        if not field:
            return jsonify({"error": "No field specified"}), 400

        # Extract the search query
        query = request.args.get('query', '').strip().lower()  # Normalize the query

        logger.debug(f"AJAX request for field: {field}, query: {query}")  # Using logger here

        # Define term types
        term_types = ['word', 'phrase']
        data = {}
        total_records = 0

        for term_type in term_types:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 100))
            skip = (page - 1) * per_page

            # Fetch filtered terms based on the query
            terms = get_top_unique_terms(db, field, term_type, query=query, limit=per_page, skip=skip)
            data[term_type + 's'] = terms

            # Fetch the count of unique terms based on the query
            count = get_unique_terms_count(db, field, term_type, query=query)
            data['unique_' + term_type + 's'] = count
            total_records += count

        data['total_records'] = total_records

        return jsonify(data)
    else:
        # Render the HTML template
        field_structure = get_field_structure(db)
        unique_fields = []  # Define if necessary
        return render_template('search-terms.html', field_structure=field_structure, unique_fields=unique_fields)

@app.route('/database-info')
# @login_required
def database_info():
    field_struct = get_field_structure(db)  # Pass 'db' here
    collection_info = []

    def count_documents_with_field(field_path):
        count = documents.count_documents({field_path: {'$exists': True}})
        return count

    def traverse_structure(structure, current_path=''):
        for field, value in structure.items():
            path = f"{current_path}.{field}" if current_path else field
            if isinstance(value, dict):
                traverse_structure(value, current_path=path)
            else:
                count = count_documents_with_field(path)
                collection_info.append({
                    'name': path,
                    'count': count
                })

    traverse_structure(field_struct)

    return render_template('database-info.html', collection_info=collection_info)


@app.route('/help')
def help_page():
    """Render the in-app help centre focused on current workflows."""

    return render_template('help.html')

@app.route('/settings', methods=['GET', 'POST'])
# @login_required
def settings():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    if request.method == 'POST':
        action = request.form.get('action', 'update_ui')

        if action == 'upload_archives':
            uploads = request.files.getlist('archives')
            has_selection = any(getattr(item, 'filename', '') for item in uploads)
            if not has_selection:
                flash('Select one or more JSON files to upload.', 'warning')
                return redirect(url_for('settings') + '#data-ingestion')

            saved_files, rejected_files = _process_archive_uploads(
                uploads,
                request.form.get('target_subdir'),
            )

            if saved_files:
                display_subset = ', '.join(saved_files[:3])
                if len(saved_files) > 3:
                    display_subset += ', …'
                flash(
                    f"Uploaded {len(saved_files)} file(s): {display_subset}",
                    'success',
                )
            else:
                flash(
                    'No files were uploaded. Please choose JSON files with supported extensions (.json, .jsonl).',
                    'warning',
                )

            if rejected_files:
                flash(
                    f"Skipped unsupported files: {', '.join(rejected_files)}",
                    'warning',
                )

            return redirect(url_for('settings') + '#data-ingestion')

        new_config = request.form.to_dict()
        new_config.pop('action', None)

        for key in ['fonts', 'sizes', 'colors', 'spacing']:
            if key in new_config:
                try:
                    new_config[key] = json.loads(new_config[key])
                except json.JSONDecodeError:
                    flash(f"Invalid JSON format for {key}.", 'danger')
                    return redirect(url_for('settings') + '#ui-preferences')

        try:
            with open(config_path, 'w') as config_file:
                json.dump(new_config, config_file, indent=4)
            app.config['UI_CONFIG'] = new_config
            flash('UI preferences updated successfully.', 'success')
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            flash('Failed to update UI settings.', 'danger')
        return redirect(url_for('settings') + '#ui-preferences')

    try:
        if os.path.exists(config_path):
            with open(config_path) as config_file:
                config = json.load(config_file)
        else:
            config = {}
    except json.JSONDecodeError:
        config = {}
        flash('Configuration file is corrupted. Using default settings.', 'warning')

    archive_root_path = _archives_root()
    archive_files, archive_list_truncated = _list_archives()
    overrides = _historian_agent_overrides()
    agent_config = HistorianAgentConfig.from_env(overrides)
    agent_error = _evaluate_agent_error(overrides=overrides, config=agent_config)

    return render_template(
        'settings.html',
        config=config,
        archive_root=str(archive_root_path),
        archive_files=archive_files,
        archive_list_truncated=archive_list_truncated,
        agent_config_payload=_serialise_agent_config(agent_config),
        provider_options=HISTORIAN_PROVIDER_OPTIONS,
        agent_error=agent_error,
        config_endpoint=url_for('historian_agent_config'),
        ingestion_default_prompt=image_ingestion.DEFAULT_PROMPT,
        ingestion_default_provider=image_ingestion.DEFAULT_PROVIDER,
        ingestion_default_ollama_model=image_ingestion.DEFAULT_OLLAMA_MODEL,
        ingestion_default_openai_model=image_ingestion.DEFAULT_OPENAI_MODEL,
        ingestion_ollama_base_url=image_ingestion.DEFAULT_OLLAMA_BASE_URL,
        ingestion_api_key_configured=image_ingestion.read_api_key() is not None,
    )

@app.route('/settings/data-ingestion/mounts', methods=['GET'])
def list_data_ingestion_mounts():
    """Return Docker mount metadata so the UI can display read-only paths."""

    # Added exists flag to highlight mounts that are not yet present on disk.
    mounts_payload = []
    for source, target in get_mounted_paths():
        target_path = Path(target)
        mounts_payload.append({
            'source': source,
            'target': str(target),
            'target_exists': target_path.exists(),
        })

    return jsonify({'mounts': mounts_payload})


@app.route('/settings/data-ingestion/tree', methods=['POST'])
def list_data_ingestion_tree():
    """Return a short directory listing for a provided mount target."""

    payload = request.get_json(silent=True) or {}
    target = str(payload.get('target', '')).strip()
    if not target:
        # Added guard so clients receive a helpful message when target is missing.
        return jsonify({'error': 'Missing target'}), 400

    tree = short_tree(target)
    return jsonify(tree)


@app.route('/settings/data-ingestion/scan', methods=['GET', 'POST'])
def scan_mounted_images():
    """Scan mounts and stream progress via SSE."""

    if request.method == 'POST':  # Added handshake branch so the UI can validate config before opening SSE.
        payload = request.get_json(silent=True) or {}  # Added payload capture to reuse the selected provider/model across the stream.
        config, _api_key, error = _ingestion_config_from_payload(payload)  # Reused helper to keep validation identical to legacy behaviour.
        if error:
            return error  # Preserved error flow so clients still receive structured validation responses.
        session['scan_payload'] = payload  # Added session storage so the subsequent GET stream can reference the chosen config.
        return jsonify({'status': 'accepted'})  # Added acknowledgement letting the frontend know it can start listening for events.

    payload = session.get('scan_payload', {})  # Added fallback payload retrieval for GET requests that stream events.
    config, api_key, error = _ingestion_config_from_payload(payload)  # Rebuilt config during GET to honour any saved selector overrides.
    if error:
        return error  # Reused guard so GET responses surface validation issues immediately.

    def generate():
        """Generator that yields SSE-formatted messages."""

        try:
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting scan...'})}\n\n"  # Added start event so the frontend can show immediate feedback.

            for source, target in get_mounted_paths():  # Reused mount discovery to keep parity with the previous implementation.
                target_path = Path(target)

                if not target_path.is_dir():
                    continue  # Maintained skip for non-directories to avoid noisy SSE entries.

                if not target_path.exists():
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'Path missing: {target}'})}\n\n"  # Added warning stream when a mount is missing so operators can react in real-time.
                    continue

                yield f"data: {json.dumps({'type': 'info', 'message': f'Scanning: {target_path.name}'})}\n\n"  # Added informational event announcing each mount being processed.

                try:
                    for progress in image_ingestion.process_directory_streaming(  # Added streaming hook so image-level updates reach the UI live.
                        target_path,
                        config,
                        reprocess_existing=False,
                        api_key=api_key,
                    ):
                        yield f"data: {json.dumps(progress)}\n\n"  # Relayed backend progress dicts directly to SSE consumers.

                except Exception as exc:  # pragma: no cover - depends on external services
                    logger.exception('Mount scan failed for %s', target_path)  # Retained logging to aid troubleshooting on backend.
                    yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"  # Added error event so the frontend displays failures inline.

            yield f"data: {json.dumps({'type': 'complete', 'message': 'Scan complete!'})}\n\n"  # Added final completion event to close out the SSE stream cleanly.

        except Exception as exc:  # pragma: no cover - depends on runtime conditions
            logger.exception('Scan streaming failed')  # Added catch-all log to highlight unexpected generator failures.
            yield f"data: {json.dumps({'type': 'error', 'message': f'Fatal error: {str(exc)}'})}\n\n"  # Added fatal error event to keep UI informed about catastrophic issues.

    return Response(
        stream_with_context(generate()),  # Added stream wrapper so Flask handles generator lifecycle safely.
        mimetype='text/event-stream',  # Added SSE mimetype to ensure browsers treat the response as an event stream.
        headers={
            'Cache-Control': 'no-cache',  # Added headers to disable buffering and keep events real-time.
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/settings/data-ingestion/rebuild', methods=['POST'])
def rebuild_ingestion_database():
    """Clear ingestion collections and rebuild them from mounted directories."""

    payload = request.get_json(silent=True) or {}  # Added payload extraction so rebuild honours frontend configuration.
    config, api_key, error = _ingestion_config_from_payload(payload)  # Added parser reuse to avoid duplicating validation logic.
    if error:
        return error  # Added early return so rebuild stops when configuration is invalid.
    logger.warning('Rebuilding ingestion data with %s provider using %s', config.provider, config.model)  # Added log entry to audit destructive rebuild operations.

    # Added wipe of ingestion-related collections prior to reprocessing.
    documents.delete_many({})
    unique_terms_collection.delete_many({})
    field_structure_collection.delete_many({})

    totals = _blank_ingestion_totals()
    results: List[Dict[str, object]] = []

    for source, target in get_mounted_paths():
        target_path = Path(target)
        entry: Dict[str, object] = {
            'source': source,
            'target': str(target_path),
        }


        if not target_path.is_dir():
            # Skip non-directory mounts (like docker-compose.yml file)
            continue

        
        if not target_path.exists():
            entry['status'] = 'missing'
            results.append(entry)
            continue

        try:
            summary = image_ingestion.process_directory(
                target_path,
                config,
                reprocess_existing=True,
                api_key=api_key,
            )
            summary_dict = summary.as_dict()
            entry.update(summary_dict)
            entry['status'] = 'ok'
            _merge_ingestion_totals(totals, summary_dict)
        except image_ingestion.IngestionError as exc:
            entry['status'] = 'error'
            entry['message'] = str(exc)
        except Exception as exc:  # pragma: no cover - depends on external services
            logger.exception('Mount rebuild failed for %s', target_path)
            entry['status'] = 'error'
            entry['message'] = str(exc)

        results.append(entry)

    return jsonify({
        'status': 'ok',
        'action': 'rebuild',
        'provider': config.provider,
        'model': config.model,  # Added model echo so rebuild responses mirror scan payloads.
        'results': results,
        'aggregate': totals,
        'collections_cleared': ['documents', 'unique_terms', 'field_structure'],
    })


@app.route('/settings/data-ingestion/options', methods=['GET'])
def data_ingestion_options():
    """Return environment-driven defaults for the image ingestion pipeline."""

    import requests

    def get_ollama_model_list(base_url: str):
        """Query Ollama runtime for available models."""
        try:
            r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
            if r.ok:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    # Pull from .env first, fall back to defaults from image_ingestion
    env_provider = image_ingestion.provider_from_string(os.getenv("HISTORIAN_AGENT_MODEL_PROVIDER", image_ingestion.DEFAULT_PROVIDER))  # Updated to normalise the provider default using shared helper.
    env_prompt = os.getenv("HISTORIAN_AGENT_PROMPT") or image_ingestion.DEFAULT_PROMPT  # Updated to ignore blank env overrides and fall back cleanly.
    env_ollama_base = os.getenv("HISTORIAN_AGENT_OLLAMA_BASE_URL") or image_ingestion.DEFAULT_OLLAMA_BASE_URL  # Updated to treat empty strings as unset for stability.
    env_ollama_model = os.getenv("HISTORIAN_AGENT_MODEL") or image_ingestion.DEFAULT_OLLAMA_MODEL  # Updated to reuse the shared historian model default when env is empty.
    env_openai_model = os.getenv("OPENAI_DEFAULT_MODEL") or image_ingestion.DEFAULT_OPENAI_MODEL  # Updated to reuse baked-in defaults if env override is missing or blank.

    requested_base = (request.args.get('ollama_base_url') or '').strip()  # Added override so the UI can probe alternate Ollama hosts without editing .env.
    if requested_base:
        env_ollama_base = requested_base  # Updated to honour query-supplied base URL when fetching models.

    # Fetch model list from running Ollama instance
    models = get_ollama_model_list(env_ollama_base)

    return jsonify({
        "default_provider": env_provider,
        "default_prompt": env_prompt,
        "ollama": {
            "base_url": env_ollama_base,
            "default_model": env_ollama_model,
            "models": models,
        },
        "openai": {
            "default_model": env_openai_model,
            "key_configured": image_ingestion.read_api_key() is not None,
        },
    })



@app.route('/settings/data-ingestion/run', methods=['POST'])
def settings_run_data_ingestion():
    """Trigger the image-to-JSON ingestion workflow for a directory."""

    payload = request.get_json(silent=True) or {}
    directory_raw = str(payload.get('directory', '')).strip()
    if not directory_raw:
        return jsonify({'status': 'error', 'message': 'Provide a directory path to ingest.'}), 400

    try:
        candidate = Path(directory_raw)
        if candidate.is_absolute():
            directory_path = image_ingestion.expand_directory(directory_raw)
        else:
            directory_path = (_archives_root() / candidate).resolve()
    except Exception as exc:
        return jsonify({'status': 'error', 'message': f'Invalid directory: {exc}'}), 400

    provider = image_ingestion.provider_from_string(payload.get('provider'))
    prompt = str(payload.get('prompt') or image_ingestion.DEFAULT_PROMPT)
    base_url = payload.get('base_url') or payload.get('ollama_base_url')

    model_value = payload.get('model') or ''
    if provider == 'ollama':
        model = model_value.strip() or image_ingestion.DEFAULT_OLLAMA_MODEL
    else:
        openai_model = payload.get('openai_model') or model_value
        model = str(openai_model or image_ingestion.DEFAULT_OPENAI_MODEL).strip()

    reprocess_raw = payload.get('reprocess')
    if isinstance(reprocess_raw, str):
        reprocess = reprocess_raw.strip().lower() in {'1', 'true', 'yes', 'on'}
    else:
        reprocess = bool(reprocess_raw)

    api_key = None
    key_saved = False
    if provider == 'openai':
        provided_key = payload.get('api_key')
        api_key = image_ingestion.ensure_api_key(provided_key)
        key_saved = bool(provided_key)
        if not api_key:
            return jsonify({
                'status': 'error',
                'code': 'missing_api_key',
                'message': 'Provide an OpenAI API key before running ingestion.',
            }), 400

    config = image_ingestion.ModelConfig(
        provider=provider,
        model=model,
        prompt=prompt,
        base_url=base_url or None,
    )

    try:
        summary = image_ingestion.process_directory(
            directory_path,
            config,
            reprocess_existing=reprocess,
            api_key=api_key,
        )
    except image_ingestion.IngestionError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        logger.exception('Unexpected failure during image ingestion')
        return jsonify({'status': 'error', 'message': f'Unexpected failure: {exc}'}), 500

    response = {
        'status': 'ok',
        'summary': summary.as_dict(),
        'provider': provider,
        'directory': str(directory_path),
    }
    if provider == 'openai':
        response['api_key_saved'] = key_saved

    return jsonify(response)


@app.route('/settings/data-ingestion/api-key', methods=['POST'])
def settings_save_data_ingestion_key():
    """Persist a ChatGPT API key for future ingestion runs."""

    payload = request.get_json(silent=True) or {}
    api_key = str(payload.get('api_key', '')).strip()
    if not api_key:
        return jsonify({'status': 'error', 'message': 'Provide an API key to save.'}), 400

    image_ingestion.ensure_api_key(api_key)
    return jsonify({'status': 'ok'})


# Consider streaming if it ends up being thousands of documents
@app.route('/export_selected_csv', methods=['POST'])
# @login_required
def export_selected_csv():
    try:
        data = request.get_json()
        document_ids = data.get('document_ids', [])
        if not document_ids:
            return jsonify({"error": "No document IDs provided"}), 400

        # Convert string IDs to ObjectIds, handle invalid IDs
        valid_ids = []
        for doc_id in document_ids:
            try:
                valid_ids.append(ObjectId(doc_id))
            except Exception as e:
                logger.warning(f"Invalid document ID: {doc_id}")

        if not valid_ids:
            return jsonify({"error": "No valid document IDs provided"}), 400

        # Check if any documents exist with the provided IDs
        count = documents.count_documents({"_id": {"$in": valid_ids}})
        if count == 0:
            return jsonify({"error": "No documents found for the provided IDs."}), 404

        # Retrieve the documents
        documents_cursor = documents.find({"_id": {"$in": valid_ids}})

        # Create CSV
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['filename', 'OCR', 'original_json'])  # Header row

        for doc in documents_cursor:
            filename = doc.get('filename', 'N/A')
            ocr = doc.get('summary', 'N/A')  # Adjust field as necessary
            original_json = json.dumps(doc, default=str)  # Convert ObjectId to string if necessary
            writer.writerow([filename, ocr, original_json])

        # Prepare CSV for download
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=selected_documents.csv'}
        )

    except Exception as e:
        logger.error(f"Error exporting selected CSV: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message='An unexpected error has occurred'), 500
