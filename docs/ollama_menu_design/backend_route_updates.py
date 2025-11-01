"""
Backend Route Updates for Data Ingestion Model Selection
Add these changes to app/routes.py
"""

# UPDATED: /settings/data-ingestion/scan endpoint
# Accept model configuration from request body

@app.route('/settings/data-ingestion/scan', methods=['POST'])
def scan_mounted_images():
    """Scan mounted directories for new images and process them with selected model."""
    
    # Parse JSON payload (sent from frontend model selector)
    payload = request.get_json(silent=True) or {}
    
    # Extract model configuration with fallbacks to defaults
    provider_str = payload.get('provider', image_ingestion.DEFAULT_PROVIDER)
    provider = image_ingestion.provider_from_string(provider_str)
    
    prompt = str(payload.get('prompt') or image_ingestion.DEFAULT_PROMPT)
    base_url = payload.get('ollama_base_url') or payload.get('base_url')
    
    # Handle model selection based on provider
    model_value = payload.get('model', '')
    if provider == 'ollama':
        model = model_value.strip() or image_ingestion.DEFAULT_OLLAMA_MODEL
    else:  # openai
        openai_model = payload.get('openai_model') or model_value
        model = str(openai_model or image_ingestion.DEFAULT_OPENAI_MODEL).strip()
    
    # Handle OpenAI API key
    api_key = None
    if provider == 'openai':
        provided_key = payload.get('api_key')
        api_key = image_ingestion.ensure_api_key(provided_key)
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'OpenAI API key required. Please configure it in the settings.',
                'code': 'missing_api_key'
            }), 400
    
    # Create model configuration
    config = image_ingestion.ModelConfig(
        provider=provider,
        model=model,
        prompt=prompt,
        base_url=base_url or None,
    )
    
    # Log configuration for debugging
    logger.info(
        'Starting scan with %s provider, model: %s',
        config.provider,
        config.model
    )
    
    # Process mounted directories
    totals = _blank_ingestion_totals()
    results: List[Dict[str, object]] = []
    
    for source, target in get_mounted_paths():
        target_path = Path(target)
        entry: Dict[str, object] = {
            'source': source,
            'target': str(target_path),
        }
        
        if not target_path.exists():
            entry['status'] = 'missing'
            results.append(entry)
            continue
        
        try:
            # Process with selected configuration
            summary = image_ingestion.process_directory(
                target_path,
                config,
                reprocess_existing=False,  # Only new images
                api_key=api_key,
            )
            summary_dict = summary.as_dict()
            entry.update(summary_dict)
            entry['status'] = 'ok'
            _merge_ingestion_totals(totals, summary_dict)
            
        except image_ingestion.IngestionError as exc:
            entry['status'] = 'error'
            entry['message'] = str(exc)
            logger.error('Ingestion error for %s: %s', target_path, exc)
            
        except Exception as exc:
            entry['status'] = 'error'
            entry['message'] = str(exc)
            logger.exception('Unexpected error scanning %s', target_path)
        
        results.append(entry)
    
    return jsonify({
        'status': 'ok',
        'action': 'scan',
        'provider': config.provider,
        'model': config.model,
        'results': results,
        'aggregate': totals,
    })


# UPDATED: /settings/data-ingestion/rebuild endpoint
# Accept model configuration from request body

@app.route('/settings/data-ingestion/rebuild', methods=['POST'])
def rebuild_ingestion_database():
    """Clear database and rebuild from all mounted directories with selected model."""
    
    # Parse JSON payload
    payload = request.get_json(silent=True) or {}
    
    # Extract model configuration with fallbacks
    provider_str = payload.get('provider', image_ingestion.DEFAULT_PROVIDER)
    provider = image_ingestion.provider_from_string(provider_str)
    
    prompt = str(payload.get('prompt') or image_ingestion.DEFAULT_PROMPT)
    base_url = payload.get('ollama_base_url') or payload.get('base_url')
    
    # Handle model selection
    model_value = payload.get('model', '')
    if provider == 'ollama':
        model = model_value.strip() or image_ingestion.DEFAULT_OLLAMA_MODEL
    else:  # openai
        openai_model = payload.get('openai_model') or model_value
        model = str(openai_model or image_ingestion.DEFAULT_OPENAI_MODEL).strip()
    
    # Handle OpenAI API key
    api_key = None
    if provider == 'openai':
        provided_key = payload.get('api_key')
        api_key = image_ingestion.ensure_api_key(provided_key)
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'Configure an OpenAI API key before rebuilding.',
                'code': 'missing_api_key'
            }), 400
    
    # Create configuration
    config = image_ingestion.ModelConfig(
        provider=provider,
        model=model,
        prompt=prompt,
        base_url=base_url or None,
    )
    
    # Log rebuild operation
    logger.warning(
        'Starting database rebuild with %s provider, model: %s',
        config.provider,
        config.model
    )
    
    # Clear existing data
    logger.info('Clearing ingestion-related collections')
    documents.delete_many({})
    unique_terms_collection.delete_many({})
    field_structure_collection.delete_many({})
    
    # Process all mounted directories
    totals = _blank_ingestion_totals()
    results: List[Dict[str, object]] = []
    
    for source, target in get_mounted_paths():
        target_path = Path(target)
        entry: Dict[str, object] = {
            'source': source,
            'target': str(target_path),
        }
        
        if not target_path.exists():
            entry['status'] = 'missing'
            results.append(entry)
            continue
        
        try:
            # Reprocess everything with selected configuration
            summary = image_ingestion.process_directory(
                target_path,
                config,
                reprocess_existing=True,  # Reprocess all
                api_key=api_key,
            )
            summary_dict = summary.as_dict()
            entry.update(summary_dict)
            entry['status'] = 'ok'
            _merge_ingestion_totals(totals, summary_dict)
            
        except image_ingestion.IngestionError as exc:
            entry['status'] = 'error'
            entry['message'] = str(exc)
            logger.error('Rebuild error for %s: %s', target_path, exc)
            
        except Exception as exc:
            entry['status'] = 'error'
            entry['message'] = str(exc)
            logger.exception('Unexpected error rebuilding %s', target_path)
        
        results.append(entry)
    
    return jsonify({
        'status': 'ok',
        'action': 'rebuild',
        'provider': config.provider,
        'model': config.model,
        'results': results,
        'aggregate': totals,
        'collections_cleared': ['documents', 'unique_terms', 'field_structure'],
    })


# EXISTING: /settings/data-ingestion/options endpoint
# This already exists and works correctly, no changes needed

@app.route('/settings/data-ingestion/options', methods=['GET'])
def data_ingestion_options():
    """Return configuration details for the image ingestion pipeline."""
    
    base_url = request.args.get('ollama_base_url') or image_ingestion.DEFAULT_OLLAMA_BASE_URL
    models = image_ingestion.ollama_models(base_url)
    
    return jsonify({
        'default_provider': image_ingestion.DEFAULT_PROVIDER,
        'default_prompt': image_ingestion.DEFAULT_PROMPT,
        'ollama': {
            'base_url': base_url,
            'default_model': image_ingestion.DEFAULT_OLLAMA_MODEL,
            'models': models,
        },
        'openai': {
            'default_model': image_ingestion.DEFAULT_OPENAI_MODEL,
            'key_configured': image_ingestion.read_api_key() is not None,
        },
    })


# HELPER: Update the _ingestion_default_config helper if needed
# This is used when no config is provided in request

def _ingestion_default_config() -> Tuple[image_ingestion.ModelConfig, Optional[str]]:
    """
    Return default ingestion configuration.
    Now deprecated in favor of accepting config from request body.
    Kept for backward compatibility.
    """
    provider = image_ingestion.provider_from_string(image_ingestion.DEFAULT_PROVIDER)
    
    if provider == 'ollama':
        model = image_ingestion.DEFAULT_OLLAMA_MODEL
    else:
        model = image_ingestion.DEFAULT_OPENAI_MODEL
    
    config = image_ingestion.ModelConfig(
        provider=provider,
        model=model,
        prompt=image_ingestion.DEFAULT_PROMPT,
        base_url=image_ingestion.DEFAULT_OLLAMA_BASE_URL,
    )
    
    api_key = None
    if provider == 'openai':
        api_key = image_ingestion.read_api_key()
    
    return config, api_key


# NOTES:
# 1. The scan endpoint now accepts JSON body with model configuration
# 2. The rebuild endpoint now accepts JSON body with model configuration
# 3. Both endpoints validate OpenAI API keys when needed
# 4. Configuration is logged for debugging
# 5. Error messages are user-friendly
# 6. The existing /options endpoint works without changes
