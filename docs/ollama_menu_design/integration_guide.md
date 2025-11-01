# Ollama Dropdown Integration Guide for Data Ingestion

This guide explains how to integrate the Ollama model selector into your data ingestion settings.

## Files Created

1. **data_ingestion_model_selector.html** - HTML/CSS for the UI component
2. **data_ingestion_model_selector.js** - JavaScript for functionality
3. **integration_guide.md** - This file

## Integration Steps

### Step 1: Update `app/templates/settings.html`

Add the model selector HTML before the scan/rebuild buttons:

```html
<section id="data-ingestion" class="settings-section">
    <div class="settings-section__header">
        <h2>Data ingestion</h2>
        <p>The mounted archive folders below are defined in <code>docker-compose.yml</code> and surfaced read-only for review.</p>
    </div>
    
    <!-- ADD THIS: Model selector component -->
    {% include 'partials/data_ingestion_model_selector.html' %}
    
    <!-- Existing mounts section -->
    <div class="settings-subsection settings-ingestion__automation"
         data-ingestion-root
         data-mounts-url="{{ url_for('list_data_ingestion_mounts') }}"
         ...>
        <!-- existing mount explorer code -->
    </div>
</section>
```

### Step 2: Create the Partial Template

Move `data_ingestion_model_selector.html` to:
```
app/templates/partials/data_ingestion_model_selector.html
```

Or copy its contents directly into `settings.html` in the appropriate location.

### Step 3: Add JavaScript to Settings Page

At the bottom of `settings.html`, add the script include:

```html
{% block scripts %}
{{ super() }}
<script src="{{ url_for('static', filename='settings_data_ingestion.js') }}"></script>
<script src="{{ url_for('static', filename='data_ingestion_model_selector.js') }}"></script>
<script src="{{ url_for('static', filename='historian_agent.js') }}"></script>
{% endblock %}
```

Or copy the JavaScript to:
```
app/static/data_ingestion_model_selector.js
```

### Step 4: Update Scan/Rebuild Functions

Modify `app/static/settings_data_ingestion.js` to use the selected model configuration:

```javascript
// Add at the top of settings_data_ingestion.js
let ingestionConfig = {
    provider: 'ollama',
    model: '',
    ollama_base_url: '',
    prompt: ''
};

// Listen for config changes
document.addEventListener('ingestionConfigChange', (e) => {
    ingestionConfig = e.detail;
    console.log('Ingestion config updated:', ingestionConfig);
});

// Update triggerScan function
async function triggerScan() {
    // Validate configuration
    const validation = window.IngestionModelSelector.validateConfig();
    if (!validation.valid) {
        window.IngestionModelSelector.showStatus(validation.message, 'error');
        return;
    }

    const config = window.IngestionModelSelector.getConfig();
    
    // Show progress
    showProgress("Scanning mounted directories for new images...");
    
    try {
        // POST with configuration
        const result = await fetchJson(scanUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        displayResults(result);
    } catch (error) {
        showError('Scan failed: ' + error.message);
    }
}

// Update triggerRebuild similarly
async function triggerRebuild() {
    if (!confirm("This will delete all documents and rebuild from scratch!")) {
        return;
    }

    const validation = window.IngestionModelSelector.validateConfig();
    if (!validation.valid) {
        window.IngestionModelSelector.showStatus(validation.message, 'error');
        return;
    }

    const config = window.IngestionModelSelector.getConfig();
    
    showProgress("Rebuilding database from mounted archives...");
    
    try {
        const result = await fetchJson(rebuildUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        displayResults(result);
    } catch (error) {
        showError('Rebuild failed: ' + error.message);
    }
}
```

### Step 5: Update Backend Endpoints

The backend endpoints already support the configuration parameters. Verify that:

#### `app/routes.py` - `/settings/data-ingestion/scan`

```python
@app.route('/settings/data-ingestion/scan', methods=['POST'])
def scan_mounted_images():
    """Scan mounted directories and process new images only."""
    
    # Get configuration from request body
    payload = request.get_json(silent=True) or {}
    
    # Extract config with fallbacks
    provider = image_ingestion.provider_from_string(
        payload.get('provider', image_ingestion.DEFAULT_PROVIDER)
    )
    model = payload.get('model') or image_ingestion.DEFAULT_OLLAMA_MODEL
    prompt = payload.get('prompt') or image_ingestion.DEFAULT_PROMPT
    base_url = payload.get('ollama_base_url')
    
    # For OpenAI
    api_key = None
    if provider == 'openai':
        api_key = image_ingestion.ensure_api_key(payload.get('api_key'))
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'OpenAI API key required'
            }), 400
    
    # Create config
    config = image_ingestion.ModelConfig(
        provider=provider,
        model=model,
        prompt=prompt,
        base_url=base_url or None,
    )
    
    # Rest of the scanning logic...
```

#### `app/routes.py` - `/settings/data-ingestion/rebuild`

Update similarly to accept configuration from request body instead of using defaults.

## Features

### 1. Dynamic Ollama Model Loading

The dropdown automatically fetches available models from your Ollama instance:

```javascript
// Triggered on page load and when base URL changes
loadOllamaModels();
```

### 2. Provider Switching

Seamlessly switch between Ollama and OpenAI:

```javascript
providerSelect.addEventListener('change', handleProviderChange);
```

### 3. Configuration Validation

Before running scan/rebuild:

```javascript
const validation = window.IngestionModelSelector.validateConfig();
if (!validation.valid) {
    showError(validation.message);
    return;
}
```

### 4. Status Messages

User-friendly feedback during operations:

```javascript
window.IngestionModelSelector.showStatus('Loading models...', 'info');
window.IngestionModelSelector.showStatus('Success!', 'success');
window.IngestionModelSelector.showStatus('Error occurred', 'error');
```

## API Reference

### JavaScript API

```javascript
// Initialize the component
window.IngestionModelSelector.init();

// Get current configuration
const config = window.IngestionModelSelector.getConfig();
// Returns: { provider, model, ollama_base_url, prompt, ... }

// Validate configuration
const validation = window.IngestionModelSelector.validateConfig();
// Returns: { valid: boolean, message: string }

// Reload Ollama models
window.IngestionModelSelector.loadOllamaModels(force = true);

// Show/hide status messages
window.IngestionModelSelector.showStatus(message, type);
window.IngestionModelSelector.hideStatus();
```

### Events

Listen for configuration changes:

```javascript
document.addEventListener('ingestionConfigChange', (event) => {
    const config = event.detail;
    console.log('New config:', config);
});
```

## Backend Endpoints

### GET `/settings/data-ingestion/options`

Returns available models and defaults:

```json
{
  "default_provider": "ollama",
  "default_prompt": "...",
  "ollama": {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.2-vision:11b",
    "models": [
      "llama3.2-vision:11b",
      "llava:13b",
      "bakllava:latest"
    ]
  },
  "openai": {
    "default_model": "gpt-4o-mini",
    "key_configured": true
  }
}
```

Query parameters:
- `ollama_base_url` (optional): Override Ollama base URL

### POST `/settings/data-ingestion/scan`

Scan and process new images:

```json
{
  "provider": "ollama",
  "model": "llama3.2-vision:11b",
  "ollama_base_url": "http://localhost:11434",
  "prompt": "Extract all text..."
}
```

### POST `/settings/data-ingestion/rebuild`

Rebuild entire database:

```json
{
  "provider": "ollama",
  "model": "llama3.2-vision:11b",
  "ollama_base_url": "http://localhost:11434",
  "prompt": "Extract all text..."
}
```

## Styling

The component uses CSS that matches your existing design system. Key classes:

- `.ingestion-config` - Main container
- `.form-field` - Field wrapper
- `.form-select`, `.form-input`, `.form-textarea` - Form controls
- `.provider-section` - Provider-specific sections
- `.config-status` - Status messages

Colors are derived from your existing palette (Tailwind-style grays and blues).

## Troubleshooting

### Models Not Loading

1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Verify base URL is correct
3. Check browser console for errors
4. Try clicking "Refresh models" button

### Configuration Not Persisting

The configuration is stored in memory and sent with each scan/rebuild request. It's not persisted between page loads by design (matching the Historian Agent behavior).

### API Key Not Working

OpenAI keys are validated server-side. Check:
1. Key format is correct (starts with `sk-`)
2. Key has sufficient credits
3. Backend logs for specific error messages

## Next Steps

1. Test the integration with your existing codebase
2. Verify scan/rebuild operations use the selected model
3. Add user preferences storage if needed (localStorage or backend)
4. Consider adding model performance metrics
5. Add support for custom model parameters (temperature, max tokens, etc.)

## Support

For issues or questions:
1. Check browser console for JavaScript errors
2. Check Flask logs for backend errors
3. Verify Ollama/OpenAI connectivity
4. Review the existing `historian_agent_settings.html` for reference patterns
