# Ollama Dropdown for Data Ingestion - Implementation Summary

## Overview

This implementation adds a dynamic Ollama model selector to your data ingestion settings page. Users can:

1. **Choose between Ollama (local) or OpenAI (cloud)** providers
2. **Select from available Ollama models** dynamically fetched from the Ollama API
3. **Configure model settings** (base URL, prompts) before running scans/rebuilds
4. **Validate configuration** before operations start

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Settings Page UI                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Model Selector Component                                 │  │
│  │  ┌─────────────┐  ┌──────────────────────────────────┐   │  │
│  │  │ Provider:   │  │ Ollama Model:                    │   │  │
│  │  │ ○ Ollama    │  │ ▼ llama3.2-vision:11b            │   │  │
│  │  │ ○ OpenAI    │  │   llava:13b                      │   │  │
│  │  └─────────────┘  │   bakllava:latest                │   │  │
│  │                   └──────────────────────────────────┘   │  │
│  │  [Refresh Models]                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Mount Explorer                                           │  │
│  │  /archives/folder1 → /mnt/archives/folder1                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  [Scan for new images]  [Rebuild database]                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Config sent with request
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Backend API                                │
│                                                                  │
│  GET  /settings/data-ingestion/options                          │
│       → Returns: { ollama: { models: [...] }, openai: {...} }   │
│                                                                  │
│  POST /settings/data-ingestion/scan                             │
│       ← Accepts: { provider, model, prompt, base_url }          │
│       → Scans new images with selected model                    │
│                                                                  │
│  POST /settings/data-ingestion/rebuild                          │
│       ← Accepts: { provider, model, prompt, base_url }          │
│       → Rebuilds database with selected model                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Model Configuration
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Image Processing (image_ingestion.py)           │
│                                                                  │
│  process_directory(config, reprocess_existing)                  │
│    ├─ If Ollama: _call_ollama(image, config)                    │
│    │    └─ POST to Ollama API with base64 image                 │
│    └─ If OpenAI: _call_openai(image, config, api_key)           │
│         └─ Call OpenAI vision API                               │
└─────────────────────────────────────────────────────────────────┘
```

## Files Provided

### 1. Frontend Components

#### `data_ingestion_model_selector.html`
- **Purpose**: HTML template with embedded CSS
- **Location**: Copy to `app/templates/partials/data_ingestion_model_selector.html`
- **Contents**:
  - Provider select (Ollama/OpenAI)
  - Ollama section (base URL + model dropdown)
  - OpenAI section (model select + API key input)
  - Prompt textarea
  - Status message container
  - Responsive CSS styling

#### `data_ingestion_model_selector.js`
- **Purpose**: JavaScript for dynamic functionality
- **Location**: Copy to `app/static/data_ingestion_model_selector.js`
- **Features**:
  - Fetches Ollama models from `/settings/data-ingestion/options`
  - Dynamically populates model dropdown
  - Handles provider switching
  - Validates configuration
  - Emits events for integration with scan/rebuild buttons
  - Shows user-friendly status messages

### 2. Backend Updates

#### `backend_route_updates.py`
- **Purpose**: Updated route handlers
- **Location**: Apply changes to `app/routes.py`
- **Changes**:
  - `/settings/data-ingestion/scan` - Accept config from POST body
  - `/settings/data-ingestion/rebuild` - Accept config from POST body
  - Both endpoints now use dynamic model configuration

### 3. Documentation

#### `integration_guide.md`
- Complete step-by-step integration instructions
- API reference for JavaScript and backend
- Troubleshooting guide
- Styling customization options

## Quick Start Integration

### Step 1: Copy Files

```bash
# Copy HTML template
cp data_ingestion_model_selector.html app/templates/partials/

# Copy JavaScript
cp data_ingestion_model_selector.js app/static/
```

### Step 2: Update Settings Template

In `app/templates/settings.html`, add before scan/rebuild buttons:

```html
{% include 'partials/data_ingestion_model_selector.html' %}
```

And in the scripts section:

```html
<script src="{{ url_for('static', filename='data_ingestion_model_selector.js') }}"></script>
```

### Step 3: Update Backend Routes

Apply the changes from `backend_route_updates.py` to your `app/routes.py`:

- Update `scan_mounted_images()` to accept JSON config
- Update `rebuild_ingestion_database()` to accept JSON config

### Step 4: Update Frontend Scan/Rebuild Functions

In `app/static/settings_data_ingestion.js`:

```javascript
// Listen for config changes
let ingestionConfig = {};
document.addEventListener('ingestionConfigChange', (e) => {
    ingestionConfig = e.detail;
});

// Update scan to send config
async function triggerScan() {
    const validation = window.IngestionModelSelector.validateConfig();
    if (!validation.valid) {
        alert(validation.message);
        return;
    }
    
    const config = window.IngestionModelSelector.getConfig();
    const result = await fetch(scanUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
    // Handle result...
}
```

## Configuration Flow

### 1. Page Load
```javascript
init() → loadInitialConfig() → loadOllamaModels()
                                      ↓
                             GET /options?ollama_base_url=...
                                      ↓
                          Populate dropdown with models
```

### 2. User Changes Provider
```javascript
handleProviderChange() → updateProviderSections()
                                ↓
                    Show Ollama or OpenAI section
                                ↓
                         loadOllamaModels() (if Ollama)
```

### 3. User Clicks Scan/Rebuild
```javascript
validateConfig() → getConfig() → POST /scan or /rebuild
                                        ↓
                                  { provider, model, ... }
                                        ↓
                            Backend processes with config
```

## Key Features

### Dynamic Model Loading
- Automatically fetches models from Ollama API
- Refreshes on base URL change
- Manual refresh button for pulling new models

### Provider Flexibility
- Seamless switching between Ollama and OpenAI
- Provider-specific settings sections
- Validation for each provider

### User Feedback
- Loading states during model fetch
- Success/error messages
- Configuration validation before operations

### Event-Driven Architecture
- `ingestionConfigChange` event for integration
- Clean separation between model selector and scan/rebuild logic
- Easy to extend and customize

## API Endpoints

### GET `/settings/data-ingestion/options`

**Query Parameters:**
- `ollama_base_url` (optional): Ollama server URL

**Response:**
```json
{
  "default_provider": "ollama",
  "default_prompt": "Extract all text...",
  "ollama": {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.2-vision:11b",
    "models": ["llama3.2-vision:11b", "llava:13b"]
  },
  "openai": {
    "default_model": "gpt-4o-mini",
    "key_configured": true
  }
}
```

### POST `/settings/data-ingestion/scan`

**Request Body:**
```json
{
  "provider": "ollama",
  "model": "llama3.2-vision:11b",
  "ollama_base_url": "http://localhost:11434",
  "prompt": "Extract all text and analyze..."
}
```

**Response:**
```json
{
  "status": "ok",
  "action": "scan",
  "provider": "ollama",
  "model": "llama3.2-vision:11b",
  "results": [...],
  "aggregate": {
    "images_total": 45,
    "generated": 12,
    "skipped_existing": 30,
    "ingested": 12
  }
}
```

### POST `/settings/data-ingestion/rebuild`

Same request/response structure as scan, but processes all images.

## Configuration Object

```javascript
{
  provider: 'ollama',              // 'ollama' or 'openai'
  model: 'llama3.2-vision:11b',   // Model name
  ollama_base_url: 'http://...',   // Ollama server URL
  prompt: 'Extract all text...',   // System prompt
  
  // For OpenAI only:
  openai_model: 'gpt-4o-mini',     // OpenAI model
  api_key: 'sk-...'                // OpenAI API key
}
```

## Testing Checklist

- [ ] Models load on page load
- [ ] Provider switching works (Ollama ↔ OpenAI)
- [ ] Ollama base URL change triggers model reload
- [ ] Refresh button works
- [ ] Validation prevents scan/rebuild with invalid config
- [ ] Scan sends correct configuration to backend
- [ ] Rebuild sends correct configuration to backend
- [ ] Status messages appear appropriately
- [ ] Error handling works (Ollama offline, invalid API key)

## Common Issues & Solutions

### Models Not Loading

**Problem**: Dropdown shows "Loading models..." indefinitely

**Solutions**:
1. Check Ollama is running: `ollama list`
2. Verify base URL is correct (check Docker network settings)
3. Check browser console for errors
4. Try: `curl http://localhost:11434/api/tags`

### Configuration Not Sent to Backend

**Problem**: Scan/rebuild uses default model instead of selected

**Solutions**:
1. Verify `ingestionConfigChange` event is being captured
2. Check `getConfig()` returns correct values
3. Ensure fetch request includes `Content-Type: application/json`
4. Check backend is reading `request.get_json()`

### API Key Issues

**Problem**: OpenAI ingestion fails with authentication error

**Solutions**:
1. Verify API key format (should start with `sk-`)
2. Check key has credits remaining
3. Ensure backend `image_ingestion.ensure_api_key()` is called
4. Check Flask logs for specific error messages

## Customization

### Styling

All CSS is inline in the HTML template. To customize:

```css
/* Change primary colors */
.form-select:focus {
    border-color: #your-color;
    box-shadow: 0 0 0 3px rgba(your-color, 0.1);
}

/* Adjust spacing */
.ingestion-config__form {
    gap: 24px; /* Change to your preferred spacing */
}
```

### Model Filtering

To show only vision-capable models:

```javascript
// In populateOllamaModels()
const visionModels = models.filter(m => 
    m.includes('vision') || m.includes('llava')
);
```

### Additional Providers

To add more providers (e.g., Anthropic, Google):

1. Add option to provider select
2. Create provider-specific section
3. Handle in `handleProviderChange()`
4. Update `getConfig()` and `validateConfig()`

## Future Enhancements

- [ ] Save last-used configuration to localStorage
- [ ] Model performance metrics (speed, accuracy)
- [ ] Batch size configuration
- [ ] Temperature/max tokens sliders
- [ ] Model health check indicator
- [ ] Estimated processing time calculator
- [ ] Progress bars during scan/rebuild
- [ ] Model comparison tool

## Support

For questions or issues:

1. Check `integration_guide.md` for detailed instructions
2. Review browser console for JavaScript errors
3. Check Flask application logs for backend errors
4. Verify Ollama connectivity: `curl http://localhost:11434/api/tags`
5. Test with minimal configuration first

## License

Follows the same license as the main Historical Document Reader application.

---

**Created**: October 2025
**Version**: 1.0
**Compatibility**: Historical Document Reader v1.x
