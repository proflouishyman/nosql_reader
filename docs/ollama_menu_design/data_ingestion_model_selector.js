/**
 * Data Ingestion Model Selector
 * Handles Ollama model dropdown and provider switching for data ingestion
 */

(function() {
    'use strict';

    // DOM elements
    const form = document.getElementById('ingestionConfigForm');
    const providerSelect = document.getElementById('ingestionProvider');
    const ollamaSection = document.getElementById('ollamaSection');
    const openaiSection = document.getElementById('openaiSection');
    const ollamaBaseUrlInput = document.getElementById('ollamaBaseUrl');
    const ollamaModelSelect = document.getElementById('ollamaModel');
    const openaiModelSelect = document.getElementById('openaiModel');
    const refreshModelsBtn = document.getElementById('refreshModelsBtn');
    const configStatus = document.getElementById('configStatus');

    // Configuration
    const OPTIONS_ENDPOINT = '/settings/data-ingestion/options';
    
    // State
    let currentConfig = {
        provider: 'ollama',
        ollama_base_url: ollamaBaseUrlInput?.value || 'http://localhost:11434',
        model: '',
        openai_model: 'gpt-4o-mini',
        api_key: ''
    };

    /**
     * Initialize the component
     */
    function init() {
        if (!form) {
            console.warn('Ingestion config form not found');
            return;
        }

        // Load initial configuration
        loadInitialConfig();

        // Set up event listeners
        setupEventListeners();

        // Load Ollama models on startup
        if (currentConfig.provider === 'ollama') {
            loadOllamaModels();
        }
    }

    /**
     * Set up event listeners
     */
    function setupEventListeners() {
        // Provider switching
        if (providerSelect) {
            providerSelect.addEventListener('change', handleProviderChange);
        }

        // Ollama base URL change
        if (ollamaBaseUrlInput) {
            ollamaBaseUrlInput.addEventListener('blur', handleOllamaUrlChange);
        }

        // Refresh models button
        if (refreshModelsBtn) {
            refreshModelsBtn.addEventListener('click', () => {
                loadOllamaModels(true);
            });
        }

        // Model selection
        if (ollamaModelSelect) {
            ollamaModelSelect.addEventListener('change', (e) => {
                currentConfig.model = e.target.value;
                updateScanAndRebuildButtons();
            });
        }

        if (openaiModelSelect) {
            openaiModelSelect.addEventListener('change', (e) => {
                currentConfig.openai_model = e.target.value;
                updateScanAndRebuildButtons();
            });
        }
    }

    /**
     * Load initial configuration from the page
     */
    function loadInitialConfig() {
        if (providerSelect) {
            currentConfig.provider = providerSelect.value;
        }
        
        if (ollamaBaseUrlInput) {
            currentConfig.ollama_base_url = ollamaBaseUrlInput.value;
        }

        // Show appropriate section
        updateProviderSections();
    }

    /**
     * Handle provider change (Ollama vs OpenAI)
     */
    function handleProviderChange(e) {
        currentConfig.provider = e.target.value;
        updateProviderSections();

        if (currentConfig.provider === 'ollama') {
            loadOllamaModels();
        }

        updateScanAndRebuildButtons();
    }

    /**
     * Update visibility of provider-specific sections
     */
    function updateProviderSections() {
        if (ollamaSection && openaiSection) {
            if (currentConfig.provider === 'ollama') {
                ollamaSection.hidden = false;
                openaiSection.hidden = true;
            } else {
                ollamaSection.hidden = true;
                openaiSection.hidden = false;
            }
        }
    }

    /**
     * Handle Ollama base URL change
     */
    function handleOllamaUrlChange(e) {
        const newUrl = e.target.value.trim();
        if (newUrl && newUrl !== currentConfig.ollama_base_url) {
            currentConfig.ollama_base_url = newUrl;
            loadOllamaModels(true);
        }
    }

    /**
     * Load available Ollama models from the server
     * @param {boolean} force - Force refresh even if models are already loaded
     */
    async function loadOllamaModels(force = false) {
        if (!ollamaModelSelect) return;

        // Check if already loaded and not forcing refresh
        if (!force && ollamaModelSelect.options.length > 1) {
            return;
        }

        // Show loading state
        showStatus('Loading Ollama models...', 'info');
        ollamaModelSelect.disabled = true;
        ollamaModelSelect.innerHTML = '<option value="">Loading models...</option>';

        try {
            // Build URL with base_url parameter
            const url = new URL(OPTIONS_ENDPOINT, window.location.origin);
            if (currentConfig.ollama_base_url) {
                url.searchParams.set('ollama_base_url', currentConfig.ollama_base_url);
            }

            const response = await fetch(url.toString());
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Populate dropdown with models
            populateOllamaModels(data.ollama.models, data.ollama.default_model);
            
            hideStatus();

        } catch (error) {
            console.error('Failed to load Ollama models:', error);
            showStatus(
                `Failed to load Ollama models: ${error.message}. Check if Ollama is running at ${currentConfig.ollama_base_url}`,
                'error'
            );
            
            // Show error state in dropdown
            ollamaModelSelect.innerHTML = '<option value="">Error loading models</option>';
        } finally {
            ollamaModelSelect.disabled = false;
        }
    }

    /**
     * Populate the Ollama model dropdown
     * @param {string[]} models - Array of model names
     * @param {string} defaultModel - Default model name
     */
    function populateOllamaModels(models, defaultModel) {
        if (!ollamaModelSelect) return;

        // Clear existing options
        ollamaModelSelect.innerHTML = '';

        if (!models || models.length === 0) {
            ollamaModelSelect.innerHTML = '<option value="">No models available</option>';
            showStatus('No Ollama models found. Pull a vision model with: ollama pull llama3.2-vision', 'error');
            return;
        }

        // Add models to dropdown
        models.forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            
            // Select default or current model
            if (modelName === defaultModel || modelName === currentConfig.model) {
                option.selected = true;
                currentConfig.model = modelName;
            }
            
            ollamaModelSelect.appendChild(option);
        });

        // If no model is selected, select the first one
        if (!currentConfig.model && models.length > 0) {
            currentConfig.model = models[0];
            ollamaModelSelect.value = models[0];
        }

        showStatus(`Loaded ${models.length} Ollama model${models.length !== 1 ? 's' : ''}`, 'success');
        
        // Auto-hide success message after 3 seconds
        setTimeout(() => {
            if (configStatus && configStatus.classList.contains('success')) {
                hideStatus();
            }
        }, 3000);
    }

    /**
     * Show status message
     * @param {string} message - Message to display
     * @param {string} type - Type: 'success', 'error', or 'info'
     */
    function showStatus(message, type = 'info') {
        if (!configStatus) return;
        
        configStatus.textContent = message;
        configStatus.className = `config-status ${type}`;
        configStatus.hidden = false;
    }

    /**
     * Hide status message
     */
    function hideStatus() {
        if (!configStatus) return;
        configStatus.hidden = true;
    }

    /**
     * Get current configuration for data ingestion
     * @returns {object} Current configuration
     */
    function getConfig() {
        const config = {
            provider: currentConfig.provider,
            prompt: document.getElementById('ingestionPrompt')?.value || ''
        };

        if (currentConfig.provider === 'ollama') {
            config.ollama_base_url = currentConfig.ollama_base_url;
            config.model = currentConfig.model;
        } else {
            config.openai_model = currentConfig.openai_model;
            const apiKeyInput = document.getElementById('openaiApiKey');
            if (apiKeyInput && apiKeyInput.value) {
                config.api_key = apiKeyInput.value;
            }
        }

        return config;
    }

    /**
     * Update scan and rebuild buttons with current configuration
     * This should be called by the settings_data_ingestion.js module
     */
    function updateScanAndRebuildButtons() {
        // Dispatch custom event that settings_data_ingestion.js can listen to
        const event = new CustomEvent('ingestionConfigChange', {
            detail: getConfig()
        });
        document.dispatchEvent(event);
    }

    /**
     * Validate configuration before running ingestion
     * @returns {object} { valid: boolean, message: string }
     */
    function validateConfig() {
        const config = getConfig();

        if (config.provider === 'ollama') {
            if (!config.ollama_base_url) {
                return { valid: false, message: 'Ollama base URL is required' };
            }
            if (!config.model) {
                return { valid: false, message: 'Please select an Ollama model' };
            }
        } else if (config.provider === 'openai') {
            if (!config.openai_model) {
                return { valid: false, message: 'Please select an OpenAI model' };
            }
            // API key validation happens on the backend
        }

        return { valid: true, message: '' };
    }

    // Public API
    window.IngestionModelSelector = {
        init,
        getConfig,
        validateConfig,
        loadOllamaModels,
        showStatus,
        hideStatus
    };

    // Auto-initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
