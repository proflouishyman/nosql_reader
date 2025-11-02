/**
 * Added standalone ingestion model selector script to orchestrate provider switching and model loading for data ingestion.
 */
(function() {
    'use strict';

    const form = document.getElementById('ingestionConfigForm'); // Added lookup so the script only runs when the selector is present.
    const providerSelect = document.getElementById('ingestionProvider'); // Added provider reference to toggle Ollama/OpenAI sections.
    const ollamaSection = document.getElementById('ollamaSection'); // Added Ollama section hook so visibility can change dynamically.
    const openaiSection = document.getElementById('openaiSection'); // Added OpenAI section hook for the same reason.
    const ollamaBaseUrlInput = document.getElementById('ollamaBaseUrl'); // Added base URL field reference to trigger reloads.
    const ollamaModelSelect = document.getElementById('ollamaModel'); // Added model dropdown reference for population.
    const openaiModelSelect = document.getElementById('openaiModel'); // Added OpenAI dropdown reference to capture selection.
    const refreshModelsBtn = document.getElementById('refreshModelsBtn'); // Added refresh button hook for manual reloads.
    const configStatus = document.getElementById('configStatus'); // Added status node reference to display load and validation states.
    const promptInput = document.getElementById('ingestionPrompt'); // Added prompt field reference so updates propagate into config.

    const OPTIONS_ENDPOINT = '/settings/data-ingestion/options'; // Added endpoint constant so future refactors only change one string.

    let currentConfig = { // Added central state object to mirror the configuration sent to the backend.
        provider: providerSelect ? providerSelect.value : 'ollama',
        ollama_base_url: ollamaBaseUrlInput ? ollamaBaseUrlInput.value : '',
        model: '',
        openai_model: openaiModelSelect ? openaiModelSelect.value : '',
        api_key: ''
    };
    let lastLoadedBaseUrl = null; // Added cache marker so we only fetch models when necessary.

    if (!form) { // Added guard so script exits cleanly on pages without the selector.
        return;
    }

    function dispatchConfigChange() { // Added helper to emit configuration updates to other scripts.
        const event = new CustomEvent('ingestionConfigChange', { detail: getConfig() });
        document.dispatchEvent(event);
    }

    function showStatus(message, type) { // Added status renderer to surface success and error feedback beside the form.
        if (!configStatus) return;
        configStatus.textContent = message;
        configStatus.className = type ? `config-status ${type}` : 'config-status';
        configStatus.hidden = !message;
    }

    function hideStatus() { // Added hide helper so transient info can disappear automatically.
        if (!configStatus) return;
        configStatus.hidden = true;
    }

    function updateProviderSections() { // Added visibility toggle so only the relevant provider panel is shown.
        if (!ollamaSection || !openaiSection) return;
        if (currentConfig.provider === 'ollama') {
            ollamaSection.hidden = false;
            openaiSection.hidden = true;
        } else {
            ollamaSection.hidden = true;
            openaiSection.hidden = false;
        }
    }

    function applyOptionsDefaults(payload) { // Added parser so defaults from the backend populate the form when available.
        if (!payload) return;
        if (payload.default_provider && providerSelect) {
            providerSelect.value = payload.default_provider;
            currentConfig.provider = payload.default_provider;
        }
        if (payload.default_prompt && promptInput && !promptInput.value) {
            promptInput.value = payload.default_prompt;
        }
        if (payload.ollama && payload.ollama.base_url && ollamaBaseUrlInput) {
            if (!ollamaBaseUrlInput.value) {
                ollamaBaseUrlInput.value = payload.ollama.base_url;
            }
            currentConfig.ollama_base_url = ollamaBaseUrlInput.value;
        }
        if (payload.openai && payload.openai.default_model && openaiModelSelect) {
            openaiModelSelect.value = payload.openai.default_model;
            currentConfig.openai_model = openaiModelSelect.value; // Added mirror so OpenAI defaults update tracked state.
            currentConfig.model = currentConfig.openai_model; // Added sync so generic model matches documentation schema for OpenAI.
        }
    }

    function populateOllamaModels(models, defaultModel) { // Added dropdown population routine to render the list of Ollama models.
        if (!ollamaModelSelect) return;
        ollamaModelSelect.innerHTML = '';
        if (!models || !models.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            ollamaModelSelect.appendChild(option);
            showStatus('No Ollama models found. Ensure the runtime is running and exposes the tags endpoint.', 'error');
            currentConfig.model = '';
            dispatchConfigChange();
            return;
        }
        models.forEach(function(name) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            if (name === currentConfig.model || (!currentConfig.model && name === defaultModel)) {
                option.selected = true;
                currentConfig.model = name;
            }
            ollamaModelSelect.appendChild(option);
        });
        if (!currentConfig.model) {
            currentConfig.model = models[0];
            ollamaModelSelect.value = models[0];
        }
        showStatus(`Loaded ${models.length} Ollama model${models.length === 1 ? '' : 's'}.`, 'success');
        setTimeout(function() { hideStatus(); }, 3000); // Added auto-hide so transient success messages disappear.
        dispatchConfigChange();
    }

    async function fetchOptions(baseUrl) { // Added network call to retrieve models and defaults from the backend helper.
        const url = new URL(OPTIONS_ENDPOINT, window.location.origin);
        if (baseUrl) {
            url.searchParams.set('ollama_base_url', baseUrl);
        }
        const response = await fetch(url.toString());
        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }
        return response.json();
    }

    async function loadOllamaModels(force) { // Added loader function used on init and when the base URL changes.
        if (!ollamaModelSelect) return;
        const baseUrl = ollamaBaseUrlInput ? ollamaBaseUrlInput.value.trim() : '';
        if (!force && lastLoadedBaseUrl === baseUrl && ollamaModelSelect.options.length) {
            return;
        }
        lastLoadedBaseUrl = baseUrl;
        showStatus('Loading Ollama models…', 'info');
        ollamaModelSelect.disabled = true;
        ollamaModelSelect.innerHTML = '<option value="">Loading models...</option>';
        try {
            const data = await fetchOptions(baseUrl);
            applyOptionsDefaults(data);
            const models = data && data.ollama ? data.ollama.models : [];
            const defaultModel = data && data.ollama ? data.ollama.default_model : '';
            populateOllamaModels(models, defaultModel);
        } catch (error) {
            showStatus(`Failed to load Ollama models: ${error.message}`, 'error');
            ollamaModelSelect.innerHTML = '<option value="">Error loading models</option>';
            currentConfig.model = '';
            dispatchConfigChange();
        } finally {
            ollamaModelSelect.disabled = false;
        }
    }

    function getConfig() { // Added accessor so other scripts can POST the selected configuration to the backend.
        const config = {
            provider: currentConfig.provider,
            prompt: promptInput ? promptInput.value : ''
        };
        if (currentConfig.provider === 'ollama') {
            config.ollama_base_url = ollamaBaseUrlInput ? ollamaBaseUrlInput.value.trim() : '';
            config.model = currentConfig.model;
        } else {
            config.model = currentConfig.openai_model; // Added general model field so payload always includes `model` per docs.
            config.openai_model = currentConfig.openai_model;
            const apiKeyInput = document.getElementById('openaiApiKey');
            if (apiKeyInput && apiKeyInput.value) {
                config.api_key = apiKeyInput.value;
            }
        }
        return config;
    }

    function validateConfig() { // Added validation so scan/rebuild buttons cannot run with incomplete settings.
        const config = getConfig();
        if (config.provider === 'ollama') {
            if (!config.ollama_base_url) {
                return { valid: false, message: 'Provide an Ollama base URL before running ingestion.' };
            }
            if (!config.model) {
                return { valid: false, message: 'Select an Ollama model before running ingestion.' };
            }
        } else if (config.provider === 'openai') {
            if (!config.openai_model) {
                return { valid: false, message: 'Select an OpenAI model before running ingestion.' };
            }
        }
        return { valid: true, message: '' };
    }

    function handleProviderChange(event) { // Added change handler so switching providers toggles UI and revalidates state.
        currentConfig.provider = event.target.value;
        updateProviderSections();
        if (currentConfig.provider === 'ollama') {
            loadOllamaModels();
        } else {
            currentConfig.model = currentConfig.openai_model; // Added sync so switching to OpenAI keeps the shared model field accurate.
            dispatchConfigChange();
        }
    }

    function handleOllamaModelChange(event) { // Added handler so selecting a model updates the shared config state.
        currentConfig.model = event.target.value;
        dispatchConfigChange();
    }

    function handleOllamaUrlBlur(event) { // Added handler so base URL edits trigger a model reload when changed.
        const newUrl = event.target.value.trim();
        if (newUrl !== currentConfig.ollama_base_url) {
            currentConfig.ollama_base_url = newUrl;
            loadOllamaModels(true);
        }
    }

    function handleOpenAiModelChange(event) { // Added handler so the selected OpenAI model is tracked.
        currentConfig.openai_model = event.target.value;
        currentConfig.model = event.target.value; // Added sync so general model stays aligned with OpenAI selection changes.
        dispatchConfigChange();
    }

    function setupEventListeners() { // Added event wiring to connect DOM interactions to state changes.
        if (providerSelect) {
            providerSelect.addEventListener('change', handleProviderChange);
        }
        if (ollamaBaseUrlInput) {
            ollamaBaseUrlInput.addEventListener('blur', handleOllamaUrlBlur);
        }
        if (refreshModelsBtn) {
            refreshModelsBtn.addEventListener('click', function() {
                loadOllamaModels(true);
            });
        }
        if (ollamaModelSelect) {
            ollamaModelSelect.addEventListener('change', handleOllamaModelChange);
        }
        if (openaiModelSelect) {
            openaiModelSelect.addEventListener('change', handleOpenAiModelChange);
        }
        if (promptInput) {
            promptInput.addEventListener('input', dispatchConfigChange); // Added prompt listener so updates propagate immediately.
        }
    }

    async function init() { // Added bootstrapper to fetch defaults and wire up events on load.
        updateProviderSections();
        setupEventListeners();
        dispatchConfigChange();
        if (currentConfig.provider === 'ollama') {
            await loadOllamaModels(true);
        } else {
            try {
                const data = await fetchOptions(null);
                applyOptionsDefaults(data);
                dispatchConfigChange();
            } catch (error) {
                showStatus(`Failed to load ingestion defaults: ${error.message}`, 'error');
            }
        }
    }

    window.IngestionModelSelector = { // Added public API so the rest of the settings page can reuse validation and loaders.
        init,
        getConfig,
        validateConfig,
        loadOllamaModels,
        showStatus,
        hideStatus
    };

    if (document.readyState === 'loading') { // Added DOM ready hook so initialization mirrors other scripts.
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
