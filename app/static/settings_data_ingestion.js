(function() {
    function ready(fn) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fn);
        } else {
            fn();
        }
    }

    ready(function() {
        const form = document.getElementById('imageIngestionForm');
        if (!form) return;

        const root = form.closest('[data-ingestion-root]') || form.parentElement;
        const providerSelect = form.querySelector('#imageIngestionProvider');
        const ollamaSection = form.querySelector('[data-provider-section="ollama"]');
        const openAiSection = form.querySelector('[data-provider-section="openai"]');
        const baseUrlInput = form.querySelector('#imageIngestionOllamaBase');
        const modelSelect = form.querySelector('#imageIngestionModel');
        const openAiModelInput = form.querySelector('#imageIngestionOpenAiModel');
        const promptField = form.querySelector('#imageIngestionPrompt');
        const directoryInput = form.querySelector('#imageIngestionDirectory');
        const directoryHint = form.querySelector('[data-directory-picked]'); // change: Reference the hint element to show the current selection summary.
        const directoryList = form.querySelector('[data-directory-list]'); // change: Store the list element used to display the selected directories for confirmation.
        const copyModeInputs = form.querySelectorAll('input[name="ingestionCopyMode"]'); // change: Collect the copy mode radios to forward the chosen behaviour to the backend.
        const reprocessInput = form.querySelector('#imageIngestionReprocess');
        const apiKeyRow = form.querySelector('[data-api-key-row]');
        const apiKeyInput = form.querySelector('#imageIngestionApiKey');
        const apiKeyHint = form.querySelector('[data-api-key-hint]');
        const refreshButton = form.querySelector('[data-refresh-models]');
        const submitButton = form.querySelector('button[type="submit"]');

        const statusNode = root ? root.querySelector('[data-ingestion-status]') : null;
        const summaryNode = root ? root.querySelector('[data-ingestion-summary]') : null;
        const errorsNode = root ? root.querySelector('[data-ingestion-errors]') : null;

        const dataset = form.dataset || {};
        const optionsUrl = dataset.optionsUrl;
        const runUrl = dataset.runUrl;
        const defaultProvider = dataset.defaultProvider || 'ollama';
        const defaultOllamaModel = dataset.defaultOllamaModel || '';
        const defaultOpenaiModel = dataset.defaultOpenaiModel || '';
        const defaultOllamaUrl = dataset.defaultOllamaUrl || '';
        let keyConfigured = dataset.keyConfigured === '1';

        if (baseUrlInput && !baseUrlInput.value && defaultOllamaUrl) {
            baseUrlInput.value = defaultOllamaUrl;
        }
        if (openAiModelInput && !openAiModelInput.value && defaultOpenaiModel) {
            openAiModelInput.value = defaultOpenaiModel;
        }
        if (providerSelect && !providerSelect.value) {
            providerSelect.value = defaultProvider;
        }
        if (modelSelect && !modelSelect.value && defaultOllamaModel) {
            modelSelect.value = defaultOllamaModel;
        }

        function setStatus(message, type) {
            if (!statusNode) return;
            statusNode.textContent = message || '';
            statusNode.dataset.statusType = type || 'info';
            statusNode.hidden = !message;
        }

        function clearStatus() {
            setStatus('', 'info');
        }

        function clearSummary() {
            if (summaryNode) {
                summaryNode.innerHTML = '';
                summaryNode.hidden = true;
            }
            if (errorsNode) {
                errorsNode.innerHTML = '';
                errorsNode.hidden = true;
            }
        }

        function renderSummary(summary) {
            if (!summaryNode) return;
            const pairs = [
                ['Images discovered', summary.images_total],
                ['JSON generated', summary.generated],
                ['Skipped (already processed)', summary.skipped_existing],
                ['Queued for ingest', summary.queued_existing],
                ['Image processing failures', summary.failed],
                ['Database inserts', summary.ingested],
                ['Database updates', summary.updated],
                ['Ingestion failures', summary.ingest_failures]
            ];
            summaryNode.innerHTML = '';
            pairs.forEach(function(pair) {
                const dt = document.createElement('dt');
                dt.textContent = pair[0];
                const dd = document.createElement('dd');
                dd.textContent = typeof pair[1] === 'number' ? pair[1] : '0';
                summaryNode.appendChild(dt);
                summaryNode.appendChild(dd);
            });
            summaryNode.hidden = false;

            if (errorsNode) {
                errorsNode.innerHTML = '';
                const errors = Array.isArray(summary.errors) ? summary.errors : [];
                if (errors.length) {
                    const list = document.createElement('ul');
                    errors.slice(0, 5).forEach(function(error) {
                        const item = document.createElement('li');
                        if (error && typeof error === 'object') {
                            const path = error.path ? String(error.path) : '';
                            const message = error.error ? String(error.error) : '';
                            item.textContent = path ? `${path}: ${message}` : message;
                        } else {
                            item.textContent = String(error);
                        }
                        list.appendChild(item);
                    });
                    errorsNode.appendChild(list);
                    errorsNode.hidden = false;
                } else {
                    errorsNode.hidden = true;
                }
            }
        }

        function updateProviderVisibility() {
            const provider = providerSelect ? providerSelect.value : 'ollama';
            if (ollamaSection) {
                ollamaSection.hidden = provider !== 'ollama';
            }
            if (openAiSection) {
                openAiSection.hidden = provider !== 'openai';
            }
            if (apiKeyRow) {
                apiKeyRow.hidden = provider !== 'openai';
                if (provider === 'openai' && apiKeyInput) {
                    apiKeyInput.placeholder = keyConfigured ? 'API key configured' : 'Enter API key';
                }
                if (provider === 'openai' && apiKeyHint) {
                    apiKeyHint.textContent = keyConfigured
                        ? 'The stored key will be reused if left blank.'
                        : 'The key is stored outside the container with other environment files.';
                }
            }
        }

        async function fetchOptions(url) {
            const response = await fetch(url, { credentials: 'same-origin' });
            if (!response.ok) {
                throw new Error(`Options request failed with status ${response.status}`);
            }
            return response.json();
        }

        function applyOptions(data) {
            if (!data) return;
            if (data.openai && typeof data.openai.key_configured !== 'undefined') {
                keyConfigured = !!data.openai.key_configured;
            }
            if (baseUrlInput && data.ollama && data.ollama.base_url && !baseUrlInput.value) {
                baseUrlInput.value = data.ollama.base_url;
            }
            if (modelSelect && data.ollama && Array.isArray(data.ollama.models) && data.ollama.models.length) {
                const current = modelSelect.value;
                modelSelect.innerHTML = '';
                data.ollama.models.forEach(function(model) {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                if (current && data.ollama.models.includes(current)) {
                    modelSelect.value = current;
                } else if (data.ollama.default_model) {
                    modelSelect.value = data.ollama.default_model;
                }
            }
            updateProviderVisibility();
        }

        async function loadOptions() {
            if (!optionsUrl) return;
            try {
                const data = await fetchOptions(optionsUrl);
                applyOptions(data);
            } catch (error) {
                setStatus(`Unable to load ingestion options: ${error.message}`, 'error');
            }
        }

        async function refreshModels() {
            if (!optionsUrl) return;
            try {
                setStatus('Checking Ollama models…', 'info');
                const url = new URL(optionsUrl, window.location.origin);
                if (baseUrlInput && baseUrlInput.value.trim()) {
                    url.searchParams.set('ollama_base_url', baseUrlInput.value.trim());
                }
                const data = await fetchOptions(url.toString());
                applyOptions(data);
                setStatus('Ollama model list updated.', 'success');
            } catch (error) {
                setStatus(`Unable to refresh models: ${error.message}`, 'error');
            }
        }

        function parseDirectoriesFromInput(value) {
            // change: Split the textarea so each pasted path is processed individually without relying on a browser picker.
            if (!value) {
                return [];
            }
            const seen = new Set(); // change: Deduplicate entries to prevent submitting the same folder multiple times.
            return value
                .split(/\r?\n|,/)
                .map(function(part) { return part.trim(); })
                .filter(function(part) {
                    if (!part.length || seen.has(part)) {
                        return false;
                    }
                    seen.add(part);
                    return true;
                });
        }

        function currentCopyMode() {
            // change: Determine whether the user wants to copy into the archives or run in place.
            let mode = 'in_place';
            copyModeInputs.forEach(function(radio) {
                if (radio.checked) {
                    mode = radio.value || mode;
                }
            });
            return mode;
        }

        function updateDirectoryDisplay(folders) {
            // change: Summarise the manually entered directories to confirm what will be processed.
            if (directoryHint) {
                if (folders.length) {
                    directoryHint.textContent = `Queued ${folders.length} director${folders.length > 1 ? 'ies' : 'y'} for ingestion.`;
                    directoryHint.hidden = false;
                } else {
                    directoryHint.hidden = true;
                    directoryHint.textContent = '';
                }
            }
            if (directoryList) {
                directoryList.innerHTML = '';
                if (folders.length) {
                    folders.forEach(function(folder) {
                        const item = document.createElement('li');
                        item.textContent = folder;
                        directoryList.appendChild(item);
                    });
                    directoryList.hidden = false;
                } else {
                    directoryList.hidden = true;
                }
            }
        }

        function syncDisplayFromManualInput() {
            // change: Keep the preview list accurate when the user edits the field manually.
            if (!directoryInput) {
                return;
            }
            const folders = parseDirectoriesFromInput(directoryInput.value);
            updateDirectoryDisplay(folders); // change: Reflect the current directory list without relying on the deprecated native picker metadata.
        }

        async function submitForm(event) {
            event.preventDefault();
            clearStatus();
            clearSummary();

            if (!runUrl) {
                setStatus('Ingestion endpoint not available.', 'error');
                return;
            }
            const directories = parseDirectoriesFromInput(directoryInput ? directoryInput.value : ''); // change: Allow multiple folder paths to be posted together.
            const payload = {
                directory: directories.length ? directories[0] : '', // change: Preserve the legacy single-directory field for backwards compatibility.
                directories: directories, // change: Send the expanded directory list so the backend can ingest multiple folders.
                provider: providerSelect ? providerSelect.value : defaultProvider,
                prompt: promptField ? promptField.value : '',
                reprocess: reprocessInput ? reprocessInput.checked : false,
                copy_mode: currentCopyMode(), // change: Forward the selected copy behaviour to the server.
            };

            if (baseUrlInput) {
                payload.base_url = baseUrlInput.value.trim();
                payload.ollama_base_url = payload.base_url;
            }
            if (payload.provider === 'ollama') {
                if (modelSelect) {
                    payload.model = modelSelect.value;
                }
            } else if (payload.provider === 'openai') {
                if (openAiModelInput) {
                    payload.openai_model = openAiModelInput.value.trim();
                }
                if (apiKeyInput && apiKeyInput.value.trim()) {
                    payload.api_key = apiKeyInput.value.trim();
                }
            }

            if (!payload.directory && (!payload.directories || !payload.directories.length)) {
                setStatus('Enter the archive directory to ingest.', 'error');
                directoryInput && directoryInput.focus();
                return;
            }

            try {
                if (submitButton) {
                    submitButton.disabled = true;
                }
                setStatus('Starting ingestion…', 'info');
                const response = await fetch(runUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    credentials: 'same-origin',
                    body: JSON.stringify(payload),
                });
                const result = await response.json().catch(() => ({}));
                if (!response.ok || result.status === 'error') {
                    const message = result && result.message ? result.message : `Ingestion failed with status ${response.status}`;
                    setStatus(message, 'error');
                    if (result && result.code === 'missing_api_key') {
                        keyConfigured = false;
                        updateProviderVisibility();
                    }
                    return;
                }

                if (result.api_key_saved) {
                    keyConfigured = true;
                    if (apiKeyInput) {
                        apiKeyInput.value = '';
                    }
                }
                updateProviderVisibility();

                if (result.summary) {
                    renderSummary(result.summary);
                }
                setStatus('Ingestion completed successfully.', 'success');
            } catch (error) {
                setStatus(`Ingestion failed: ${error.message}`, 'error');
            } finally {
                if (submitButton) {
                    submitButton.disabled = false;
                }
            }
        }

        if (refreshButton) {
            refreshButton.addEventListener('click', function(event) {
                event.preventDefault();
                refreshModels();
            });
        }

        if (providerSelect) {
            providerSelect.addEventListener('change', updateProviderVisibility);
        }

        form.addEventListener('submit', submitForm);

        if (directoryInput) {
            // change: Rebuild the preview list as the user edits the directory field manually.
            directoryInput.addEventListener('input', syncDisplayFromManualInput);
        }

        updateProviderVisibility();
        loadOptions();
        syncDisplayFromManualInput(); // change: Ensure any prefilled value populates the folder preview on load.
    });
})();
