(function() {
    // Settings page controller for the "Data Ingestion" pane. The script wires
    // the folder picker, provider controls, and progress feedback to the Flask
    // endpoints exposed in ``routes.py``.
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

        // Cache useful elements so the rest of the module can toggle visibility
        // and update values without repeatedly querying the DOM.
        const root = form.closest('[data-ingestion-root]') || form.parentElement;
        const providerSelect = form.querySelector('#imageIngestionProvider');
        const ollamaSection = form.querySelector('[data-provider-section="ollama"]');
        const openAiSection = form.querySelector('[data-provider-section="openai"]');
        const baseUrlInput = form.querySelector('#imageIngestionOllamaBase');
        const modelSelect = form.querySelector('#imageIngestionModel');
        const openAiModelInput = form.querySelector('#imageIngestionOpenAiModel');
        const promptField = form.querySelector('#imageIngestionPrompt');
        const directoryInput = form.querySelector('#imageIngestionDirectory');
        const directoryPicker = form.querySelector('#imageIngestionDirectoryPicker');
        const directoryButton = form.querySelector('[data-directory-picker]');
        const directoryAddButton = form.querySelector('[data-directory-add]');
        const directoryList = form.querySelector('[data-directory-list]');
        const directoryEmpty = form.querySelector('[data-directory-empty]');
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

        // Apply defaults from the server-rendered dataset so the form mirrors
        // the configuration currently persisted on disk.
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
            // Helper to show short feedback messages inline with the form.
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
            // Display the ``IngestionSummary`` payload returned from the backend
            // as a definition list, mirroring the metrics tracked in Python.
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
            // Toggle provider-specific controls depending on the selected radio.
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
            // Shared helper to call the Flask options endpoint.
            const response = await fetch(url, { credentials: 'same-origin' });
            if (!response.ok) {
                throw new Error(`Options request failed with status ${response.status}`);
            }
            return response.json();
        }

        function applyOptions(data) {
            // Update the UI using the payload returned from ``/options``.
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
            // Initialisation step to fetch Ollama models when the page loads.
            if (!optionsUrl) return;
            try {
                const data = await fetchOptions(optionsUrl);
                applyOptions(data);
            } catch (error) {
                setStatus(`Unable to load ingestion options: ${error.message}`, 'error');
            }
        }

        async function refreshModels() {
            // Manual refresh hook so the operator can pick up newly pulled
            // Ollama models without reloading the entire settings page.
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

        function deriveDirectoryFromFiles(files) {
            // The folder picker returns a list of files; infer the common path
            // prefix so we can populate the directory text field automatically.
            if (!files || !files.length) return '';
            const dirSegments = [];
            files.forEach(function(file) {
                const relative = file.webkitRelativePath || '';
                if (!relative) {
                    return;
                }
                const parts = relative.split('/');
                if (!parts.length) {
                    return;
                }
                parts.pop();
                if (parts.length) {
                    dirSegments.push(parts);
                }
            });
            if (!dirSegments.length) {
                const fallback = files[0].webkitRelativePath || '';
                const idx = fallback.lastIndexOf('/');
                return idx > 0 ? fallback.slice(0, idx) : '';
            }
            let prefix = dirSegments[0].slice();
            for (let i = 1; i < dirSegments.length; i += 1) {
                const current = dirSegments[i];
                let j = 0;
                while (j < prefix.length && j < current.length && prefix[j] === current[j]) {
                    j += 1;
                }
                prefix = prefix.slice(0, j);
                if (!prefix.length) {
                    break;
                }
            }
            return prefix.join('/');
        }

        const selectedDirectories = [];
        const directoryMeta = new Map();
        let processing = false;

        function ensureDirectoryMeta(path) {
            if (!directoryMeta.has(path)) {
                directoryMeta.set(path, { status: 'pending', message: '', summary: null });
            }
            return directoryMeta.get(path);
        }

        function renderDirectoryList() {
            if (!directoryList) return;
            directoryList.innerHTML = '';
            selectedDirectories.forEach(function(path) {
                const meta = ensureDirectoryMeta(path);
                const item = document.createElement('li');
                item.className = 'directory-selection__item';
                item.dataset.state = meta.status || 'pending';

                const info = document.createElement('div');
                info.className = 'directory-selection__info';

                const pathLabel = document.createElement('span');
                pathLabel.className = 'directory-selection__path';
                pathLabel.textContent = path;
                info.appendChild(pathLabel);

                const statusLabel = document.createElement('span');
                statusLabel.className = 'directory-selection__status';
                statusLabel.textContent = {
                    running: 'Processing…',
                    success: 'Complete',
                    error: 'Failed',
                    pending: 'Pending',
                }[meta.status || 'pending'];
                info.appendChild(statusLabel);

                if (meta.message) {
                    const messageNode = document.createElement('span');
                    messageNode.className = 'directory-selection__message';
                    messageNode.textContent = meta.message;
                    info.appendChild(messageNode);
                }

                item.appendChild(info);

                const removeButton = document.createElement('button');
                removeButton.type = 'button';
                removeButton.className = 'button button--icon';
                removeButton.dataset.directoryRemove = path;
                removeButton.title = 'Remove folder';
                removeButton.disabled = processing;
                removeButton.innerHTML = '&times;';
                item.appendChild(removeButton);

                directoryList.appendChild(item);
            });

            if (directoryEmpty) {
                directoryEmpty.hidden = selectedDirectories.length > 0;
            }
        }

        function addDirectory(path) {
            const trimmed = (path || '').trim();
            if (!trimmed) {
                return false;
            }
            if (selectedDirectories.includes(trimmed)) {
                setStatus(`Folder already selected: ${trimmed}`, 'info');
                return false;
            }
            selectedDirectories.push(trimmed);
            directoryMeta.set(trimmed, { status: 'pending', message: '', summary: null });
            renderDirectoryList();
            clearStatus();
            return true;
        }

        function removeDirectory(path) {
            const index = selectedDirectories.indexOf(path);
            if (index === -1) {
                return;
            }
            selectedDirectories.splice(index, 1);
            directoryMeta.delete(path);
            renderDirectoryList();
        }

        function resetDirectoryStates() {
            selectedDirectories.forEach(function(path) {
                const meta = ensureDirectoryMeta(path);
                meta.status = 'pending';
                meta.message = '';
                meta.summary = null;
            });
            renderDirectoryList();
        }

        function setDirectoryState(path, status, message, summary) {
            const meta = ensureDirectoryMeta(path);
            meta.status = status;
            meta.message = message || '';
            meta.summary = summary || null;
            renderDirectoryList();
        }

        function combineSummaries(target, source) {
            if (!source) return target;
            const numericKeys = [
                'images_total',
                'generated',
                'skipped_existing',
                'queued_existing',
                'failed',
                'ingested',
                'updated',
                'ingest_failures',
            ];
            numericKeys.forEach(function(key) {
                const value = Number(source[key] || 0);
                target[key] = (target[key] || 0) + (Number.isFinite(value) ? value : 0);
            });
            const errors = Array.isArray(source.errors) ? source.errors : [];
            if (!Array.isArray(target.errors)) {
                target.errors = [];
            }
            Array.prototype.push.apply(target.errors, errors);
            return target;
        }

        function emptySummary() {
            return {
                images_total: 0,
                generated: 0,
                skipped_existing: 0,
                queued_existing: 0,
                failed: 0,
                ingested: 0,
                updated: 0,
                ingest_failures: 0,
                errors: [],
            };
        }

        function formatSummary(summary) {
            if (!summary) return '';
            const generated = Number(summary.generated || 0);
            const ingested = Number(summary.ingested || 0);
            const updated = Number(summary.updated || 0);
            const skipped = Number(summary.skipped_existing || 0);
            const failed = Number(summary.failed || 0) + Number(summary.ingest_failures || 0);
            const parts = [];
            if (generated) parts.push(`${generated} new JSON`);
            if (ingested) parts.push(`${ingested} inserted`);
            if (updated) parts.push(`${updated} updated`);
            if (skipped) parts.push(`${skipped} skipped`);
            if (failed) parts.push(`${failed} errors`);
            return parts.length ? parts.join(', ') : 'No files processed.';
        }

        async function submitForm(event) {
            event.preventDefault();
            clearStatus();
            clearSummary();

            if (processing) {
                return;
            }

            if (!runUrl) {
                setStatus('Ingestion endpoint not available.', 'error');
                return;
            }

            if (directoryInput && directoryInput.value.trim()) {
                if (addDirectory(directoryInput.value.trim())) {
                    directoryInput.value = '';
                }
            }

            if (!selectedDirectories.length) {
                setStatus('Select at least one folder to ingest.', 'error');
                if (directoryInput) {
                    directoryInput.focus();
                }
                return;
            }

            resetDirectoryStates();

            const directoriesToProcess = selectedDirectories.slice();
            const basePayload = {
                provider: providerSelect ? providerSelect.value : defaultProvider,
                prompt: promptField ? promptField.value : '',
                reprocess: reprocessInput ? reprocessInput.checked : false,
            };

            if (baseUrlInput) {
                basePayload.base_url = baseUrlInput.value.trim();
                basePayload.ollama_base_url = basePayload.base_url;
            }
            if (basePayload.provider === 'ollama') {
                if (modelSelect) {
                    basePayload.model = modelSelect.value;
                }
            } else if (basePayload.provider === 'openai') {
                if (openAiModelInput) {
                    basePayload.openai_model = openAiModelInput.value.trim();
                }
                if (apiKeyInput && apiKeyInput.value.trim()) {
                    basePayload.api_key = apiKeyInput.value.trim();
                }
            }

            let aggregatedSummary = emptySummary();
            let successfulRuns = 0;
            let failedRuns = 0;
            processing = true;

            if (submitButton) submitButton.disabled = true;
            if (directoryButton) directoryButton.disabled = true;
            if (directoryAddButton) directoryAddButton.disabled = true;
            if (directoryPicker) directoryPicker.disabled = true;

            try {
                for (let index = 0; index < directoriesToProcess.length; index += 1) {
                    const directory = directoriesToProcess[index];
                    setDirectoryState(directory, 'running', '');
                    setStatus(`Processing folder ${index + 1} of ${directoriesToProcess.length}: ${directory}`, 'info');

                    const payload = Object.assign({ directory }, basePayload);

                    let response;
                    let result = {};
                    try {
                        response = await fetch(runUrl, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            credentials: 'same-origin',
                            body: JSON.stringify(payload),
                        });
                        result = await response.json().catch(() => ({}));
                    } catch (networkError) {
                        setDirectoryState(
                            directory,
                            'error',
                            `Request failed: ${networkError && networkError.message ? networkError.message : networkError}`
                        );
                        failedRuns += 1;
                        continue;
                    }

                    if (!response.ok || result.status === 'error') {
                        const message = result && result.message
                            ? result.message
                            : `Ingestion failed with status ${response.status}`;
                        setDirectoryState(directory, 'error', message);
                        failedRuns += 1;
                        if (result && result.code === 'missing_api_key') {
                            keyConfigured = false;
                            updateProviderVisibility();
                            setStatus(message, 'error');
                            break;
                        }
                        continue;
                    }

                    if (result.api_key_saved) {
                        keyConfigured = true;
                        if (apiKeyInput) {
                            apiKeyInput.value = '';
                        }
                    }
                    updateProviderVisibility();

                    const summary = result.summary || null;
                    aggregatedSummary = combineSummaries(aggregatedSummary, summary);
                    successfulRuns += 1;
                    setDirectoryState(directory, 'success', formatSummary(summary), summary);
                }

                if (successfulRuns > 0) {
                    renderSummary(aggregatedSummary);
                    if (failedRuns > 0) {
                        setStatus('Ingestion completed with some errors. Review the folder list for details.', 'error');
                    } else {
                        setStatus('Ingestion completed successfully.', 'success');
                    }
                } else {
                    setStatus('Ingestion finished with no successful folders.', 'error');
                }
            } finally {
                processing = false;
                if (submitButton) submitButton.disabled = false;
                if (directoryButton) directoryButton.disabled = false;
                if (directoryAddButton) directoryAddButton.disabled = false;
                if (directoryPicker) directoryPicker.disabled = false;
                renderDirectoryList();
            }
        }

        if (directoryButton && directoryPicker) {
            directoryButton.addEventListener('click', function(event) {
                event.preventDefault();
                directoryPicker.click();
            });
            directoryPicker.addEventListener('change', function() {
                const files = Array.from(directoryPicker.files || []);
                if (!files.length) {
                    return;
                }
                const directory = deriveDirectoryFromFiles(files);
                if (directory) {
                    addDirectory(directory);
                } else {
                    setStatus('Unable to determine the selected folder. Please enter it manually.', 'error');
                }
                directoryPicker.value = '';
            });
        }

        if (directoryAddButton) {
            directoryAddButton.addEventListener('click', function(event) {
                event.preventDefault();
                if (!directoryInput) return;
                if (addDirectory(directoryInput.value)) {
                    directoryInput.value = '';
                }
            });
        }

        if (directoryList) {
            directoryList.addEventListener('click', function(event) {
                const target = event.target;
                if (!(target instanceof HTMLElement)) return;
                const path = target.dataset.directoryRemove;
                if (path) {
                    event.preventDefault();
                    if (!processing) {
                        removeDirectory(path);
                    }
                }
            });
        }

        if (directoryInput) {
            directoryInput.addEventListener('keydown', function(event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    if (addDirectory(directoryInput.value)) {
                        directoryInput.value = '';
                    }
                }
            });
        }

        if (refreshButton) {
            refreshButton.addEventListener('click', function(event) {
                event.preventDefault();
                refreshModels();
            });
        }

        renderDirectoryList();

        if (providerSelect) {
            providerSelect.addEventListener('change', updateProviderVisibility);
        }

        form.addEventListener('submit', submitForm);

        updateProviderVisibility();
        loadOptions();
    });
})();
