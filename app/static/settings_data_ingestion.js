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
        const directoryPickerButton = form.querySelector('[data-directory-picker]'); // change: Capture the visible button that now always opens the API-backed browser overlay.
        const directoryHint = form.querySelector('[data-directory-picked]'); // change: Reference the hint element to show the current selection summary.
        const directoryList = form.querySelector('[data-directory-list]'); // change: Store the list element used to display the selected directories for confirmation.
        const directoryBrowser = root ? root.querySelector('[data-directory-browser]') : null; // change: Locate the overlay that renders server-side directory listings.
        const browserList = directoryBrowser ? directoryBrowser.querySelector('[data-browser-list]') : null; // change: Reference the list container for browse results.
        const browserPathInput = directoryBrowser ? directoryBrowser.querySelector('[data-browser-path]') : null; // change: Allow navigation to arbitrary paths supplied by the user.
        const browserError = directoryBrowser ? directoryBrowser.querySelector('[data-browser-error]') : null; // change: Surface backend errors encountered during browsing.
        const browserApplyButton = directoryBrowser ? directoryBrowser.querySelector('[data-browser-apply]') : null; // change: Confirm the selected folders and write them into the ingestion form.
        const browserUpButton = directoryBrowser ? directoryBrowser.querySelector('[data-browser-up]') : null; // change: Provide parent-directory navigation from the overlay.
        const browserJumpButton = directoryBrowser ? directoryBrowser.querySelector('[data-browser-jump]') : null; // change: Trigger API lookups for manually typed paths.
        const browserDismissButtons = directoryBrowser ? directoryBrowser.querySelectorAll('[data-browser-dismiss]') : []; // change: Close the overlay when the backdrop or header control is activated.
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
        const browseUrl = dataset.browseUrl || ''; // change: Store the server endpoint that enumerates directories for the UI picker.
        const archiveRoot = dataset.archiveRoot || ''; // change: Remember the configured archive root so the browser has a sensible default location.
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
            // change: Split the directory input so both manual entries and picker selections can be ingested together.
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
            // change: Summarise the selected directories using the backend browser or manual entries without relying on the deprecated native picker.
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

        function setBrowserError(message) {
            // change: Centralise browser error handling so API failures can be surfaced consistently.
            if (!browserError) {
                return;
            }
            if (message) {
                browserError.textContent = message;
                browserError.hidden = false;
            } else {
                browserError.textContent = '';
                browserError.hidden = true;
            }
        }

        function renderBrowserEntries(entries) {
            // change: Populate the overlay list with the directories returned by the backend endpoint.
            if (!browserList) {
                return;
            }
            const existing = new Set(parseDirectoriesFromInput(directoryInput ? directoryInput.value : ''));
            browserList.innerHTML = '';
            if (!entries.length) {
                const empty = document.createElement('li');
                empty.textContent = 'No subdirectories found in this location.';
                browserList.appendChild(empty);
                return;
            }
            entries.forEach(function(entry) {
                const item = document.createElement('li');
                item.className = 'settings-directory-browser__item';
                const label = document.createElement('label');
                label.className = 'checkbox';
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.value = entry.path;
                checkbox.setAttribute('data-browser-select', '1');
                if (existing.has(entry.path)) {
                    checkbox.checked = true;
                }
                const nameSpan = document.createElement('span');
                nameSpan.textContent = entry.name || entry.path;
                label.appendChild(checkbox);
                label.appendChild(nameSpan);
                const openButton = document.createElement('button');
                openButton.type = 'button';
                openButton.className = 'button button--secondary';
                openButton.textContent = 'Open';
                openButton.dataset.browserEnter = entry.path;
                item.appendChild(label);
                item.appendChild(openButton);
                browserList.appendChild(item);
            });
        }

        async function loadBrowserPath(targetPath) {
            // change: Query the Flask endpoint for the specified directory and refresh the overlay with the results.
            if (!browseUrl) {
                return;
            }
            const trimmed = targetPath && typeof targetPath === 'string' ? targetPath.trim() : '';
            const url = new URL(browseUrl, window.location.origin);
            if (trimmed) {
                url.searchParams.set('path', trimmed);
            }
            setBrowserError('');
            try {
                const response = await fetch(url.toString(), { credentials: 'same-origin' });
                const payload = await response.json().catch(function() { return {}; });
                if (!response.ok || (payload && payload.status === 'error')) {
                    const message = payload && payload.message ? payload.message : `Listing failed with status ${response.status}`;
                    setBrowserError(message);
                    return;
                }
                if (browserPathInput) {
                    browserPathInput.value = payload.path || trimmed;
                }
                renderBrowserEntries(Array.isArray(payload.entries) ? payload.entries : []);
                if (browserUpButton) {
                    const parentPath = payload && payload.parent && payload.parent !== payload.path ? payload.parent : '';
                    browserUpButton.disabled = !parentPath;
                    if (parentPath) {
                        browserUpButton.dataset.browserParent = parentPath;
                    } else {
                        delete browserUpButton.dataset.browserParent;
                    }
                }
            } catch (error) {
                setBrowserError(`Unable to list directories: ${error.message}`);
            }
        }

        function openDirectoryBrowser(initialPath) {
            // change: Notify the user to type the path manually when the API-backed browser endpoint is unavailable.
            if (!directoryBrowser || !browseUrl) {
                setStatus('Directory browser unavailable. Enter the full path manually.', 'warning');
                return;
            }
            directoryBrowser.hidden = false;
            const startingPath = initialPath && initialPath.trim()
                ? initialPath.trim()
                : (browserPathInput && browserPathInput.value.trim())
                    ? browserPathInput.value.trim()
                    : archiveRoot;
            loadBrowserPath(startingPath || archiveRoot);
        }

        function closeDirectoryBrowser() {
            // change: Hide the overlay after selection or cancellation.
            if (!directoryBrowser) {
                return;
            }
            directoryBrowser.hidden = true;
            setBrowserError('');
        }

        function applyBrowserSelection() {
            // change: Merge the checked directories into the main input field and refresh the preview list.
            if (!browserList || !directoryInput) {
                closeDirectoryBrowser();
                return;
            }
            const selections = browserList.querySelectorAll('input[type="checkbox"][data-browser-select]');
            const aggregate = new Set(parseDirectoriesFromInput(directoryInput.value));
            selections.forEach(function(checkbox) {
                if (checkbox.checked && checkbox.value) {
                    aggregate.add(checkbox.value);
                }
            });
            directoryInput.value = Array.from(aggregate).join('\n');
            syncDisplayFromManualInput();
            closeDirectoryBrowser();
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

        if (directoryPickerButton) {
            // change: Always route folder selection through the API-powered overlay.
            directoryPickerButton.addEventListener('click', function(event) {
                event.preventDefault();
                openDirectoryBrowser(directoryInput ? directoryInput.value.split(/\r?\n|,/)[0] : archiveRoot);
            });
        }

        if (browserList) {
            // change: Enable navigation into subdirectories directly from the overlay list.
            browserList.addEventListener('click', function(event) {
                const target = event.target;
                if (target && target.dataset && target.dataset.browserEnter) {
                    event.preventDefault();
                    loadBrowserPath(target.dataset.browserEnter);
                }
            });
        }

        browserDismissButtons.forEach(function(node) {
            // change: Dismiss the overlay when the user clicks the backdrop or close button.
            node.addEventListener('click', function(event) {
                event.preventDefault();
                closeDirectoryBrowser();
            });
        });

        if (browserApplyButton) {
            // change: Commit the checked folders and close the browser overlay.
            browserApplyButton.addEventListener('click', function(event) {
                event.preventDefault();
                applyBrowserSelection();
            });
        }

        if (browserUpButton) {
            // change: Request the parent directory listing when available.
            browserUpButton.addEventListener('click', function(event) {
                event.preventDefault();
                const parentPath = browserUpButton.dataset.browserParent || '';
                const fallback = browserPathInput ? browserPathInput.value : archiveRoot;
                loadBrowserPath(parentPath || fallback);
            });
        }

        if (browserJumpButton && browserPathInput) {
            // change: Allow direct navigation to typed paths without leaving the overlay.
            browserJumpButton.addEventListener('click', function(event) {
                event.preventDefault();
                loadBrowserPath(browserPathInput.value);
            });
            browserPathInput.addEventListener('keydown', function(event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    loadBrowserPath(browserPathInput.value);
                }
            });
        }

        if (directoryInput) {
            // change: Rebuild the preview list as the user edits the directory field manually.
            directoryInput.addEventListener('input', syncDisplayFromManualInput);
        }

        updateProviderVisibility();
        loadOptions();
        syncDisplayFromManualInput(); // change: Ensure any prefilled value populates the folder preview on load.
    });
})();
