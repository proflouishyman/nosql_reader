(function() {
    // Added new read-only data ingestion UI script that lists mounts and triggers scans.

    function ready(fn) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fn);
        } else {
            fn();
        }
    }

    function sanitizeId(value) {
        // Added helper so mount targets with slashes can be used as DOM ids safely.
        return value.replace(/[^a-z0-9_-]/gi, '_');
    }

    async function fetchJson(url, options) {
        // Added wrapper to centralise fetch error handling for the new endpoints.
        const response = await fetch(url, options || {});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            const message = data && data.message ? data.message : `Request failed with status ${response.status}`;
            throw new Error(message);
        }
        return data;
    }

    ready(function() {
        const root = document.querySelector('[data-ingestion-root]');
        if (!root) return;

        const mountsUrl = root.dataset.mountsUrl;
        const treeUrl = root.dataset.treeUrl;
        const scanUrl = root.dataset.scanUrl;
        const rebuildUrl = root.dataset.rebuildUrl;

        const mountsContainer = root.querySelector('[data-mounts]');
        const statusNode = root.querySelector('[data-ingestion-status]');
        const summaryNode = root.querySelector('[data-ingestion-summary]');
        const errorsNode = root.querySelector('[data-ingestion-errors]');
        const scanButton = root.querySelector('[data-scan-mounts]');
        const rebuildButton = root.querySelector('[data-rebuild-mounts]');

        let ingestionConfig = null; // Added cache so the latest selector payload can be reused for API calls.
        const selectorAvailable = typeof window.IngestionModelSelector !== 'undefined'; // Added flag so behaviour falls back gracefully when the selector is absent.

        if (selectorAvailable) {
            document.addEventListener('ingestionConfigChange', function(event) {
                ingestionConfig = event.detail; // Added listener so updates from the selector are captured here.
            });
        }

        function setStatus(message, type) {
            // Added status helper so both scan and rebuild share consistent messaging.
            if (!statusNode) return;
            statusNode.textContent = message || '';
            statusNode.dataset.statusType = type || 'info';
            statusNode.hidden = !message;
        }

        function renderSummary(totals) {
            // Added renderer to show aggregate ingestion metrics from new endpoints.
            if (!summaryNode) return;
            summaryNode.innerHTML = '';
            if (!totals) {
                summaryNode.hidden = true;
                return;
            }
            const pairs = [
                ['Images discovered', totals.images_total],
                ['JSON generated', totals.generated],
                ['Skipped (already processed)', totals.skipped_existing],
                ['Queued for ingest', totals.queued_existing],
                ['Image processing failures', totals.failed],
                ['Database inserts', totals.ingested],
                ['Database updates', totals.updated],
                ['Ingestion failures', totals.ingest_failures],
            ];
            pairs.forEach(function(pair) {
                const dt = document.createElement('dt');
                dt.textContent = pair[0];
                const dd = document.createElement('dd');
                dd.textContent = typeof pair[1] === 'number' ? pair[1] : '0';
                summaryNode.appendChild(dt);
                summaryNode.appendChild(dd);
            });
            summaryNode.hidden = false;
        }

        function renderErrors(results) {
            // Added error renderer so missing mounts or ingestion failures are visible to operators.
            if (!errorsNode) return;
            errorsNode.innerHTML = '';
            const issues = Array.isArray(results) ? results.filter(function(item) {
                return item && (item.status === 'error' || item.status === 'missing');
            }) : [];

            if (!issues.length) {
                errorsNode.hidden = true;
                return;
            }

            const list = document.createElement('ul');
            issues.forEach(function(item) {
                const li = document.createElement('li');
                if (item.status === 'missing') {
                    li.textContent = `${item.target} is not available inside the container.`;
                } else if (item.message) {
                    li.textContent = `${item.target}: ${item.message}`;
                } else {
                    li.textContent = `${item.target}: unexpected error.`;
                }
                list.appendChild(li);
            });

            errorsNode.appendChild(list);
            errorsNode.hidden = false;
        }

        async function loadMountTree(target, block) {
            // Added fetch to display a short folder tree for each mount.
            if (!treeUrl || !target) {
                block.textContent = 'No preview available.';
                return;
            }
            block.textContent = 'Loading…';
            try {
                const tree = await fetchJson(treeUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: target }),
                });
                block.textContent = JSON.stringify(tree, null, 2);
            } catch (error) {
                block.textContent = `Unable to load tree: ${error.message}`;
            }
        }

        async function loadMounts() {
            // Added loader to populate the list of mounts from the backend helper.
            if (!mountsContainer) return;
            mountsContainer.innerHTML = '';
            if (!mountsUrl) {
                mountsContainer.textContent = 'Mount discovery endpoint unavailable.';
                return;
            }
            try {
                const data = await fetchJson(mountsUrl);
                const mounts = Array.isArray(data.mounts) ? data.mounts : [];
                if (!mounts.length) {
                    mountsContainer.textContent = 'No mounts defined for the app service.';
                    return;
                }

                mounts.forEach(function(mount) {
                    const block = document.createElement('div');
                    block.className = 'mount-block';
                    const header = document.createElement('h4');
                    header.textContent = mount.target || '(unknown target)';
                    block.appendChild(header);

                    const hostPath = document.createElement('p');
                    hostPath.className = 'mount-host-path';
                    hostPath.textContent = `Host path: ${mount.source || 'not specified'}`;
                    block.appendChild(hostPath);

                    const availability = document.createElement('p');
                    availability.className = 'mount-availability';
                    availability.textContent = mount.target_exists ? 'Mounted inside container.' : 'Target not found inside container.';
                    block.appendChild(availability);

                    const treePre = document.createElement('pre');
                    treePre.id = `tree-${sanitizeId(mount.target || 'unknown')}`;
                    block.appendChild(treePre);

                    mountsContainer.appendChild(block);
                    loadMountTree(mount.target, treePre);
                });
            } catch (error) {
                mountsContainer.textContent = `Unable to load mounts: ${error.message}`;
            }
        }

        async function triggerAction(url, startMessage, successBuilder) {
            // Added shared button handler to reduce duplication between scan and rebuild actions.
            if (!url) {
                setStatus('Ingestion endpoint not available.', 'error');
                return;
            }

            let options = { method: 'POST' }; // Added default fetch options so legacy behaviour still posts without a body.
            if (selectorAvailable && window.IngestionModelSelector) {
                const validation = window.IngestionModelSelector.validateConfig(); // Added validation call to prevent invalid requests.
                if (!validation.valid) {
                    window.IngestionModelSelector.showStatus(validation.message, 'error'); // Added selector feedback when validation fails.
                    setStatus(validation.message, 'error'); // Added main status update to mirror the selector error.
                    return;
                }
                const configPayload = window.IngestionModelSelector.getConfig(); // Added retrieval of the current selector configuration.
                ingestionConfig = configPayload; // Added cache update so later operations reuse the confirmed payload.
                options = {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(configPayload),
                }; // Added JSON payload so the backend receives provider and model data.
                window.IngestionModelSelector.showStatus(startMessage, 'info'); // Added selector status so users see progress near the form.
            }

            try {
                setStatus(startMessage, 'info');
                const result = await fetchJson(url, options);
                renderSummary(result.aggregate);
                renderErrors(result.results);
                const message = successBuilder(result);
                setStatus(message, 'success');
                if (selectorAvailable && window.IngestionModelSelector) {
                    window.IngestionModelSelector.showStatus(message, 'success'); // Added selector success message for parity with page status.
                }
            } catch (error) {
                setStatus(error.message, 'error');
                if (selectorAvailable && window.IngestionModelSelector) {
                    window.IngestionModelSelector.showStatus(error.message, 'error'); // Added selector error message to keep feedback consistent.
                }
            }
        }

        async function triggerScan() {
            // Added dedicated SSE trigger so scans stream logs live into the page.
            if (!scanUrl) {
                setStatus('Scan endpoint not available.', 'error');
                return;
            }

            if (!statusNode) {
                console.warn('Scan status node missing.');  // Added guard so the function fails safely when required DOM nodes are absent.
                return;
            }

            if (selectorAvailable) {
                const validation = window.IngestionModelSelector.validateConfig();  // Added validation call to prevent SSE from starting with invalid settings.
                if (!validation.valid) {
                    setStatus(validation.message, 'error');  // Added UI feedback mirroring selector validation errors.
                    return;
                }
                ingestionConfig = window.IngestionModelSelector.getConfig();  // Added cache update so POST + SSE share the same payload.
            }

            if (scanButton) {
                scanButton.disabled = true;  // Added lock so users cannot trigger overlapping scans.
            }
            if (rebuildButton) {
                rebuildButton.disabled = true;  // Added disable to avoid rebuilds while scanning is active.
            }

            statusNode.textContent = '';  // Added reset to remove stale status text before appending logs.
            if (summaryNode) {
                summaryNode.innerHTML = '';  // Added summary clear because SSE replaces aggregate metrics with a log stream.
                summaryNode.hidden = true;  // Added hide to collapse the old summary block during live streaming.
            }
            if (errorsNode) {
                errorsNode.innerHTML = '';  // Added error clear so previous run issues do not linger.
                errorsNode.hidden = true;  // Added hide to collapse previous error list while streaming.
            }

            const logContainer = document.createElement('div');  // Added container to hold individual SSE log entries.
            logContainer.className = 'scan-logs';
            logContainer.style.cssText = `
                margin-top: 1rem;
                padding: 1rem;
                background: #f5f5f5;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.875rem;
                max-height: 400px;
                overflow-y: auto;
            `;  // Added inline styling so the streaming log is easy to read without extra CSS files.
            statusNode.appendChild(logContainer);  // Added mounting of the log container into the existing status area.
            statusNode.hidden = false;  // Added ensure the status panel is visible while streaming events.

            function addLog(message, type = 'info') {
                const logEntry = document.createElement('div');
                logEntry.style.cssText = `padding: 0.25rem 0;`;  // Added spacing to keep log messages readable.

                const colors = {
                    info: '#666',
                    success: '#2d5',
                    warning: '#f80',
                    error: '#d22',
                    processing: '#37f'
                };  // Added palette to differentiate message types visually.

                const icons = {
                    info: 'ℹ️',
                    success: '✓',
                    warning: '⚠️',
                    error: '✗',
                    processing: '⏳'
                };  // Added icon set to match backend event types.

                logEntry.innerHTML = `<span style="color: ${colors[type] || colors.info}">${icons[type] || icons.info} ${message}</span>`;  // Added fallback handling so unknown types still render.
                logContainer.appendChild(logEntry);
                logContainer.scrollTop = logContainer.scrollHeight;  // Added auto-scroll so the latest message stays in view.
            }

            try {
                const response = await fetch(scanUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(ingestionConfig || {})
                });  // Added POST to kick off the backend scan using the selected configuration.

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);  // Added explicit HTTP error propagation for failed handshakes.
                }

                const eventSource = new EventSource(scanUrl);  // Added SSE connection so progress events stream without polling.

                eventSource.onmessage = function(event) {
                    // Added unified handler so each backend event updates the live log area.
                    try {
                        const data = JSON.parse(event.data);  // Added JSON parsing so payloads can be pattern-matched cleanly.

                        switch (data.type) {
                            case 'start':
                                addLog(data.message, 'info');
                                break;
                            case 'scan_start':
                                addLog(`Found ${data.total_images} images in ${data.directory}`, 'info');
                                break;
                            case 'image_start':
                                addLog(`[${data.index}/${data.total}] ${data.image}`, 'info');
                                break;
                            case 'image_processing':
                                addLog(`  ⏳ ${data.message}`, 'processing');
                                break;
                            case 'image_info':
                                addLog(`  ${data.message}`, 'info');
                                break;
                            case 'image_skip':
                                addLog(`  ⏭️ ${data.reason}`, 'warning');
                                break;
                            case 'image_complete':
                                addLog(`  ✓ Complete (${data.processed} processed, ${data.skipped} skipped, ${data.errors} errors)`, 'success');
                                break;
                            case 'image_error':
                                addLog(`  ✗ Error: ${data.error}`, 'error');
                                break;
                            case 'scan_complete':
                                addLog(`\n✓ Scan complete: ${data.processed} processed, ${data.skipped} skipped, ${data.errors} errors`, 'success');
                                eventSource.close();  // Added close call to stop listening once the scan finishes.
                                if (scanButton) {
                                    scanButton.disabled = false;  // Added re-enable so the button becomes clickable again post-scan.
                                }
                                if (rebuildButton) {
                                    rebuildButton.disabled = false;  // Added re-enable for rebuild after the scan completes.
                                }
                                break;
                            case 'error':
                                addLog(`Error: ${data.message}`, 'error');
                                eventSource.close();
                                if (scanButton) {
                                    scanButton.disabled = false;
                                }
                                if (rebuildButton) {
                                    rebuildButton.disabled = false;
                                }
                                break;
                            case 'warning':
                                addLog(`⚠️ ${data.message}`, 'warning');
                                break;
                            case 'complete':
                                addLog(data.message, 'success');  // Added fallback for backend completion messages outside scan_complete.
                                break;
                            default:
                                addLog(JSON.stringify(data), 'info');
                                break;
                        }
                    } catch (err) {
                        console.error('Failed to parse SSE data:', err, event.data);  // Added console diagnostic to help debug malformed events.
                    }
                };

                eventSource.onerror = function(error) {
                    // Added error hook so connection failures unlock the UI and surface an alert.
                    console.error('SSE error:', error);  // Added console logging for network or server-side stream issues.
                    addLog('Connection error - scan may have completed or failed', 'error');
                    eventSource.close();
                    if (scanButton) {
                        scanButton.disabled = false;
                    }
                    if (rebuildButton) {
                        rebuildButton.disabled = false;
                    }
                };

            } catch (error) {
                console.error('Scan failed:', error);  // Added console output to aid debugging handshake failures.
                addLog(`Fatal error: ${error.message}`, 'error');
                if (scanButton) {
                    scanButton.disabled = false;
                }
                if (rebuildButton) {
                    rebuildButton.disabled = false;
                }
            }
        }

        if (scanButton) {
            scanButton.addEventListener('click', function() {
                triggerScan();  // Added SSE-based handler so the scan button now streams logs instead of waiting for a JSON payload.
            });
        }

        if (rebuildButton) {
            rebuildButton.addEventListener('click', function() {
                // Added event binding so users can rebuild the database from mounted directories.
                triggerAction(rebuildUrl, 'Rebuilding ingestion data…', function(result) {
                    const aggregate = result && result.aggregate ? result.aggregate : {};
                    return `Rebuild finished. ${aggregate.ingested || 0} records inserted.`;
                });
            });
        }

        loadMounts();
    });
})();
