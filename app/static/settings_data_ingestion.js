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

            try {
                setStatus(startMessage, 'info');
                const result = await fetchJson(url, { method: 'POST' });
                renderSummary(result.aggregate);
                renderErrors(result.results);
                const message = successBuilder(result);
                setStatus(message, 'success');
            } catch (error) {
                setStatus(error.message, 'error');
            }
        }

        if (scanButton) {
            scanButton.addEventListener('click', function() {
                // Added event binding so users can scan mounts for new images.
                triggerAction(scanUrl, 'Scanning mounts for new images…', function(result) {
                    const aggregate = result && result.aggregate ? result.aggregate : {};
                    return `Scan complete. ${aggregate.generated || 0} JSON files generated.`;
                });
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
