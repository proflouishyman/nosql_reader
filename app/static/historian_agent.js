(function() {
    'use strict';

    document.addEventListener('DOMContentLoaded', function() {
        const configDataEl = document.getElementById('historianAgentConfigData');
        let agentConfig = null;
        if (configDataEl) {
            try {
                agentConfig = JSON.parse(configDataEl.textContent);
            } catch (error) {
                console.warn('Failed to parse Historian Agent config payload', error);
            }
        }

        const agentPanel = document.querySelector('.agent-panel');
        const form = document.getElementById('agentForm');
        const questionInput = document.getElementById('agentQuestion');
        const historyContainer = document.getElementById('agentHistory');
        const sourcesContainer = document.getElementById('agentSources');
        const sourcesList = document.getElementById('agentSourcesList');
        const resetButton = document.getElementById('agentReset');
        const status = document.getElementById('agentStatus');
        const statusText = document.getElementById('agentStatusText');
        const suggestionButtons = Array.from(document.querySelectorAll('.agent-suggestion'));
        const submitButton = form ? form.querySelector('button[type="submit"]') : null;

        // Pre-fill question when arriving from a Corpus Explorer "Investigate ->" link.
        const ceParams = new URLSearchParams(window.location.search);
        const cePrefillQuestion = ceParams.get('q');
        if (cePrefillQuestion && questionInput) {
            questionInput.value = decodeURIComponent(cePrefillQuestion);
            const note = document.createElement('p');
            note.className = 'ce-prefill-note';
            note.textContent = '<- Pre-filled from Corpus Explorer';
            questionInput.parentNode.insertBefore(note, questionInput);
        }

        const methodSelect = document.getElementById('agentMethod');
        const methodHint = document.getElementById('agentMethodHint');
        const debugOutput = document.getElementById('agentDebugOutput');
        const debugClear = document.getElementById('agentDebugClear');

        let conversationId = null;
        let isSubmitting = false;

        // Updated hint copy to match the new Quick/Verified/Deep labels without changing method values.
        const METHOD_HINTS = {
            basic: 'Direct retrieval - fast, no cross-check. Good for orientation questions.',
            adversarial: 'Answer is verified by a second LLM call that challenges the first.',
            tiered: 'Retrieves evidence, escalates to deeper analysis if confidence is low.'
        };

        if (methodSelect && methodHint) {
            methodHint.textContent = METHOD_HINTS[methodSelect.value] || '';
            methodSelect.addEventListener('change', function() {
                methodHint.textContent = METHOD_HINTS[methodSelect.value] || '';
            });
        }

        // Added activity stream controller so logs auto-expand while running and quiet down when idle.
        const ActivityStream = (() => {
            const indicator = document.getElementById('activityIndicator');
            const label = document.getElementById('activityLabel');
            const toggle = document.getElementById('activityToggle');
            const log = document.getElementById('agentDebugOutput');
            const clearBtn = document.getElementById('agentDebugClear');
            let running = false;
            let lastDuration = null;

            if (toggle && log) {
                toggle.addEventListener('click', () => {
                    const shouldOpen = log.hidden;
                    log.hidden = !shouldOpen;
                    toggle.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
                    toggle.textContent = shouldOpen ? 'Hide log' : 'Show log';
                });
            }

            if (clearBtn) {
                clearBtn.addEventListener('click', () => {
                    if (log) {
                        log.innerHTML = '';
                    }
                });
            }

            function setRunning(isRunning) {
                running = isRunning;
                if (!indicator || !label) {
                    return;
                }

                if (isRunning) {
                    indicator.className = 'agent-activity__indicator is-running';
                    label.textContent = 'Running...';
                    if (log) {
                        log.hidden = false;
                    }
                    if (toggle) {
                        toggle.setAttribute('aria-expanded', 'true');
                        toggle.textContent = 'Hide log';
                    }
                    if (clearBtn) {
                        clearBtn.hidden = false;
                    }
                    return;
                }

                indicator.className = 'agent-activity__indicator is-idle';
                const durationText = lastDuration ? ` - completed in ${lastDuration}` : '';
                label.textContent = `Done${durationText}`;

                setTimeout(() => {
                    if (running) {
                        return;
                    }
                    if (log) {
                        log.hidden = true;
                    }
                    if (clearBtn) {
                        clearBtn.hidden = true;
                    }
                    if (toggle) {
                        toggle.setAttribute('aria-expanded', 'false');
                        toggle.textContent = 'Show log';
                    }
                    label.textContent = `Ready${durationText}`;
                }, 4000);
            }

            function setDuration(seconds) {
                if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds >= 0) {
                    lastDuration = `${seconds.toFixed(1)}s`;
                }
            }

            return { setRunning, setDuration };
        })();

        function debugLog(message, type = 'info') {
            if (!debugOutput) {
                return;
            }
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = `agent-debug-entry agent-debug-entry--${type}`;

            const timeEl = document.createElement('span');
            timeEl.className = 'agent-debug-time';
            timeEl.textContent = `[${timestamp}]`;

            const msgEl = document.createElement('span');
            msgEl.className = 'agent-debug-message';
            msgEl.textContent = message;

            entry.appendChild(timeEl);
            entry.appendChild(msgEl);
            debugOutput.appendChild(entry);
            debugOutput.scrollTop = debugOutput.scrollHeight;
        }

        function clearDebugLog() {
            if (debugOutput) {
                debugOutput.innerHTML = '';
            }
        }

        function openLogStream(onDone) {
            const source = new EventSource('/api/log-stream');
            let closed = false;

            source.onmessage = function(event) {
                try {
                    const payload = JSON.parse(event.data);
                    const displayMsg = payload.source
                        ? `${payload.source} ${payload.message}`
                        : payload.message;
                    debugLog(displayMsg, payload.level || 'info');
                } catch (error) {
                    debugLog(event.data, 'info');
                }
            };

            source.addEventListener('done', function() {
                if (!closed) {
                    closed = true;
                    source.close();
                    if (onDone) {
                        onDone();
                    }
                }
            });

            source.onerror = function() {
                if (!closed) {
                    closed = true;
                    source.close();
                }
            };

            return function close() {
                if (!closed) {
                    closed = true;
                    source.close();
                }
            };
        }

        // Normalizes current backend metrics (total_time/doc_count/tiers) into UI pill data without changing API contracts.
        function normalizeInlineMetrics(rawMetrics, method) {
            if (!rawMetrics || typeof rawMetrics !== 'object') {
                return null;
            }

            const safeNumber = (value) => {
                const parsed = Number(value);
                return Number.isFinite(parsed) ? parsed : null;
            };

            const totalTimeSeconds = safeNumber(rawMetrics.total_time);
            const durationMs = safeNumber(rawMetrics.duration_ms) || (totalTimeSeconds !== null ? totalTimeSeconds * 1000 : null);
            const docsRetrieved = safeNumber(rawMetrics.docs_retrieved) || safeNumber(rawMetrics.doc_count);

            let confidence = safeNumber(rawMetrics.confidence);
            let tier = rawMetrics.tier || null;

            if (method === 'tiered' && Array.isArray(rawMetrics.tiers)) {
                // Derive tier and confidence from verification stage objects returned by current tiered pipeline.
                let derivedTier = 1;
                let latestScore = null;

                rawMetrics.tiers.forEach((entry) => {
                    if (!entry || typeof entry !== 'object') {
                        return;
                    }
                    const tierLabel = String(entry.tier || '');
                    if (tierLabel.startsWith('2')) {
                        derivedTier = 2;
                    }
                    const score = safeNumber(entry.score);
                    if (score !== null) {
                        latestScore = score;
                    }
                });

                tier = derivedTier;
                if (confidence === null && latestScore !== null) {
                    confidence = latestScore > 1 ? latestScore / 100 : latestScore;
                }
            }

            if (confidence !== null) {
                confidence = Math.max(0, Math.min(1, confidence));
            }

            return {
                confidence,
                tier,
                docs_retrieved: docsRetrieved,
                duration_ms: durationMs
            };
        }

        // Added compact per-answer pills to replace the old bottom metrics panel.
        function renderInlineMetrics(bubble, metrics) {
            if (!bubble || !metrics) {
                return;
            }

            const bar = document.createElement('div');
            bar.className = 'agent-metrics-bar';

            const conf = metrics.confidence ?? null;
            const tier = metrics.tier ?? null;
            const docs = metrics.docs_retrieved ?? null;
            const ms = metrics.duration_ms ?? null;

            let confClass = 'metric-pill--neutral';
            if (conf !== null) {
                confClass = conf >= 0.7 ? 'metric-pill--good'
                    : conf >= 0.4 ? 'metric-pill--warn'
                    : 'metric-pill--low';
            }

            const items = [
                conf !== null ? `<span class="metric-pill ${confClass}" title="Verification confidence score"><i class="fa-solid fa-shield-halved"></i> ${Math.round(conf * 100)}% confidence</span>` : '',
                tier !== null ? `<span class="metric-pill metric-pill--neutral" title="Query tier used"><i class="fa-solid fa-layer-group"></i> Tier ${tier}</span>` : '',
                docs !== null ? `<span class="metric-pill metric-pill--neutral" title="Source documents retrieved"><i class="fa-solid fa-file-lines"></i> ${docs} docs</span>` : '',
                ms !== null ? `<span class="metric-pill metric-pill--neutral" title="Wall-clock response time"><i class="fa-regular fa-clock"></i> ${(ms / 1000).toFixed(1)}s</span>` : ''
            ].filter(Boolean);

            if (!items.length) {
                return;
            }

            bar.innerHTML = items.join('');
            bubble.appendChild(bar);
        }

        function setChatSubmitting(active, statusMsg) {
            isSubmitting = active;
            if (submitButton) {
                submitButton.disabled = active;
            }
            if (status) {
                status.hidden = !active;
                if (statusText && statusMsg) {
                    statusText.textContent = statusMsg;
                }
            }
        }

        function appendMessage(role, content, sources, inlineMetrics) {
            if (!historyContainer) {
                return;
            }

            const wrapper = document.createElement('div');
            wrapper.className = `agent-message agent-message--${role}`;

            const bubble = document.createElement('div');
            bubble.className = 'agent-message__bubble';

            if (role === 'assistant' && typeof marked !== 'undefined') {
                bubble.innerHTML = marked.parse(content);
            } else {
                bubble.textContent = content;
            }

            if (role === 'assistant') {
                renderInlineMetrics(bubble, inlineMetrics);
            }

            wrapper.appendChild(bubble);

            if (role === 'assistant' && sources && typeof sources === 'object' && Object.keys(sources).length > 0) {
                if (sourcesContainer) {
                    sourcesContainer.hidden = false;
                }
                if (sourcesList) {
                    sourcesList.innerHTML = '';
                    Object.entries(sources).forEach(([label, source]) => {
                        const item = document.createElement('li');
                        const link = document.createElement('a');
                        link.href = source.url;
                        link.target = '_blank';
                        link.rel = 'noopener noreferrer';
                        link.setAttribute('data-help', 'Open the cited source document in a new tab.');
                        link.textContent = `${label}: ${source.display_name}`;
                        item.appendChild(link);
                        sourcesList.appendChild(item);
                    });
                }
            }

            historyContainer.appendChild(wrapper);
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }

        function resetConversation() {
            if (historyContainer) {
                historyContainer.innerHTML = '';
            }
            if (sourcesContainer) {
                sourcesContainer.hidden = true;
            }
            if (sourcesList) {
                sourcesList.innerHTML = '';
            }
            clearDebugLog();
        }

        function renderHistory(history, latestMetrics, method) {
            if (!historyContainer) {
                return;
            }

            historyContainer.innerHTML = '';
            if (sourcesContainer) {
                sourcesContainer.hidden = true;
            }
            if (sourcesList) {
                sourcesList.innerHTML = '';
            }

            let lastAssistantIndex = -1;
            history.forEach((item, index) => {
                if (item && item.role === 'assistant') {
                    lastAssistantIndex = index;
                }
            });

            history.forEach((item, index) => {
                const useInlineMetrics = item.role === 'assistant' && index === lastAssistantIndex
                    ? normalizeInlineMetrics(latestMetrics, method)
                    : null;
                appendMessage(item.role, item.content, item.sources || {}, useInlineMetrics);
            });
        }

        function hasTierTwoStages(metrics) {
            return !!(metrics && Array.isArray(metrics.tiers) && metrics.tiers.some((entry) => {
                if (!entry || typeof entry !== 'object') {
                    return false;
                }
                return String(entry.tier || '').startsWith('2');
            }));
        }

        async function submitQuestion(payload) {
            if (!form || !agentPanel) {
                return;
            }

            const method = methodSelect ? methodSelect.value : 'tiered';
            const endpoint = `/historian-agent/query-${method}`;

            clearDebugLog();
            debugLog(`=== Starting ${method.toUpperCase()} query ===`, 'primary');
            debugLog(`Question: ${payload.question.substring(0, 100)}...`, 'info');

            ActivityStream.setRunning(true);
            setChatSubmitting(true, `Processing with ${method} method...`);
            const closeStream = openLogStream(null);
            const streamTimeout = setTimeout(() => closeStream(), 300000);
            const startTime = performance.now();

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                const elapsedSeconds = (performance.now() - startTime) / 1000;

                if (!response.ok) {
                    throw new Error(data.error || 'Query failed');
                }

                const serverDuration = data && data.metrics ? Number(data.metrics.total_time) : NaN;
                const durationSeconds = Number.isFinite(serverDuration) ? serverDuration : elapsedSeconds;

                debugLog(`Response received in ${durationSeconds.toFixed(1)}s`, 'success');
                debugLog(`Method used: ${data.method || method}`, 'info');

                if ((data.method || method) === 'tiered') {
                    const escalated = hasTierTwoStages(data.metrics);
                    debugLog(`Escalated to Tier 2: ${escalated ? 'YES' : 'NO'}`, escalated ? 'warning' : 'success');
                }

                if (data.metrics && Number.isFinite(Number(data.metrics.doc_count))) {
                    debugLog(`Sources found: ${Number(data.metrics.doc_count)}`, 'info');
                } else {
                    debugLog(`Sources found: ${Object.keys(data.sources || {}).length}`, 'info');
                }

                conversationId = data.conversation_id || payload.conversation_id;
                renderHistory(data.history || [], data.metrics || null, data.method || method);
                ActivityStream.setDuration(durationSeconds);

                if (questionInput) {
                    questionInput.value = '';
                    questionInput.focus();
                }

                debugLog('=== Query complete ===', 'success');
            } catch (error) {
                console.error(error);
                debugLog(`Error: ${error.message}`, 'error');
                appendMessage('assistant', error.message || 'Unable to process your question right now.', {}, null);
            } finally {
                clearTimeout(streamTimeout);
                closeStream();
                setChatSubmitting(false);
                ActivityStream.setRunning(false);
            }
        }

        if (form) {
            form.addEventListener('submit', function(event) {
                event.preventDefault();
                if (isSubmitting) {
                    return;
                }
                const question = questionInput ? questionInput.value.trim() : '';
                if (!question) {
                    if (questionInput) {
                        questionInput.focus();
                    }
                    return;
                }
                appendMessage('user', question, {}, null);
                submitQuestion({
                    question,
                    conversation_id: conversationId,
                });
            });
        }

        if (resetButton) {
            resetButton.addEventListener('click', function() {
                resetConversation();
                conversationId = null;
                if (questionInput) {
                    questionInput.focus();
                }
            });
        }

        if (suggestionButtons.length && form && questionInput) {
            suggestionButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const question = button.dataset.question || '';
                    const autoSend = button.dataset.autosend === 'true';
                    questionInput.value = question;
                    questionInput.focus();
                    if (autoSend) {
                        form.dispatchEvent(new Event('submit'));
                    }
                });
            });
        }

        // Keep startup log entries so operators can immediately confirm the selected strategy.
        debugLog('Historian Agent initialized', 'success');
        debugLog(`Current method: ${methodSelect ? methodSelect.value : 'tiered'}`, 'info');

        // Read parsed config to satisfy tooling/lint flows that enforce usage of initialization values.
        if (agentConfig && agentConfig.enabled === false) {
            debugLog('Agent is disabled by configuration.', 'warning');
        }
    });
})();
