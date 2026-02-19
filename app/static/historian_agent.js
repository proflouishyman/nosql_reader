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

        // Method selector elements
        const methodSelect = document.getElementById('agentMethod');
        const methodHint = document.getElementById('agentMethodHint');
        const tier0ResearchLensContainer = document.getElementById('tier0ResearchLensContainer');
        const tier0ResearchLensInput = document.getElementById('tier0ResearchLens');
        
        // Debug console elements
        const debugConsole = document.getElementById('agentDebugConsole');
        const debugOutput = document.getElementById('agentDebugOutput');
        const debugClear = document.getElementById('agentDebugClear');
        
        // Metrics display
        const metricsContainer = document.getElementById('agentMetrics');
        const metricsGrid = document.getElementById('agentMetricsGrid');

        let conversationId = null;
        let isSubmitting = false;

        const METHOD_HINTS = {
            'basic': 'Fast hybrid retrieval with direct LLM generation (~15-30s)',
            'adversarial': 'Same as Good but with detailed pipeline monitoring (~15-30s)',
            'tiered': 'Best quality with confidence-based escalation (~20-60s)',
            'tier0': 'Corpus exploration and question generation (budget-driven, may take a few minutes)' // Added Tier 0 hint for corpus exploration.
        };

        function updateTier0ControlsVisibility() {
            // Added method-based toggle so research-lens input only appears for Tier 0 runs.
            if (!tier0ResearchLensContainer || !methodSelect) return;
            tier0ResearchLensContainer.hidden = methodSelect.value !== 'tier0';
        }

        function parseTier0ResearchLens(rawValue) {
            // Added lightweight parser so UI input maps to API-ready lens items.
            if (!rawValue) return [];
            return rawValue
                .split(',')
                .map(item => item.trim())
                .filter(Boolean);
        }

        if (methodSelect && methodHint) {
            methodSelect.addEventListener('change', function() {
                methodHint.textContent = METHOD_HINTS[methodSelect.value] || '';
                updateTier0ControlsVisibility();
            });
        }
        updateTier0ControlsVisibility();

        function debugLog(message, type = 'info') {
            if (!debugOutput) return;
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
            if (debugOutput) debugOutput.innerHTML = '';
        }

        if (debugClear) {
            debugClear.addEventListener('click', clearDebugLog);
        }

        /**
         * Unified Display Metrics Function (Crash Fix)
         * Standardized to use data.metrics.total_time and ignore undefined latency.
         */
        function displayMetrics(data) {
            if (!metricsContainer || !metricsGrid || !data.metrics) return;
            
            metricsGrid.innerHTML = '';
            metricsContainer.hidden = false;

            // 1. Standard "High-Level" Metrics (Safe Fallbacks)
            const totalTime = data.metrics.total_time || 0;
            const totalTokens = data.metrics.tokens || 0;
            const docCount = data.metrics.doc_count || 0;

            addMetric('Total Duration', `${totalTime.toFixed(1)}s`, 'primary');
            
            if (totalTokens > 0) {
                addMetric('Context Tokens', totalTokens.toLocaleString(), 'info');
            }
            
            const docLabel = data.method === 'tier0' ? 'Documents Read' : 'Sources Cited'; // Added Tier 0 label to match corpus exploration output.
            addMetric(docLabel, docCount, 'info');

            // 2. Method-Specific Detail Logic
            if (data.method === 'tiered' && data.metrics.stages) {
                const escalated = data.metrics.escalated ? 'Yes (Tier 2 Triggered)' : 'No (Tier 1 Sufficient)';
                addMetric('Escalation', escalated, data.metrics.escalated ? 'warning' : 'success');
                
                data.metrics.stages.forEach(stage => {
                    addMetric(`├─ ${stage.stage}`, `${(stage.total_time || 0).toFixed(1)}s`, 'info');
                });
            } else if (data.method === 'basic' || data.method === 'adversarial') {
                if (data.metrics.retrieval_time) {
                    addMetric('Retrieval Time', `${data.metrics.retrieval_time.toFixed(2)}s`, 'info');
                }
                if (data.metrics.llm_time) {
                    addMetric('LLM Time', `${data.metrics.llm_time.toFixed(2)}s`, 'info');
                }
            } else if (data.method === 'tier0') {
                // Added Tier 0 metrics so corpus exploration runs show batch and question counts.
                if (data.metrics.batches_processed) {
                    addMetric('Batches', data.metrics.batches_processed, 'info');
                }
                if (data.metrics.questions) {
                    addMetric('Questions', data.metrics.questions, 'info');
                }
                if (data.metrics.patterns) {
                    addMetric('Patterns', data.metrics.patterns, 'info');
                }
                if (data.metrics.entities) {
                    addMetric('Entities', data.metrics.entities, 'info');
                }
                if (data.metrics.contradictions) {
                    addMetric('Contradictions', data.metrics.contradictions, 'info');
                }
            }
        }

        function addMetric(label, value, type = 'info') {
            const metric = document.createElement('div');
            metric.className = `agent-metric agent-metric--${type}`;
            const labelEl = document.createElement('span');
            labelEl.className = 'agent-metric__label';
            labelEl.textContent = label;
            const valueEl = document.createElement('span');
            valueEl.className = 'agent-metric__value';
            valueEl.textContent = value;
            metric.appendChild(labelEl);
            metric.appendChild(valueEl);
            metricsGrid.appendChild(metric);
        }

        function setChatSubmitting(active, statusMsg) {
            isSubmitting = active;
            if (submitButton) submitButton.disabled = active;
            if (status) {
                status.hidden = !active;
                if (statusText && statusMsg) {
                    statusText.textContent = statusMsg;
                }
            }
        }

        function appendMessage(role, content, sources, searchId) {
            if (!historyContainer) return;
            const wrapper = document.createElement('div');
            wrapper.className = `agent-message agent-message--${role}`;
            const bubble = document.createElement('div');
            bubble.className = 'agent-message__bubble';
            
            if (role === 'assistant' && typeof marked !== 'undefined') {
                bubble.innerHTML = marked.parse(content);
            } else {
                bubble.textContent = content;
            }
            
            wrapper.appendChild(bubble);

            if (role === 'assistant' && sources && typeof sources === 'object' && Object.keys(sources).length > 0) {
                sourcesContainer.hidden = false;
                sourcesList.innerHTML = '';
                
                Object.entries(sources).forEach(([label, source]) => {
                    const item = document.createElement('li');
                    const link = document.createElement('a');
                    link.href = source.url;  // Use the url property directly
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = `${label}: ${source.display_name}`;  // Already clean!
                    item.appendChild(link);
                    sourcesList.appendChild(item);
                });
            }

            historyContainer.appendChild(wrapper);
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }

        function resetConversation() {
            if (historyContainer) historyContainer.innerHTML = '';
            if (sourcesContainer) sourcesContainer.hidden = true;
            if (sourcesList) sourcesList.innerHTML = '';
            if (metricsContainer) metricsContainer.hidden = true;
            clearDebugLog();
        }

        function renderHistory(history) {
            if (!historyContainer) return;
            historyContainer.innerHTML = '';
            history.forEach(item => {
                // Pass sources stored in history so links persist on reload
                appendMessage(item.role, item.content, item.sources || {}, null);
            });
        }

        async function submitQuestion(payload) {
            if (!form || !agentPanel) return;

            const method = methodSelect ? methodSelect.value : 'tiered';
            const endpoint = `/historian-agent/query-${method}`;
            if (method === 'tier0' && tier0ResearchLensInput) {
                const parsedLens = parseTier0ResearchLens(tier0ResearchLensInput.value);
                if (parsedLens.length) {
                    payload.research_lens = parsedLens; // Added Tier 0 payload wiring so corpus exploration can prioritize historian-selected intersections.
                }
            }
            
            debugLog(`=== Starting ${method.toUpperCase()} query ===`, 'primary');
            debugLog(`Endpoint: ${endpoint}`, 'info');
            debugLog(`Question: ${payload.question.substring(0, 100)}...`, 'info');
            if (method === 'tier0' && payload.research_lens) {
                debugLog(`Research lens: ${payload.research_lens.join(', ')}`, 'info'); // Added visibility for lens values in the debug console.
            }
            
            setChatSubmitting(true, `Processing with ${method} method...`);
            
            try {
                const startTime = performance.now();
                debugLog('Sending request to backend...', 'info');
                
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                const elapsedTime = ((performance.now() - startTime) / 1000).toFixed(1);
                
                if (!response.ok) {
                    throw new Error(data.error || 'Query failed');
                }
                
                debugLog(`✓ Response received in ${elapsedTime}s`, 'success');
                debugLog(`Method used: ${data.method || method}`, 'info');
                
                // Logging logic (kept for console clarity)
                if (method === 'tiered') {
                    debugLog(`Escalated to Tier 2: ${data.metrics.escalated ? 'YES' : 'NO'}`, data.metrics.escalated ? 'warning' : 'success');
                } else if (method === 'basic' || method === 'adversarial') {
                    debugLog(`Total Time: ${data.metrics.total_time.toFixed(1)}s`, 'info');
                } else if (method === 'tier0') {
                    debugLog(`Documents read: ${data.metrics.doc_count || 0}`, 'info'); // Added Tier 0 logging for exploration runs.
                    if (data.metrics.questions) {
                        debugLog(`Questions generated: ${data.metrics.questions}`, 'info');
                    }
                }
                
                debugLog(`Sources found: ${Object.keys(data.sources || {}).length}`, 'info');
                
                conversationId = data.conversation_id || payload.conversation_id;
                renderHistory(data.history || []);
                
                //if (data.answer) {
                //    appendMessage('assistant', data.answer, data.sources || {}, data.search_id);
                //}
                
                // Call displayMetrics without the second arg now that logic is unified
                displayMetrics(data);
                
                if (questionInput) {
                    questionInput.value = '';
                    questionInput.focus();
                }
                debugLog('=== Query complete ===', 'success');
                
            } catch (error) {
                console.error(error);
                debugLog(`✗ Error: ${error.message}`, 'error');
                appendMessage('assistant', error.message || 'Unable to process your question right now.', []);
            } finally {
                setChatSubmitting(false);
            }
        }

        if (form) {
            form.addEventListener('submit', function(event) {
                event.preventDefault();
                if (isSubmitting) return;
                const question = questionInput ? questionInput.value.trim() : '';
                if (!question) {
                    if (questionInput) questionInput.focus();
                    return;
                }
                appendMessage('user', question, []);
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
                if (questionInput) questionInput.focus();
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

        debugLog('Historian Agent initialized', 'success');
        debugLog(`Current method: ${methodSelect ? methodSelect.value : 'tiered'}`, 'info');
    });
})();
