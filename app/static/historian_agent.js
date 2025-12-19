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
        const disabledBanner = document.getElementById('agentDisabledBanner');
        const errorBanner = document.getElementById('agentErrorBanner');

        // NEW: Method selector elements
        const methodSelect = document.getElementById('agentMethod');
        const methodHint = document.getElementById('agentMethodHint');
        
        // NEW: Debug console elements
        const debugConsole = document.getElementById('agentDebugConsole');
        const debugOutput = document.getElementById('agentDebugOutput');
        const debugClear = document.getElementById('agentDebugClear');
        
        // NEW: Metrics display
        const metricsContainer = document.getElementById('agentMetrics');
        const metricsGrid = document.getElementById('agentMetricsGrid');

        let conversationId = null;
        let isSubmitting = false;
        let agentEnabled = agentConfig ? Boolean(agentConfig.enabled) : true;
        let hasAgentError = agentPanel ? agentPanel.dataset.agentError === 'true' : false;

        // NEW: Method descriptions
        const METHOD_HINTS = {
            'basic': 'Fast hybrid retrieval with direct LLM generation (~15-30s)',
            'adversarial': 'Same as Good but with detailed pipeline monitoring (~15-30s)',
            'tiered': 'Best quality with confidence-based escalation (~20-60s)'
        };

        // NEW: Update hint when method changes
        if (methodSelect && methodHint) {
            methodSelect.addEventListener('change', function() {
                methodHint.textContent = METHOD_HINTS[methodSelect.value] || '';
            });
        }

        // NEW: Debug console functions
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
            if (debugOutput) {
                debugOutput.innerHTML = '';
            }
        }

        if (debugClear) {
            debugClear.addEventListener('click', clearDebugLog);
        }

        // NEW: Display performance metrics
        function displayMetrics(data, method) {
            if (!metricsContainer || !metricsGrid) return;
            
            metricsGrid.innerHTML = '';
            metricsContainer.hidden = false;

            if (method === 'tiered' && data.metrics && Array.isArray(data.metrics)) {
                // Tiered metrics - show stage breakdown
                const escalated = data.escalated ? 'Yes (Tier 2 triggered)' : 'No (Tier 1 sufficient)';
                addMetric('Escalated', escalated, data.escalated ? 'warning' : 'success');
                addMetric('Total Duration', `${data.total_duration.toFixed(1)}s`, 'primary');
                
                // Stage breakdown
                data.metrics.forEach(stage => {
                    addMetric(
                        stage.stage,
                        `${stage.total_time.toFixed(1)}s (${stage.tokens.toLocaleString()} tokens, ${stage.doc_count} docs)`,
                        'info'
                    );
                });
                
            } else if (method === 'adversarial' && data.latency) {
                // Adversarial metrics
                addMetric('Total Latency', `${data.latency.toFixed(1)}s`, 'primary');
                addMetric('Sources Used', Object.keys(data.sources || {}).length, 'info');
                
            } else if (method === 'basic' && data.metrics) {
                // Basic metrics
                addMetric('Total Time', `${data.metrics.total_time.toFixed(1)}s`, 'primary');
                addMetric('Retrieval', `${data.metrics.retrieval_time.toFixed(1)}s`, 'info');
                addMetric('LLM Generation', `${data.metrics.llm_time.toFixed(1)}s`, 'info');
                addMetric('Context Tokens', data.metrics.tokens.toLocaleString(), 'info');
                addMetric('Documents', data.metrics.doc_count, 'info');
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

        function appendMessage(role, content, sources) {
            if (!historyContainer) return;
            const wrapper = document.createElement('div');
            wrapper.className = `agent-message agent-message--${role}`;
            const bubble = document.createElement('div');
            bubble.className = 'agent-message__bubble';
            bubble.textContent = content;
            wrapper.appendChild(bubble);

            if (role === 'assistant' && sources && sources.length && sourcesContainer && sourcesList) {
                sourcesContainer.hidden = false;
                sourcesList.innerHTML = '';
                sources.forEach(source => {
                    const item = document.createElement('li');
                    const link = document.createElement('a');
                    link.href = source.url || '#';
                    link.textContent = source.title || source.id || 'Source document';
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    item.appendChild(link);
                    sourcesList.appendChild(item);
                });
            }

            historyContainer.appendChild(wrapper);
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }

        function resetConversation() {
            if (historyContainer) historyContainer.innerHTML = '';
            if (sourcesContainer) {
                sourcesContainer.hidden = true;
            }
            if (sourcesList) sourcesList.innerHTML = '';
            if (metricsContainer) metricsContainer.hidden = true;
            clearDebugLog();
        }

        function renderHistory(history) {
            if (!historyContainer) return;
            historyContainer.innerHTML = '';
            history.forEach(item => {
                appendMessage(item.role, item.content, item.sources || []);
            });
        }

        function applyAgentEnabledState(enabled) {
            agentEnabled = enabled;
            if (disabledBanner) {
                disabledBanner.hidden = enabled && !hasAgentError;
            }
            if (form) {
                Array.from(form.elements).forEach(el => {
                    if (el === resetButton) return;
                    el.disabled = !enabled;
                });
            }
        }

        // NEW: Submit question with method selection
        async function submitQuestion(payload) {
            if (!form || !agentPanel) return;
            if (!agentEnabled) {
                if (disabledBanner) disabledBanner.hidden = false;
                return;
            }

            const method = methodSelect ? methodSelect.value : 'tiered';
            const endpoint = `/historian-agent/query-${method}`;
            
            debugLog(`=== Starting ${method.toUpperCase()} query ===`, 'primary');
            debugLog(`Endpoint: ${endpoint}`, 'info');
            debugLog(`Question: ${payload.question.substring(0, 100)}...`, 'info');
            
            setChatSubmitting(true, `Processing with ${method} method...`);
            
            try {
                const startTime = performance.now();
                
                debugLog('Sending request to backend...', 'info');
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                const elapsedTime = ((performance.now() - startTime) / 1000).toFixed(1);
                
                if (!response.ok) {
                    throw new Error(data.error || 'Query failed');
                }
                
                debugLog(`✓ Response received in ${elapsedTime}s`, 'success');
                debugLog(`Method used: ${data.method || method}`, 'info');
                
                // Log method-specific details
                if (method === 'tiered') {
                    debugLog(`Escalated to Tier 2: ${data.escalated ? 'YES' : 'NO'}`, data.escalated ? 'warning' : 'success');
                    debugLog(`Total stages: ${data.metrics ? data.metrics.length : 0}`, 'info');
                    if (data.metrics) {
                        data.metrics.forEach((stage, idx) => {
                            debugLog(`  Stage ${idx + 1}: ${stage.stage} (${stage.total_time.toFixed(1)}s, ${stage.tokens} tokens)`, 'info');
                        });
                    }
                } else if (method === 'adversarial') {
                    debugLog(`Latency: ${data.latency.toFixed(1)}s`, 'info');
                } else if (method === 'basic') {
                    debugLog(`Retrieval: ${data.metrics.retrieval_time.toFixed(1)}s`, 'info');
                    debugLog(`LLM: ${data.metrics.llm_time.toFixed(1)}s`, 'info');
                    debugLog(`Tokens: ${data.metrics.tokens}`, 'info');
                }
                
                debugLog(`Sources found: ${Object.keys(data.sources || {}).length}`, 'info');
                
                conversationId = data.conversation_id || payload.conversation_id;
                renderHistory(data.history || []);
                
                if (data.answer) {
                    appendMessage('assistant', data.answer, data.sources || []);
                }
                
                // Display metrics
                displayMetrics(data, method);
                
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

        function hydrateFromConfig() {
            if (!agentConfig) return;
            applyAgentEnabledState(Boolean(agentConfig.enabled));
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

        hydrateFromConfig();
        debugLog('Historian Agent initialized', 'success');
        debugLog(`Current method: ${methodSelect ? methodSelect.value : 'tiered'}`, 'info');
    });
})();