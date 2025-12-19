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
        
        // Debug console elements
        const debugConsole = document.getElementById('agentDebugConsole');
        const debugOutput = document.getElementById('agentDebugOutput');
        const debugClear = document.getElementById('agentDebugClear');
        
        // Metrics display
        const metricsContainer = document.getElementById('agentMetrics');
        const metricsGrid = document.getElementById('agentMetricsGrid');

        let conversationId = null;
        let isSubmitting = false;

        // Method descriptions
        const METHOD_HINTS = {
            'basic': 'Fast hybrid retrieval with direct LLM generation (~15-30s)',
            'adversarial': 'Same as Good but with detailed pipeline monitoring (~15-30s)',
            'tiered': 'Best quality with confidence-based escalation (~20-60s)'
        };

        // Update hint when method changes
        if (methodSelect && methodHint) {
            methodSelect.addEventListener('change', function() {
                methodHint.textContent = METHOD_HINTS[methodSelect.value] || '';
            });
        }

        // Debug console functions
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

        // historian_agent.js - Updated displayMetrics function
        function displayMetrics(data, method) {
            if (!metricsContainer || !metricsGrid) return;
            
            metricsGrid.innerHTML = '';
            metricsContainer.hidden = false;

            // 1. Handle Tiered (data.metrics is an Array)
            if (method === 'tiered' && Array.isArray(data.metrics)) {
                const escalated = data.escalated ? 'Yes (Tier 2 triggered)' : 'No (Tier 1 sufficient)';
                addMetric('Escalated', escalated, data.escalated ? 'warning' : 'success');
                addMetric('Total Duration', `${(data.total_duration || 0).toFixed(1)}s`, 'primary');
                
                data.metrics.forEach(stage => {
                    addMetric(
                        stage.stage,
                        `${(stage.total_time || 0).toFixed(1)}s (${(stage.tokens || 0).toLocaleString()} tokens)`,
                        'info'
                    );
                });
                
            // 2. Handle Adversarial (data.latency is a float)
            } else if (method === 'adversarial') {
                addMetric('Total Latency', `${(data.latency || 0).toFixed(1)}s`, 'primary');
                addMetric('Sources Used', Object.keys(data.sources || {}).length, 'info');
                
            // 3. Handle Basic (data.metrics is an Object)
            } else if (method === 'basic' && data.metrics) {
                addMetric('Total Time', `${(data.metrics.total_time || 0).toFixed(1)}s`, 'primary');
                addMetric('Retrieval', `${(data.metrics.retrieval_time || 0).toFixed(1)}s`, 'info');
                addMetric('LLM Generation', `${(data.metrics.llm_time || 0).toFixed(1)}s`, 'info');
                addMetric('Context Tokens', (data.metrics.tokens || 0).toLocaleString(), 'info');
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
            
            // Render markdown for assistant messages
            if (role === 'assistant' && typeof marked !== 'undefined') {
                bubble.innerHTML = marked.parse(content);
            } else {
                bubble.textContent = content;
            }
            
            wrapper.appendChild(bubble);

            // Display sources as clickable document links
            if (role === 'assistant' && sources && typeof sources === 'object' && Object.keys(sources).length > 0) {
                sourcesContainer.hidden = false;
                sourcesList.innerHTML = '';
                
                // sources is {"filename": "doc_id", ...}
                Object.entries(sources).forEach(([filename, docId]) => {
                    const item = document.createElement('li');
                    const link = document.createElement('a');
                    
                    // Create link with search_id for prev/next navigation
                    if (searchId) {
                        link.href = `/document/${docId}?search_id=${searchId}`;
                    } else {
                        link.href = `/document/${docId}`;
                    }
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    
                    // Display filename without .json extension
                    const displayName = filename.replace(/\.json$/i, '');
                    link.textContent = displayName;
                    
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
                // History items don't have search_id, so sources won't have prev/next
                appendMessage(item.role, item.content, item.sources || {}, null);
            });
        }

        // Submit question with method selection
        async function submitQuestion(payload) {
            if (!form || !agentPanel) return;

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
                    appendMessage('assistant', data.answer, data.sources || {}, data.search_id);
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