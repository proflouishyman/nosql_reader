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
        const suggestionButtons = Array.from(document.querySelectorAll('.agent-suggestion'));
        const submitButton = form ? form.querySelector('button[type="submit"]') : null;
        const disabledBanner = document.getElementById('agentDisabledBanner');
        const errorBanner = document.getElementById('agentErrorBanner');

        const agentSettingsRoot = document.getElementById('agentSettings');
        const settingsForm = document.getElementById('agentSettingsForm');
        const configStatus = document.getElementById('agentConfigStatus');
        const providerSelect = document.getElementById('agentModelProvider');
        const enabledToggle = document.getElementById('agentEnabled');
        const ollamaSection = agentSettingsRoot ? agentSettingsRoot.querySelector('.agent-provider--ollama') : null;
        const openaiSection = agentSettingsRoot ? agentSettingsRoot.querySelector('.agent-provider--openai') : null;
        const ollamaBaseUrlInput = document.getElementById('agentOllamaBaseUrl');
        const openAiKeyInput = document.getElementById('agentOpenAiKey');
        const modelNameInput = document.getElementById('agentModelName');
        const temperatureInput = document.getElementById('agentTemperature');
        const contextDocsInput = document.getElementById('agentContextDocuments');
        const systemPromptInput = document.getElementById('agentSystemPrompt');
        const contextFieldsInput = document.getElementById('agentContextFields');
        const summaryFieldInput = document.getElementById('agentSummaryField');
        const fallbackToggle = document.getElementById('agentAllowFallback');
        const resetConfigButton = document.getElementById('agentSettingsReset');

        const configEndpoint = agentSettingsRoot ? agentSettingsRoot.dataset.configEndpoint : null;

        let conversationId = null;
        let isSubmitting = false;
        let agentEnabled = agentConfig ? Boolean(agentConfig.enabled) : true;
        let hasAgentError = agentPanel ? agentPanel.dataset.agentError === 'true' : false;

        function updateProviderSections(provider) {
            if (ollamaSection) {
                ollamaSection.hidden = provider !== 'ollama';
            }
            if (openaiSection) {
                openaiSection.hidden = provider !== 'openai';
            }
        }

        function markConfigStatus(message, type) {
            if (!configStatus) return;
            configStatus.textContent = message || '';
            configStatus.hidden = !message;
            if (!message) {
                delete configStatus.dataset.status;
                return;
            }
            configStatus.dataset.status = type || 'info';
        }

        function setConfigSubmitting(active) {
            if (!settingsForm) return;
            const saveButton = settingsForm.querySelector('button[type="submit"]');
            if (saveButton) saveButton.disabled = active;
            if (resetConfigButton) resetConfigButton.disabled = active;
        }

        function setChatSubmitting(active) {
            isSubmitting = active;
            if (submitButton) submitButton.disabled = active;
            if (status) {
                status.hidden = !active;
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

        function updateConfigInputs(payload) {
            if (!settingsForm || !payload) return;
            if (typeof payload.enabled !== 'undefined' && enabledToggle) {
                enabledToggle.checked = Boolean(payload.enabled);
            }
            if (payload.model_provider && providerSelect) {
                providerSelect.value = payload.model_provider;
                updateProviderSections(payload.model_provider);
            }
            if (typeof payload.ollama_base_url !== 'undefined' && ollamaBaseUrlInput) {
                ollamaBaseUrlInput.value = payload.ollama_base_url || '';
            }
            if (payload.openai_api_key_present && openAiKeyInput) {
                openAiKeyInput.placeholder = 'API key configured';
            }
            if (typeof payload.model_name !== 'undefined' && modelNameInput) {
                modelNameInput.value = payload.model_name || '';
            }
            if (typeof payload.temperature !== 'undefined' && temperatureInput) {
                temperatureInput.value = Number(payload.temperature).toFixed(2);
            }
            if (typeof payload.max_context_documents !== 'undefined' && contextDocsInput) {
                contextDocsInput.value = payload.max_context_documents;
            }
            if (typeof payload.system_prompt !== 'undefined' && systemPromptInput) {
                systemPromptInput.value = payload.system_prompt || '';
            }
            if (payload.context_fields && contextFieldsInput) {
                contextFieldsInput.value = Array.isArray(payload.context_fields) ? payload.context_fields.join('\n') : payload.context_fields;
            }
            if (typeof payload.summary_field !== 'undefined' && summaryFieldInput) {
                summaryFieldInput.value = payload.summary_field || '';
            }
            if (typeof payload.allow_general_fallback !== 'undefined' && fallbackToggle) {
                fallbackToggle.checked = Boolean(payload.allow_general_fallback);
            }
        }

        function hydrateFromConfig() {
            if (!agentConfig) return;
            applyAgentEnabledState(Boolean(agentConfig.enabled));
            updateConfigInputs(agentConfig);
        }

        async function submitQuestion(payload) {
            if (!form || !agentPanel) return;
            if (!agentEnabled) {
                if (disabledBanner) disabledBanner.hidden = false;
                return;
            }
            setChatSubmitting(true);
            try {
                const response = await fetch('/historian-agent/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Historian Agent request failed.');
                }
                conversationId = data.conversation_id || payload.conversation_id;
                renderHistory(data.history || []);
                if (data.answer) {
                    appendMessage('assistant', data.answer, data.sources || []);
                }
                if (questionInput) {
                    questionInput.value = '';
                    questionInput.focus();
                }
            } catch (error) {
                console.error(error);
                appendMessage('assistant', error.message || 'Unable to process your question right now.', []);
            } finally {
                setChatSubmitting(false);
            }
        }

        function serializeSettingsForm() {
            if (!settingsForm) return {};
            const formData = new FormData(settingsForm);
            const payload = {};
            formData.forEach((value, key) => {
                if (payload[key]) {
                    if (!Array.isArray(payload[key])) {
                        payload[key] = [payload[key]];
                    }
                    payload[key].push(value);
                } else {
                    payload[key] = value;
                }
            });
            payload.enabled = settingsForm.querySelector('#agentEnabled')?.checked;
            payload.allow_general_fallback = settingsForm.querySelector('#agentAllowFallback')?.checked;
            return payload;
        }

        async function submitSettings(payload) {
            if (!configEndpoint) return;
            setConfigSubmitting(true);
            markConfigStatus('Saving configuration…');
            try {
                const response = await fetch(configEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Unable to update configuration.');
                }
                agentConfig = data.config;
                applyAgentEnabledState(Boolean(agentConfig.enabled));
                updateConfigInputs(agentConfig);
                markConfigStatus(data.message || 'Configuration saved.', 'success');
                hasAgentError = Boolean(data.agent_error);
                if (disabledBanner) disabledBanner.hidden = agentEnabled && !hasAgentError;
                if (errorBanner) {
                    errorBanner.textContent = data.agent_error || '';
                    errorBanner.hidden = !data.agent_error;
                }
            } catch (error) {
                console.error(error);
                markConfigStatus(error.message || 'Failed to save configuration.', 'error');
            } finally {
                setConfigSubmitting(false);
            }
        }

        async function resetSettings() {
            if (!configEndpoint) return;
            setConfigSubmitting(true);
            markConfigStatus('Restoring defaults…');
            try {
                const response = await fetch(configEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ reset: true })
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Unable to reset configuration.');
                }
                agentConfig = data.config;
                updateConfigInputs(agentConfig);
                applyAgentEnabledState(Boolean(agentConfig.enabled));
                markConfigStatus(data.message || 'Configuration reset.', 'success');
                hasAgentError = Boolean(data.agent_error);
                if (disabledBanner) disabledBanner.hidden = agentEnabled && !hasAgentError;
                if (errorBanner) {
                    errorBanner.textContent = data.agent_error || '';
                    errorBanner.hidden = !data.agent_error;
                }
            } catch (error) {
                console.error(error);
                markConfigStatus(error.message || 'Failed to reset configuration.', 'error');
            } finally {
                setConfigSubmitting(false);
            }
        }

        if (providerSelect) {
            providerSelect.addEventListener('change', function(event) {
                updateProviderSections(event.target.value);
            });
        }

        if (settingsForm) {
            settingsForm.addEventListener('submit', function(event) {
                event.preventDefault();
                const payload = serializeSettingsForm();
                submitSettings(payload);
            });
        }

        if (resetConfigButton) {
            resetConfigButton.addEventListener('click', function() {
                resetSettings();
            });
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
    });
})();
