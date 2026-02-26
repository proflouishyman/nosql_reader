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
        const resetButton = document.getElementById('agentReset');
        const status = document.getElementById('agentStatus');
        const statusText = document.getElementById('agentStatusText');
        const suggestionButtons = Array.from(document.querySelectorAll('.agent-suggestion'));
        const submitButton = form ? form.querySelector('button[type="submit"]') : null;
        const refinementToggleInline = document.getElementById('refinementToggleInline');
        const refinementToggleSettings = document.getElementById('refinementToggleSettings');
        const refinementModal = document.getElementById('refinementModal');
        const refinementSuggestions = document.getElementById('refinementSuggestions');
        const searchAsIsButton = document.getElementById('searchAsIs');
        const searchRefinedButton = document.getElementById('searchRefined');
        const REFINEMENT_STORAGE_KEY = 'refinement_enabled';
        const REFINEMENT_SUGGESTIONS = [
            (question) => `What specific time period does this question cover? Try: "${question} between 1900 and 1920".`,
            (question) => `Is there a specific person, role, or location? Try adding: "${question} [name/place]".`,
            () => 'Are you looking for causes, counts, or examples? Consider naming one clearly.'
        ];

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
        let pendingRefinementQuestion = '';

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

        function isRefinementEnabled() {
            try {
                const storedValue = window.localStorage.getItem(REFINEMENT_STORAGE_KEY);
                return storedValue !== 'false';
            } catch (error) {
                return true;
            }
        }

        function applyRefinementToggleState(enabled) {
            if (refinementToggleInline) {
                refinementToggleInline.checked = enabled;
            }
            if (refinementToggleSettings) {
                refinementToggleSettings.checked = enabled;
            }
        }

        function persistRefinementToggle(enabled) {
            try {
                window.localStorage.setItem(REFINEMENT_STORAGE_KEY, enabled ? 'true' : 'false');
            } catch (error) {
                // Ignore localStorage write failures and keep in-memory toggle state active.
            }
            applyRefinementToggleState(enabled);
        }

        function closeRefinementModal() {
            if (refinementModal) {
                refinementModal.style.display = 'none';
            }
            pendingRefinementQuestion = '';
        }

        function showRefinementModal(question) {
            if (!refinementModal || !refinementSuggestions) {
                return false;
            }

            pendingRefinementQuestion = question;
            refinementSuggestions.innerHTML = REFINEMENT_SUGGESTIONS
                .map((buildSuggestion) => `<p class="agent-refinement-tip">${buildSuggestion(question)}</p>`)
                .join('');
            refinementModal.style.display = 'flex';
            return true;
        }

        function shouldSuggestRefinement(question) {
            if (!isRefinementEnabled()) {
                return false;
            }
            const wordCount = question.split(/\s+/).filter(Boolean).length;
            return wordCount > 0 && wordCount < 5;
        }

        // Keep inline and settings toggles synchronized through one local-storage preference.
        applyRefinementToggleState(isRefinementEnabled());
        [refinementToggleInline, refinementToggleSettings].forEach((toggle) => {
            if (!toggle) {
                return;
            }
            toggle.addEventListener('change', function() {
                persistRefinementToggle(Boolean(toggle.checked));
            });
        });

        if (searchAsIsButton) {
            searchAsIsButton.addEventListener('click', function() {
                const question = pendingRefinementQuestion;
                closeRefinementModal();
                if (!question || isSubmitting) {
                    return;
                }
                executeUserQuestion(question);
            });
        }

        if (searchRefinedButton) {
            searchRefinedButton.addEventListener('click', function() {
                closeRefinementModal();
                if (questionInput) {
                    questionInput.focus();
                    try {
                        questionInput.setSelectionRange(0, questionInput.value.length);
                    } catch (error) {
                        // Keep focus behavior even if selection APIs are unavailable.
                    }
                }
            });
        }

        if (refinementModal) {
            refinementModal.addEventListener('click', function(event) {
                if (event.target === refinementModal) {
                    closeRefinementModal();
                }
            });
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape' && refinementModal.style.display !== 'none') {
                    closeRefinementModal();
                }
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

            if (role === 'assistant') {
                // Render Markdown first, then annotate inline [Source N] markers as footnotes.
                const rawHtml = typeof marked !== 'undefined' ? marked.parse(content) : content;
                const sourceRefMap = buildSourceReferenceMap(sources);
                const annotatedHtml = injectFootnotes(rawHtml, sourceRefMap);
                bubble.innerHTML = annotatedHtml;
                bubble.classList.add('agent-message__bubble--research');
                styleHistorianSections(bubble);
            } else {
                bubble.textContent = content;
            }

            if (role === 'assistant') {
                renderInlineMetrics(bubble, inlineMetrics);
            }

            wrapper.appendChild(bubble);

            if (role === 'assistant' && sources && typeof sources === 'object' && Object.keys(sources).length > 0) {
                const footnoteBlock = buildFootnoteBlock(sources);
                // Keep citations visually at the bottom of each answer bubble.
                bubble.appendChild(footnoteBlock);
            }

            historyContainer.appendChild(wrapper);
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }

        function escapeHtml(value) {
            return String(value ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        }

        /**
         * Replace citation markers in rendered assistant HTML with linked superscripts.
         * Handles common model variants: [Source N], (Source N), (Sources N-M), etc.
         */
        function injectFootnotes(html, sourceRefMap) {
            const sourcePattern = /(?:\[\s*([Ss]ources?[^\]]*)\s*\]|\(\s*([Ss]ources?[^\)]*)\s*\)|\b([Ss]ources?)\s*\[\s*([^\]]+)\s*\]|\[\s*(\d[^\]]*)\s*\])/g;
            const withRefs = String(html || '').replace(sourcePattern, (match, bracketLabel, parenLabel, sourceWord, sourceBracketBody, bracketBodyOnly) => {
                let marker = String(bracketLabel || parenLabel || '').trim();
                if (!marker && sourceWord && sourceBracketBody) {
                    // Normalize "Source [1: RDApp-...]" into a marker string we can parse.
                    marker = `${sourceWord} ${sourceBracketBody}`;
                } else if (!marker && bracketBodyOnly) {
                    // Normalize bare numeric bracket citations like "[1: RDApp-...]" to source markers.
                    marker = `Source ${bracketBodyOnly}`;
                }
                const refs = resolveMarkerReferences(marker, sourceRefMap);
                if (!refs.length) {
                    return match;
                }
                const linkHtml = refs.map((ref) => {
                    const safeTitle = escapeHtml(ref.tooltip);
                    return `<a href="#fn-${ref.num}" class="fn-link" title="${safeTitle}" aria-label="${safeTitle}">${ref.num}</a>`;
                }).join('<span class="fn-link-sep">,</span>');
                const multiClass = refs.length > 1 ? ' fn-ref--multi' : '';
                return `<sup class="fn-ref${multiClass}">${linkHtml}</sup>`;
            });
            // Chicago-style placement: if punctuation follows the marker, place superscript after punctuation.
            return moveLeadingSuperscriptsToSentenceEnd(moveSuperscriptsToClaimEnd(withRefs));
        }

        function buildSourceReferenceMap(sources) {
            const map = { byLabel: {}, byNumber: {} };
            if (!sources || typeof sources !== 'object') {
                return map;
            }

            Object.entries(sources).forEach(([label, source], idx) => {
                const num = idx + 1;
                const citationHtml = formatArchivalCitation(source?.display_name, source || {});
                const citationText = decodeHtmlEntities(stripHtmlTags(citationHtml));
                const tooltip = `${num}. ${citationText}`;
                const variants = new Set();

                variants.add(normalizeSourceLabel(label));
                variants.add(normalizeSourceLabel(`Source ${num}`));
                variants.add(normalizeSourceLabel(`Sources ${num}`));

                const labelMatch = String(label || '').match(/source\s*(\d+)/i);
                if (labelMatch) {
                    variants.add(normalizeSourceLabel(`Source ${labelMatch[1]}`));
                }

                map.byNumber[num] = { num, tooltip };
                variants.forEach((variant) => {
                    if (!variant) {
                        return;
                    }
                    map.byLabel[variant] = { num, tooltip };
                });
            });

            return map;
        }

        /**
         * Resolve citation marker text to concrete source references.
         * Uses exact-label matching first, then parses valid source numbers/ranges.
         */
        function resolveMarkerReferences(marker, sourceRefMap) {
            const normalizedKey = normalizeSourceLabel(marker);
            const direct = sourceRefMap.byLabel?.[normalizedKey];
            if (direct) {
                return [direct];
            }

            const numbers = extractSourceNumbers(marker);
            if (!numbers.length) {
                return [];
            }

            const refs = [];
            const seen = new Set();
            numbers.forEach((num) => {
                const ref = sourceRefMap.byNumber?.[num];
                if (ref && !seen.has(ref.num)) {
                    seen.add(ref.num);
                    refs.push(ref);
                }
            });
            return refs;
        }

        /**
         * Parse source numbers from markers like:
         *   "Source 1: RDApp-225116Hupp003"
         *   "Sources 2-10"
         *   "Sources 1, 3, and 5"
         */
        function extractSourceNumbers(marker) {
            let working = String(marker || '').toLowerCase();
            if (!working) {
                return [];
            }

            // Normalize dash variants to simplify range parsing.
            working = working.replace(/[–—]/g, '-');
            // Ignore explanatory suffixes after colon/semicolon to avoid document-ID digits.
            working = working.split(/[:;]/, 1)[0];
            // Strip "source"/"sources" prefix when present.
            working = working.replace(/^\s*sources?\s*/i, '');

            const numbers = new Set();

            const addRange = (startRaw, endRaw) => {
                const start = Number(startRaw);
                const end = Number(endRaw);
                if (!Number.isInteger(start) || !Number.isInteger(end) || start <= 0 || end <= 0) {
                    return;
                }
                const from = Math.min(start, end);
                const to = Math.max(start, end);
                // Guard against pathological spans in malformed text.
                if ((to - from) > 30) {
                    return;
                }
                for (let value = from; value <= to; value += 1) {
                    numbers.add(value);
                }
            };

            working.replace(/\b(\d{1,3})\s*-\s*(\d{1,3})\b/g, (_m, start, end) => {
                addRange(start, end);
                return '';
            });

            working.replace(/\b(\d{1,3})\s*(?:to|through)\s*(\d{1,3})\b/g, (_m, start, end) => {
                addRange(start, end);
                return '';
            });

            const singleMatches = working.match(/\b\d{1,3}\b/g) || [];
            singleMatches.forEach((token) => {
                const value = Number(token);
                if (Number.isInteger(value) && value > 0) {
                    numbers.add(value);
                }
            });

            return Array.from(numbers).sort((a, b) => a - b);
        }

        function normalizeSourceLabel(label) {
            return String(label || '').replace(/\s+/g, ' ').trim().toLowerCase();
        }

        function stripHtmlTags(value) {
            return String(value || '').replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
        }

        function decodeHtmlEntities(value) {
            const decoder = document.createElement('textarea');
            decoder.innerHTML = String(value || '');
            return decoder.value;
        }

        function moveSuperscriptsToClaimEnd(html) {
            return String(html || '').replace(
                /\s*(<sup class="fn-ref[^"]*">[\s\S]*?<\/sup>)\s*([.,;:!?])/g,
                '$2$1'
            );
        }

        /**
         * If a sentence starts with a citation marker (for example "Source [1] ..."),
         * move that citation to the end of the sentence to match Chicago-style placement.
         */
        function moveLeadingSuperscriptsToSentenceEnd(html) {
            return String(html || '').replace(
                /(^|<p>\s*|<li>\s*|[.!?]\s+)\s*(<sup class="fn-ref[^"]*">[\s\S]*?<\/sup>)\s*([^<]*?)([.!?])(?=(\s|<\/p>|<\/li>|$))/g,
                '$1$3$4$2'
            );
        }

        function buildFootnoteBlock(sources) {
            const block = document.createElement('div');
            block.className = 'agent-footnotes';

            const heading = document.createElement('p');
            heading.className = 'agent-footnotes__heading';
            heading.textContent = 'Sources';
            block.appendChild(heading);

            const ol = document.createElement('ol');
            ol.className = 'agent-footnotes__list';

            Object.entries(sources).forEach(([, source], idx) => {
                const num = idx + 1;
                const li = document.createElement('li');
                li.id = `fn-${num}`;
                li.className = 'agent-footnotes__item';

                const citation = formatArchivalCitation(source?.display_name, source || {});
                const safeUrl = escapeHtml(source?.url || '#');
                li.innerHTML = `
                    <span class="fn-num">${num}.</span>
                    <span class="fn-citation">
                        ${citation}
                        <a class="fn-open" href="${safeUrl}" target="_blank"
                           rel="noopener noreferrer"
                           data-help="Open this source document in a new tab.">→ Open</a>
                    </span>
                `;
                ol.appendChild(li);
            });

            block.appendChild(ol);
            return block;
        }

        function formatArchivalCitation(displayName, source) {
            if (!displayName) {
                return escapeHtml(source?.url || 'Unknown source');
            }
            const safeDisplay = escapeHtml(displayName);
            if (String(displayName).includes('Baltimore')) {
                return `<em>${safeDisplay}</em>`;
            }
            return `Baltimore &amp; Ohio Railroad, Relief Department Records, <em>${safeDisplay}</em>`;
        }

        function styleHistorianSections(bubble) {
            bubble.querySelectorAll('p').forEach((p) => {
                const text = p.textContent.trim();
                if (text.startsWith('What the record does not show')) {
                    p.classList.add('historian-gap');
                } else if (text.startsWith('To investigate further')) {
                    p.classList.add('historian-leads');
                } else if (text.includes('single record') && text.includes('corroborated')) {
                    p.classList.add('historian-caveat');
                }
            });
        }

        function resetConversation() {
            if (historyContainer) {
                historyContainer.innerHTML = '';
            }
            clearDebugLog();
        }

        function renderHistory(history, latestMetrics, method) {
            if (!historyContainer) {
                return;
            }

            historyContainer.innerHTML = '';

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

        function executeUserQuestion(question) {
            appendMessage('user', question, {}, null);
            submitQuestion({
                question,
                conversation_id: conversationId,
            });
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
                if (shouldSuggestRefinement(question) && showRefinementModal(question)) {
                    return;
                }
                executeUserQuestion(question);
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
