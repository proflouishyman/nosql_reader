// app/static/corpus_explorer.js

(function () {
    'use strict';

    // ── DOM refs ────────────────────────────────────────────────────────
    const formPanel = document.getElementById('ceFormPanel');
    const runningPanel = document.getElementById('ceRunningPanel');
    const resultsPanel = document.getElementById('ceResults');
    const form = document.getElementById('ceForm');
    const elapsedEl = document.getElementById('ceElapsed');
    const progressLog = document.getElementById('ceProgressLog');
    const estimateEl = document.getElementById('ceEstimate');
    const notebookList = document.getElementById('ceNotebookList');
    const newExplBtn = document.getElementById('ceNewExploration');
    const exportBtn = document.getElementById('ceExportMarkdown');
    const refreshNbBtn = document.getElementById('ceRefreshNotebooks');
    const runningLabel = document.getElementById('ceRunningLabel');

    if (!formPanel || !runningPanel || !resultsPanel || !form) {
        return;
    }

    // ── Strategy / scope card selection ────────────────────────────────
    document.querySelectorAll('.ce-strategy-card').forEach((card) => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.ce-strategy-card').forEach((node) => node.classList.remove('ce-strategy-card--selected'));
            card.classList.add('ce-strategy-card--selected');
            const input = card.querySelector('input');
            if (input) input.checked = true;
        });
    });

    const ESTIMATES = {
        100: '~2–5 min for Quick Sample',
        500: '~10–20 min for Standard',
        2000: '~30–60 min for Deep Scan',
        100000: 'Hours for Full Corpus — run overnight',
    };

    document.querySelectorAll('.ce-scope-card').forEach((card) => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.ce-scope-card').forEach((node) => node.classList.remove('ce-scope-card--selected'));
            card.classList.add('ce-scope-card--selected');
            const input = card.querySelector('input');
            if (input) input.checked = true;
            if (estimateEl) estimateEl.textContent = ESTIMATES[parseInt(card.dataset.value, 10)] || '';
        });
    });

    // ── State machine ───────────────────────────────────────────────────
    function setState(state) {
        formPanel.hidden = state !== 'form';
        runningPanel.hidden = state !== 'running';
        resultsPanel.hidden = state !== 'results';
    }
    setState('form');

    // ── Progress log ────────────────────────────────────────────────────
    function clearProgressLog() {
        if (progressLog) progressLog.innerHTML = '';
    }

    function logProgress(message, type = 'info') {
        if (!progressLog) return;
        const ts = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `agent-debug-entry agent-debug-entry--${type}`;

        const timeEl = document.createElement('span');
        timeEl.className = 'agent-debug-time';
        timeEl.textContent = `[${ts}]`;

        const msgEl = document.createElement('span');
        msgEl.className = 'agent-debug-message';
        msgEl.textContent = message;

        entry.appendChild(timeEl);
        entry.appendChild(msgEl);
        progressLog.appendChild(entry);
        progressLog.scrollTop = progressLog.scrollHeight;
    }

    /**
     * Open shared backend log stream and pipe events to the progress log.
     */
    function openLogStream(onDone) {
        const source = new EventSource('/api/log-stream');
        let closed = false;

        source.onmessage = function(event) {
            try {
                const payload = JSON.parse(event.data);
                const displayMsg = payload.source
                    ? `${payload.source} ${payload.message}`
                    : payload.message;
                logProgress(displayMsg, payload.level || 'info');
                if (runningLabel && payload.message) {
                    runningLabel.textContent = payload.message.length > 120
                        ? `${payload.message.slice(0, 117)}...`
                        : payload.message;
                }
            } catch (error) {
                logProgress(event.data, 'info');
            }
        };

        source.addEventListener('done', function() {
            if (!closed) {
                closed = true;
                source.close();
                if (onDone) onDone();
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

    // ── Elapsed timer ───────────────────────────────────────────────────
    let elapsedInterval = null;

    function startElapsed() {
        let seconds = 0;
        elapsedInterval = setInterval(() => {
            seconds += 1;
            const m = Math.floor(seconds / 60);
            const s = String(seconds % 60).padStart(2, '0');
            if (elapsedEl) elapsedEl.textContent = `${m}:${s}`;
        }, 1000);
    }

    function stopElapsed() {
        if (elapsedInterval) {
            clearInterval(elapsedInterval);
            elapsedInterval = null;
        }
    }

    // ── Form submission ─────────────────────────────────────────────────
    let activeReport = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const strategy = document.querySelector('input[name="strategy"]:checked')?.value || 'balanced';
        const budget = parseInt(document.querySelector('input[name="scope"]:checked')?.value || '500', 10);
        const lens = document.getElementById('ceResearchLens')?.value.trim() || null;
        const yearFrom = parseInt(document.getElementById('ceYearFrom')?.value || '', 10) || null;
        const yearTo = parseInt(document.getElementById('ceYearTo')?.value || '', 10) || null;

        const payload = {
            strategy,
            total_budget: budget,
            save_notebook: true,
        };
        if (lens) payload.research_lens = lens;
        if (yearFrom && yearTo) payload.year_range = [yearFrom, yearTo];

        setState('running');
        clearProgressLog();
        stopElapsed();
        startElapsed();
        if (runningLabel) runningLabel.textContent = 'Exploring corpus...';

        logProgress(`Starting ${strategy} exploration - budget ${budget} documents`, 'primary');
        if (lens) logProgress(`Research lens: "${lens.substring(0, 80)}..."`, 'info');
        if (yearFrom && yearTo) logProgress(`Year range: ${yearFrom}-${yearTo}`, 'info');

        // Open stream before fetch so early backend logs are not missed.
        const closeStream = openLogStream(null);
        const streamTimeout = setTimeout(() => {
            logProgress('Log stream timeout reached (10m); closing stream.', 'warning');
            closeStream();
        }, 600000);

        const start = performance.now();

        try {
            const resp = await fetch('/api/rag/explore_corpus', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const report = await resp.json();

            if (!resp.ok) {
                throw new Error(report.error || `HTTP ${resp.status}`);
            }

            const elapsed = ((performance.now() - start) / 1000).toFixed(1);
            logProgress(`Exploration complete in ${elapsed}s`, 'success');
            logProgress(
                `Generated ${report.questions?.length || 0} questions, found ${report.patterns?.length || 0} patterns`,
                'success'
            );
            activeReport = report;
            renderResults(report);
            setState('results');
            loadNotebooks();
        } catch (err) {
            logProgress(`Error: ${err.message}`, 'error');
            const backBtn = document.createElement('button');
            backBtn.textContent = '<- Back to form';
            backBtn.className = 'button button--secondary';
            backBtn.style.marginTop = '1rem';
            backBtn.onclick = () => setState('form');
            runningPanel.appendChild(backBtn);
        } finally {
            clearTimeout(streamTimeout);
            closeStream();
            stopElapsed();
        }
    });

    // ── Results rendering ───────────────────────────────────────────────

    function renderResults(report) {
        renderOverview(report.corpus_map);
        renderGrandNarrative(report.question_synthesis);
        renderAgenda(report.question_synthesis, report.questions);
        renderGaps(report.question_synthesis?.gaps);
        renderContradictions(report.contradictions);
        renderPatterns(report.patterns);
        renderEntities(report.entities);
    }

    function renderOverview(corpusMap) {
        const statsGrid = document.getElementById('ceStatsGrid');
        const notesEl = document.getElementById('ceArchiveNotes');
        if (!statsGrid || !notesEl || !corpusMap) return;

        const stats = corpusMap.statistics || {};
        const statItems = [
            ['Documents Read', stats.total_documents_read],
            ['Date Range', stats.date_range || (stats.year_min && stats.year_max ? `${stats.year_min}–${stats.year_max}` : null)],
            ['Collections', stats.collection_count],
            ['Document Types', stats.document_type_count],
            ['Entities Found', stats.total_entities],
            ['Patterns Found', stats.total_patterns],
        ];
        statsGrid.innerHTML = statItems
            .filter(([, v]) => v != null)
            .map(([label, value]) => `
                <div class="ce-stat">
                    <span class="ce-stat__value">${value}</span>
                    <span class="ce-stat__label">${label}</span>
                </div>`)
            .join('');

        if (corpusMap.archive_notes) {
            notesEl.innerHTML = formatArchiveNotes(corpusMap.archive_notes);
        } else {
            notesEl.innerHTML = '';
        }
    }

    function renderGrandNarrative(synthesis) {
        const section = document.getElementById('ceGrandNarrative');
        const questionEl = document.getElementById('ceGrandQuestion');
        const themesEl = document.getElementById('ceGrandThemes');
        if (!section || !questionEl || !themesEl) return;
        if (!synthesis?.grand_narrative) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        const gn = synthesis.grand_narrative;
        questionEl.textContent = typeof gn === 'string' ? gn : gn.question || '';
        const themes = gn.themes || [];
        themesEl.innerHTML = themes.map((t) => `<span class="ce-tag">${escHtml(t)}</span>`).join('');
    }

    function renderAgenda(synthesis, flatQuestions) {
        const container = document.getElementById('ceThemes');
        if (!container) return;
        container.innerHTML = '';

        if (synthesis?.themes?.length) {
            container.innerHTML = synthesis.themes
                .map((theme) => {
                    const qs = theme.questions || [];
                    return `
                <div class="ce-theme">
                    <div class="ce-theme__header">
                        <h3 class="ce-theme__name">${escHtml(theme.name || theme.key || 'Theme')}</h3>
                        ${theme.description ? `<p class="ce-theme__desc">${escHtml(theme.description)}</p>` : ''}
                    </div>
                    <div class="ce-theme__questions">${qs.map(renderQuestionCard).join('')}</div>
                </div>`;
                })
                .join('');

            [
                { key: 'temporal_questions', label: 'Temporal Questions' },
                { key: 'contradiction_questions', label: 'Questions from Contradictions' },
                { key: 'group_difference_questions', label: 'Group Difference Questions' },
            ].forEach(({ key, label }) => {
                const qs = synthesis[key];
                if (!qs?.length) return;
                const block = document.createElement('div');
                block.className = 'ce-theme';
                block.innerHTML = `
                    <div class="ce-theme__header">
                        <h3 class="ce-theme__name">${label}</h3>
                    </div>
                    <div class="ce-theme__questions">${qs.map(renderQuestionCard).join('')}</div>`;
                container.appendChild(block);
            });
        } else if (flatQuestions?.length) {
            container.innerHTML = `
                <div class="ce-theme">
                    <div class="ce-theme__header">
                        <h3 class="ce-theme__name">Generated Questions</h3>
                    </div>
                    <div class="ce-theme__questions">
                        ${flatQuestions.map(renderQuestionCard).join('')}
                    </div>
                </div>`;
        }
    }

    function renderQuestionCard(q) {
        const text = typeof q === 'string' ? q : q.question_text || q.question || '';
        if (!text) return '';
        const type = typeof q === 'object' ? (q.type || q.question_type || '') : '';
        const score = typeof q === 'object' ? (q.validation_score ?? q.validation?.score ?? null) : null;
        const docs = typeof q === 'object' ? (q.answerability_precheck?.doc_count ?? q.evidence_count ?? null) : null;

        const scoreBadge = score != null ? `<span class="ce-q-score ce-q-score--${scoreClass(score)}">${score}/100</span>` : '';
        const typeBadge = type ? `<span class="ce-tag ce-tag--type">${escHtml(type)}</span>` : '';
        const docsBadge = docs != null ? `<span class="ce-q-docs">${docs} docs</span>` : '';

        return `
        <div class="ce-question-card">
            <div class="ce-question-card__meta">${typeBadge}${scoreBadge}${docsBadge}</div>
            <p class="ce-question-card__text">${escHtml(text)}</p>
            <a href="/historian-agent?q=${encodeURIComponent(text)}" target="_blank"
               class="ce-question-card__investigate button button--sm">
               Investigate →
            </a>
        </div>`;
    }

    function scoreClass(score) {
        if (score >= 75) return 'good';
        if (score >= 50) return 'ok';
        return 'low';
    }

    function renderGaps(gaps) {
        const section = document.getElementById('ceGapsSection');
        const container = document.getElementById('ceGaps');
        if (!section || !container) return;
        if (!gaps?.length) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        container.innerHTML = gaps
            .map((gap) => `
            <div class="ce-gap-card">
                <h4 class="ce-gap-card__title">${escHtml(gap.gap || gap.title || '')}</h4>
                ${gap.suggested_questions?.length
                    ? `
                    <ul class="ce-gap-card__suggestions">
                        ${gap.suggested_questions
                            .map((sq) => `
                            <li>
                                <span>${escHtml(sq)}</span>
                                <a href="/historian-agent?q=${encodeURIComponent(sq)}"
                                   target="_blank" class="ce-gap-card__link">Investigate →</a>
                            </li>`)
                            .join('')}
                    </ul>`
                    : ''}
            </div>`)
            .join('');
    }

    function renderContradictions(contradictions) {
        const section = document.getElementById('ceContradictionsSection');
        const container = document.getElementById('ceContradictions');
        const badge = document.getElementById('ceContradictionCount');
        if (!section || !container || !badge) return;
        if (!contradictions?.length) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        badge.textContent = contradictions.length;
        container.innerHTML = contradictions
            .map((c) => `
            <div class="ce-contradiction-card">
                ${c.contradiction_type ? `<span class="ce-tag">${escHtml(c.contradiction_type)}</span>` : ''}
                <div class="ce-contradiction-card__claims">
                    <div class="ce-contradiction-card__claim ce-contradiction-card__claim--a">
                        <span class="ce-contradiction-card__label">Claim A</span>
                        <p>${escHtml(c.claim_a || '')}</p>
                    </div>
                    <div class="ce-contradiction-card__claim ce-contradiction-card__claim--b">
                        <span class="ce-contradiction-card__label">Claim B</span>
                        <p>${escHtml(c.claim_b || '')}</p>
                    </div>
                </div>
            </div>`)
            .join('');
    }

    function renderPatterns(patterns) {
        const section = document.getElementById('cePatternsSection');
        const container = document.getElementById('cePatterns');
        const badge = document.getElementById('cePatternCount');
        if (!section || !container || !badge) return;
        if (!patterns?.length) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        badge.textContent = patterns.length;
        container.innerHTML = patterns
            .map((p) => {
                const text = p.pattern_text || p.pattern || '';
                const conf = p.confidence || '';
                const docs = p.evidence_doc_ids?.length ?? p.evidence_count ?? '';
                return `
            <div class="ce-pattern-card">
                <div class="ce-pattern-card__meta">
                    ${conf ? `<span class="ce-tag ce-tag--conf-${conf}">${conf}</span>` : ''}
                    ${docs !== '' ? `<span class="ce-q-docs">${docs} docs</span>` : ''}
                </div>
                <p>${escHtml(text)}</p>
            </div>`;
            })
            .join('');
    }

    function renderEntities(entities) {
        const section = document.getElementById('ceEntitiesSection');
        const container = document.getElementById('ceEntities');
        const badge = document.getElementById('ceEntityCount');
        if (!section || !container || !badge) return;
        if (!entities?.length) {
            section.hidden = true;
            return;
        }
        section.hidden = false;
        badge.textContent = entities.length;
        container.innerHTML = `<div class="ce-entity-grid">${
            entities
                .slice(0, 30)
                .map((e) => {
                    const name = e.name || '';
                    const type = e.entity_type || e.type || '';
                    const count = e.doc_count ?? e.mention_count ?? '';
                    return `
                <div class="ce-entity-card">
                    <span class="ce-entity-card__name">${escHtml(name)}</span>
                    ${type ? `<span class="ce-tag ce-tag--type">${escHtml(type)}</span>` : ''}
                    ${count !== '' ? `<span class="ce-q-docs">${count} docs</span>` : ''}
                </div>`;
                })
                .join('')
        }</div>`;
    }

    // ── Notebook history ────────────────────────────────────────────────

    async function loadNotebooks() {
        try {
            const resp = await fetch('/api/rag/exploration_notebooks');
            const data = await resp.json();
            renderNotebooks(data.notebooks || []);
        } catch (_) {
            // non-critical
        }
    }

    function renderNotebooks(notebooks) {
        if (!notebookList) return;
        if (!notebooks.length) {
            notebookList.innerHTML = '<p class="ce-notebooks__empty">No saved explorations found.</p>';
            return;
        }
        notebookList.innerHTML = notebooks
            .map((nb) => {
                const kind = (nb.artifact_type || 'notebook').toLowerCase();
                const helpText = buildNotebookHelpText(nb);
                return `
            <button class="ce-notebook-item" data-path="${escAttr(nb.path)}" data-help="${escAttr(helpText)}" title="${escAttr(helpText)}">
                <span class="ce-notebook-item__name">${escHtml(nb.filename)}</span>
                <span class="ce-notebook-item__meta">
                    ${escHtml(nb.modified.slice(0, 16).replace('T', ' '))} · ${nb.size_kb} KB · ${escHtml(kind)}
                </span>
            </button>`;
            })
            .join('');

        notebookList.querySelectorAll('.ce-notebook-item').forEach((btn) => {
            btn.addEventListener('click', async () => {
                const previousLabel = btn.querySelector('.ce-notebook-item__name')?.textContent || 'exploration';
                const allButtons = Array.from(notebookList.querySelectorAll('.ce-notebook-item'));
                try {
                    allButtons.forEach((b) => { b.disabled = true; });
                    setNotebookNotice(`Loading ${previousLabel}...`);
                    const loadResp = await fetch('/api/rag/exploration_notebooks/load', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path: btn.dataset.path }),
                    });
                    if (!loadResp.ok) {
                        let errMsg = `HTTP ${loadResp.status}`;
                        try {
                            const errData = await loadResp.json();
                            errMsg = errData.error || errMsg;
                        } catch (_) {
                            // Keep HTTP fallback message.
                        }
                        throw new Error(errMsg);
                    }
                    const loadData = await loadResp.json();
                    if (loadData.status === 'loaded') {
                        let report = loadData.report || null;
                        if (!report) {
                            const reportResp = await fetch('/api/rag/exploration_report');
                            if (!reportResp.ok) {
                                throw new Error(`Failed to fetch report (HTTP ${reportResp.status})`);
                            }
                            report = await reportResp.json();
                        }
                        if (!report || report.error) {
                            throw new Error(report?.error || 'No report payload available');
                        }
                        activeReport = report;
                        setState('results');
                        renderResults(activeReport);
                        const summary = getReportSummary(activeReport);
                        const loadedPath = loadData.loaded_path || report.loaded_path || '';
                        const pathNote = loadedPath ? ` · source: ${loadedPath.split('/').pop()}` : '';
                        const summaryNote = `${summary.docs} docs, ${summary.patterns} patterns, ${summary.entities} entities, ${summary.questions} questions`;
                        const isEmpty = summary.docs === 0 && summary.patterns === 0 && summary.entities === 0;
                        setNotebookNotice(`Loaded ${previousLabel} (${summaryNote})${pathNote}.`, isEmpty);
                        showResultsStatus(`Loaded exploration: ${previousLabel} (${summaryNote})${pathNote}.`, isEmpty);
                        scrollToResults();
                    } else {
                        throw new Error(loadData.error || 'Notebook load did not return loaded status');
                    }
                } catch (err) {
                    console.error('Failed to load notebook:', err);
                    setNotebookNotice(`Failed to load exploration: ${err.message || 'unknown error'}`, true);
                } finally {
                    allButtons.forEach((b) => { b.disabled = false; });
                }
            });
        });
    }

    if (refreshNbBtn) refreshNbBtn.addEventListener('click', loadNotebooks);

    // ── Toolbar actions ─────────────────────────────────────────────────

    if (newExplBtn) newExplBtn.addEventListener('click', () => {
        stopElapsed();
        stopProgressPolling();
        setState('form');
    });
    if (exportBtn) exportBtn.addEventListener('click', () => exportMarkdown(activeReport));

    function exportMarkdown(report) {
        if (!report) return;
        const lines = [];
        const synthesis = report.question_synthesis || {};
        const metadata = report.exploration_metadata || {};

        lines.push('# Corpus Exploration Report');
        lines.push(`Generated: ${new Date().toLocaleString()}\n`);

        if (Object.keys(metadata).length) {
            lines.push('## Run Metadata');
            Object.entries(metadata).forEach(([key, value]) => {
                if (value == null || value === '') return;
                if (Array.isArray(value)) {
                    lines.push(`- **${key}:** ${value.join(', ')}`);
                } else if (typeof value === 'object') {
                    lines.push(`- **${key}:** \`${JSON.stringify(value)}\``);
                } else {
                    lines.push(`- **${key}:** ${value}`);
                }
            });
            lines.push('');
        }

        const stats = report.corpus_map?.statistics || {};
        const statEntries = Object.entries(stats).filter(([, value]) => value != null && value !== '');
        if (statEntries.length) {
            lines.push('## Corpus Statistics');
            statEntries.forEach(([key, value]) => {
                lines.push(`- **${key}:** ${value}`);
            });
            lines.push('');
        }

        if (report.corpus_map?.archive_notes) {
            lines.push('## Archive Overview');
            lines.push(report.corpus_map.archive_notes);
            lines.push('');
        }

        const gn = synthesis.grand_narrative;
        if (gn) {
            lines.push('## Central Research Question');
            lines.push(`> ${typeof gn === 'string' ? gn : gn.question || ''}`);
            lines.push('');
        }

        if (synthesis.themes?.length) {
            lines.push('## Research Agenda');
            synthesis.themes.forEach((theme) => {
                lines.push(`\n### ${theme.name || theme.key}`);
                if (theme.description) lines.push(`_${theme.description}_\n`);
                (theme.questions || []).forEach((q) => {
                    const text = typeof q === 'string' ? q : q.question_text || q.question || '';
                    lines.push(`- ${text}`);
                });
            });
            lines.push('');
        }

        [
            ['temporal_questions', 'Temporal Questions'],
            ['contradiction_questions', 'Questions from Contradictions'],
            ['group_difference_questions', 'Group Difference Questions'],
        ].forEach(([key, title]) => {
            const qs = synthesis[key] || [];
            if (!qs.length) return;
            lines.push(`## ${title}`);
            qs.forEach((q) => {
                const text = typeof q === 'string' ? q : q.question_text || q.question || '';
                if (text) lines.push(`- ${text}`);
            });
            lines.push('');
        });

        if (report.questions?.length) {
            lines.push('## All Generated Questions');
            report.questions.forEach((q) => {
                const text = typeof q === 'string' ? q : q.question_text || q.question || '';
                if (!text) return;
                const why = typeof q === 'object' ? (q.why_interesting || '') : '';
                const approach = typeof q === 'object' ? (q.approach || q.evidence_needed || '') : '';
                lines.push(`- ${text}`);
                if (why) lines.push(`  - Why: ${why}`);
                if (approach) lines.push(`  - Approach: ${approach}`);
            });
            lines.push('');
        }

        if (synthesis.gaps?.length) {
            lines.push('## Evidence Gaps');
            synthesis.gaps.forEach((g) => {
                lines.push(`\n### ${g.gap || g.title || ''}`);
                (g.suggested_questions || []).forEach((sq) => lines.push(`- ${sq}`));
            });
            lines.push('');
        }

        if (report.patterns?.length) {
            lines.push('## Patterns');
            report.patterns.forEach((p, idx) => {
                const text = p.pattern_text || p.pattern || '';
                const type = p.type || p.pattern_type || '';
                const conf = p.confidence || '';
                const evidenceDocs = p.evidence_doc_ids || p.evidence || [];
                const evidenceBlocks = p.evidence_blocks || [];
                lines.push(`\n### Pattern ${idx + 1}`);
                if (text) lines.push(`- Text: ${text}`);
                if (type) lines.push(`- Type: ${type}`);
                if (conf) lines.push(`- Confidence: ${conf}`);
                if (evidenceDocs.length) lines.push(`- Evidence docs: ${evidenceDocs.join(', ')}`);
                if (evidenceBlocks.length) lines.push(`- Evidence blocks: ${evidenceBlocks.join(', ')}`);
            });
            lines.push('');
        }

        if (report.entities?.length) {
            lines.push('## Key Entities');
            report.entities.forEach((e, idx) => {
                const name = e.name || e.canonical_name || '';
                const type = e.entity_type || e.type || '';
                const docCount = e.doc_count ?? e.mention_count ?? null;
                const variants = e.variants || [];
                lines.push(`\n### Entity ${idx + 1}`);
                if (name) lines.push(`- Name: ${name}`);
                if (type) lines.push(`- Type: ${type}`);
                if (docCount != null) lines.push(`- Count: ${docCount}`);
                if (variants.length) lines.push(`- Variants: ${variants.join(', ')}`);
            });
            lines.push('');
        }

        if (report.group_indicators?.length) {
            lines.push('## Group Indicators');
            report.group_indicators.forEach((g, idx) => {
                const groupType = g.group_type || g.type || '';
                const label = g.label || g.value || '';
                const conf = g.confidence || '';
                const evidenceBlocks = g.evidence_blocks || g.evidence || [];
                lines.push(`\n### Group Indicator ${idx + 1}`);
                if (groupType) lines.push(`- Group type: ${groupType}`);
                if (label) lines.push(`- Label: ${label}`);
                if (conf) lines.push(`- Confidence: ${conf}`);
                if (evidenceBlocks.length) lines.push(`- Evidence blocks: ${evidenceBlocks.join(', ')}`);
            });
            lines.push('');
        }

        if (report.contradictions?.length) {
            lines.push('## Contradictions');
            report.contradictions.forEach((c) => {
                if (c.contradiction_type) lines.push(`\n- Type: ${c.contradiction_type}`);
                lines.push(`\n**A:** ${c.claim_a || ''}`);
                lines.push(`**B:** ${c.claim_b || ''}`);
                if (c.context) lines.push(`Context: ${c.context}`);
                if (c.source_a) lines.push(`Source A: ${c.source_a}`);
                if (c.source_b) lines.push(`Source B: ${c.source_b}`);
            });
            lines.push('');
        }

        if (report.notebook_summary) {
            lines.push('## Notebook Summary');
            if (typeof report.notebook_summary === 'string') {
                lines.push(report.notebook_summary);
            } else {
                lines.push('```json');
                lines.push(JSON.stringify(report.notebook_summary, null, 2));
                lines.push('```');
            }
            lines.push('');
        }

        lines.push('## Full Report Payload (JSON)');
        lines.push('```json');
        lines.push(JSON.stringify(report, null, 2));
        lines.push('```');
        lines.push('');

        const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `corpus-exploration-${new Date().toISOString().slice(0, 10)}.md`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // ── Utilities ───────────────────────────────────────────────────────

    function escHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function escAttr(str) {
        return escHtml(str).replace(/'/g, '&#39;');
    }

    function buildNotebookHelpText(nb) {
        const p = nb.preview || {};
        const lines = [];
        lines.push(`Exploration file: ${nb.filename || 'unknown'}`);
        lines.push(`Type: ${nb.artifact_type || 'unknown'}`);
        if (p.loaded_file_hint && p.loaded_file_hint !== nb.filename) {
            lines.push(`Loads as: ${p.loaded_file_hint}`);
        }
        if (p.strategy) lines.push(`Strategy: ${p.strategy}`);
        if (p.total_budget != null && p.total_budget !== '') lines.push(`Budget: ${p.total_budget}`);
        if (p.year_range) lines.push(`Year range: ${p.year_range}`);
        lines.push(`Documents read: ${p.documents_read || 0}`);
        lines.push(`Questions: ${p.question_count || 0}`);
        lines.push(`Patterns: ${p.pattern_count || 0}`);
        lines.push(`Entities: ${p.entity_count || 0}`);
        lines.push(`Contradictions: ${p.contradiction_count || 0}`);
        if (p.theme_count != null || p.gap_count != null) {
            lines.push(`Themes: ${p.theme_count || 0} · Gaps: ${p.gap_count || 0}`);
        }
        if (Array.isArray(p.research_lens) && p.research_lens.length) {
            lines.push(`Research lens: ${p.research_lens.join('; ')}`);
        }
        if (p.researcher_question) {
            lines.push(`Researcher asked: ${p.researcher_question}`);
        }
        if (p.primary_question) {
            lines.push(`Question: ${p.primary_question}`);
        }
        if (p.question_quality_gate) {
            lines.push(`Question quality gate: ${p.question_quality_gate}`);
        }
        if (p.is_empty) {
            lines.push('This snapshot is mostly empty and may not show rich results.');
        }
        if (p.preview_error) {
            lines.push(`Preview note: ${p.preview_error}`);
        }
        return lines.join(' | ');
    }

    function setNotebookNotice(message, isError = false) {
        const host = document.getElementById('ceNotebooks');
        if (!host) return;
        let notice = host.querySelector('.ce-notebooks__notice');
        if (!notice) {
            notice = document.createElement('p');
            notice.className = 'ce-notebooks__notice';
            notice.style.margin = '0.5rem 0 0';
            notice.style.fontSize = '0.9rem';
            host.appendChild(notice);
        }
        notice.textContent = message || '';
        notice.style.color = isError ? '#991b1b' : '#0f5132';
    }

    function getReportSummary(report) {
        const stats = report?.corpus_map?.statistics || {};
        const docs = Number(stats.total_documents_read ?? report?.exploration_metadata?.documents_read ?? 0) || 0;
        const patterns = Array.isArray(report?.patterns) ? report.patterns.length : 0;
        const entities = Array.isArray(report?.entities) ? report.entities.length : 0;
        const questions = Array.isArray(report?.questions) ? report.questions.length : 0;
        const contradictions = Array.isArray(report?.contradictions) ? report.contradictions.length : 0;
        return { docs, patterns, entities, questions, contradictions };
    }

    function showResultsStatus(message, isWarning = false) {
        if (!resultsPanel) return;
        let banner = document.getElementById('ceResultsStatus');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'ceResultsStatus';
            banner.style.margin = '0 0 0.75rem';
            banner.style.padding = '0.55rem 0.75rem';
            banner.style.borderRadius = '8px';
            banner.style.fontWeight = '600';
            resultsPanel.insertBefore(banner, resultsPanel.firstChild);
        }
        banner.textContent = message || '';
        banner.style.background = isWarning ? '#fef3c7' : '#e9f7ef';
        banner.style.color = isWarning ? '#92400e' : '#0f5132';
        banner.style.border = isWarning ? '1px solid #fcd34d' : '1px solid #b7ebc6';
    }

    function scrollToResults() {
        if (!resultsPanel) return;
        // scrollIntoView can fail silently in some browser/layout combinations for hidden->visible transitions.
        if (typeof resultsPanel.scrollIntoView === 'function') {
            resultsPanel.scrollIntoView({ behavior: 'auto', block: 'start' });
        }
        const y = resultsPanel.getBoundingClientRect().top + window.scrollY - 16;
        window.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
    }

    function formatArchiveNotes(raw) {
        if (!raw) return '';

        const cleaned = String(raw)
            .replace(/\*\*/g, '')
            .replace(/\[(location|event|name)\]/gi, 'not evidenced in this run')
            .trim();

        const paragraphs = cleaned
            .split(/\n{2,}/)
            .map((p) => p.trim())
            .filter(Boolean);

        if (!paragraphs.length) return '';

        return paragraphs
            .map((p) => `<p>${escHtml(p).replace(/\n/g, '<br>')}</p>`)
            .join('');
    }

    // ── Init ────────────────────────────────────────────────────────────
    loadNotebooks();

    // If a report already exists in memory (e.g. after page reload), show it
    fetch('/api/rag/exploration_report')
        .then((r) => (r.ok ? r.json() : null))
        .then((report) => {
            if (report && !report.error) {
                activeReport = report;
                renderResults(report);
                setState('results');
            }
        })
        .catch(() => {});
})();
