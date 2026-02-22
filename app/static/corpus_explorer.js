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
    function logProgress(message, type = 'info') {
        if (!progressLog) return;
        const icons = { info: 'ℹ', success: '✓', warning: '⚠', error: '✗' };
        const colors = { info: '#94a3b8', success: '#22c55e', warning: '#f59e0b', error: '#ef4444' };
        const entry = document.createElement('div');
        entry.className = 'agent-debug-entry';
        const ts = new Date().toLocaleTimeString();
        entry.innerHTML = `<span class="agent-debug-time">[${ts}]</span> `
            + `<span style="color:${colors[type] || colors.info}">${icons[type] || icons.info} ${message}</span>`;
        progressLog.appendChild(entry);
        progressLog.scrollTop = progressLog.scrollHeight;
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
        if (progressLog) progressLog.innerHTML = '';
        stopElapsed();
        startElapsed();

        logProgress(`Starting ${strategy} exploration — budget ${budget} documents`);
        if (lens) logProgress(`Research lens: "${lens.substring(0, 80)}…"`);
        if (yearFrom && yearTo) logProgress(`Year range: ${yearFrom}–${yearTo}`);
        logProgress('Sending request to backend…');

        const start = performance.now();

        try {
            const resp = await fetch('/api/rag/explore_corpus', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            stopElapsed();
            const elapsed = ((performance.now() - start) / 1000).toFixed(1);

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.error || `HTTP ${resp.status}`);
            }

            const report = await resp.json();
            logProgress(`Exploration complete in ${elapsed}s`, 'success');
            logProgress(
                `Generated ${report.questions?.length || 0} questions, `
                + `found ${report.patterns?.length || 0} patterns`,
                'success'
            );

            activeReport = report;
            renderResults(report);
            setState('results');
            loadNotebooks();
        } catch (err) {
            stopElapsed();
            logProgress(`Error: ${err.message}`, 'error');
            const backBtn = document.createElement('button');
            backBtn.textContent = '← Back to form';
            backBtn.className = 'button button--secondary';
            backBtn.style.marginTop = '1rem';
            backBtn.onclick = () => setState('form');
            runningPanel.appendChild(backBtn);
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
            notesEl.innerHTML = `<p>${escHtml(corpusMap.archive_notes)}</p>`;
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
            .map((nb) => `
            <button class="ce-notebook-item" data-path="${escAttr(nb.path)}">
                <span class="ce-notebook-item__name">${escHtml(nb.filename)}</span>
                <span class="ce-notebook-item__meta">
                    ${escHtml(nb.modified.slice(0, 16).replace('T', ' '))} · ${nb.size_kb} KB
                </span>
            </button>`)
            .join('');

        notebookList.querySelectorAll('.ce-notebook-item').forEach((btn) => {
            btn.addEventListener('click', async () => {
                try {
                    const loadResp = await fetch('/api/rag/exploration_notebooks/load', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path: btn.dataset.path }),
                    });
                    const loadData = await loadResp.json();
                    if (loadData.status === 'loaded') {
                        const reportResp = await fetch('/api/rag/exploration_report');
                        activeReport = await reportResp.json();
                        renderResults(activeReport);
                        setState('results');
                    }
                } catch (err) {
                    console.error('Failed to load notebook:', err);
                }
            });
        });
    }

    if (refreshNbBtn) refreshNbBtn.addEventListener('click', loadNotebooks);

    // ── Toolbar actions ─────────────────────────────────────────────────

    if (newExplBtn) newExplBtn.addEventListener('click', () => setState('form'));
    if (exportBtn) exportBtn.addEventListener('click', () => exportMarkdown(activeReport));

    function exportMarkdown(report) {
        if (!report) return;
        const lines = [];
        const synthesis = report.question_synthesis || {};

        lines.push('# Corpus Exploration Report');
        lines.push(`Generated: ${new Date().toLocaleString()}\n`);

        const stats = report.corpus_map?.statistics || {};
        if (stats.total_documents_read) lines.push(`**Documents read:** ${stats.total_documents_read}`);
        if (stats.date_range || (stats.year_min && stats.year_max)) {
            lines.push(`**Date range:** ${stats.date_range || `${stats.year_min}–${stats.year_max}`}`);
        }
        lines.push('');

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

        if (synthesis.gaps?.length) {
            lines.push('## Evidence Gaps');
            synthesis.gaps.forEach((g) => {
                lines.push(`\n### ${g.gap || g.title || ''}`);
                (g.suggested_questions || []).forEach((sq) => lines.push(`- ${sq}`));
            });
            lines.push('');
        }

        if (report.contradictions?.length) {
            lines.push('## Contradictions');
            report.contradictions.forEach((c) => {
                lines.push(`\n**A:** ${c.claim_a || ''}`);
                lines.push(`**B:** ${c.claim_b || ''}`);
            });
            lines.push('');
        }

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
