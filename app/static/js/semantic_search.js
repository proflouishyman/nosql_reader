(function () {
    'use strict';

    // Keep token filtering lightweight so meaning summaries stay deterministic and fast in-browser.
    const STOP_WORDS = new Set([
        'about', 'after', 'again', 'also', 'among', 'and', 'are', 'because', 'been', 'before',
        'between', 'both', 'but', 'can', 'could', 'each', 'for', 'from', 'had', 'has', 'have',
        'into', 'its', 'just', 'may', 'more', 'most', 'not', 'only', 'other', 'our', 'out',
        'over', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
        'they', 'this', 'those', 'through', 'under', 'very', 'was', 'were', 'what', 'when',
        'where', 'which', 'while', 'with', 'would', 'your'
    ]);

    const initSemanticSearch = function () {
        const form = document.getElementById('semanticSearchForm');
        const queryInput = document.getElementById('semanticQuery');
        const topKInput = document.getElementById('topK');
        const searchButton = document.getElementById('semanticSearchBtn');
        const clearButton = document.getElementById('clearSemanticBtn');
        const panel = document.getElementById('searchSpacePanel');
        const resultsContainer = document.getElementById('resultsContainer');
        const meaningContainer = document.getElementById('meaningContainer');
        const neighborsViewButton = document.getElementById('neighborsViewBtn');
        const meaningViewButton = document.getElementById('meaningViewBtn');
        const meta = document.getElementById('semanticMeta');

        if (!form || !queryInput || !topKInput || !searchButton || !clearButton ||
            !panel || !resultsContainer || !meaningContainer || !neighborsViewButton ||
            !meaningViewButton || !meta) {
            return;
        }

        let activeView = 'neighbors';
        let metaPrefix = '';

        function ensureNativeHelp(root) {
            // Mirror data-help into title as a fallback when delayed custom tooltips are unavailable.
            const scope = root && root.querySelectorAll ? root : document;
            const nodes = scope.querySelectorAll('[data-help]');
            nodes.forEach(function (node) {
                const helpText = safeText(node.getAttribute('data-help'));
                if (!helpText) {
                    return;
                }
                if (!node.getAttribute('title')) {
                    node.setAttribute('title', helpText);
                }
            });
        }

        function attachTooltips(target) {
            // Explicitly attach for dynamically created nodes so delayed tooltips always work.
            ensureNativeHelp(target);
            if (window.AppHelpTooltips && typeof window.AppHelpTooltips.attach === 'function') {
                window.AppHelpTooltips.attach(target);
            }
        }

        function parseTopK(value) {
            const parsed = Number.parseInt(String(value), 10);
            if (!Number.isFinite(parsed)) {
                return 20;
            }
            return Math.max(1, Math.min(parsed, 100));
        }

        function asNumber(value, fallback) {
            const parsed = Number(value);
            return Number.isFinite(parsed) ? parsed : fallback;
        }

        function formatScore(value) {
            return asNumber(value, 0).toFixed(3);
        }

        function safeText(value) {
            return String(value || '').trim();
        }

        function extractTokens(text) {
            return safeText(text)
                .toLowerCase()
                .replace(/[^a-z0-9\s]/g, ' ')
                .split(/\s+/)
                .filter(function (token) {
                    return token && token.length > 2 && !STOP_WORDS.has(token);
                });
        }

        function tokenCountToTopList(countMap, limit) {
            return Array.from(countMap.entries())
                .sort(function (a, b) {
                    if (b[1] !== a[1]) {
                        return b[1] - a[1];
                    }
                    return a[0].localeCompare(b[0]);
                })
                .slice(0, limit)
                .map(function (entry) {
                    return { term: entry[0], count: entry[1] };
                });
        }

        function buildMeaningModel(query, results) {
            const queryTokens = new Set(extractTokens(query));
            const aggregatedTermCounts = new Map();
            const queryOverlapCounts = new Map();
            const groupsByDocument = new Map();
            const scoreValues = [];
            let strongCount = 0;
            let mediumCount = 0;
            let broadCount = 0;

            (results || []).forEach(function (result) {
                const score = asNumber(result && result.score, 0);
                scoreValues.push(score);
                if (score >= 0.8) {
                    strongCount += 1;
                } else if (score >= 0.65) {
                    mediumCount += 1;
                } else {
                    broadCount += 1;
                }

                const title = safeText(result && result.title) || 'Untitled';
                const docId = safeText(result && result.doc_id);
                const date = safeText(result && result.date);
                const source = safeText(result && result.source);
                const snippet = safeText(result && result.content);
                const groupKey = docId || (title + '|' + date) || safeText(result && result.chunk_id);
                const tokens = extractTokens(title + ' ' + snippet + ' ' + source);

                tokens.forEach(function (token) {
                    if (queryTokens.has(token)) {
                        queryOverlapCounts.set(token, (queryOverlapCounts.get(token) || 0) + 1);
                        return;
                    }
                    aggregatedTermCounts.set(token, (aggregatedTermCounts.get(token) || 0) + 1);
                });

                if (!groupsByDocument.has(groupKey)) {
                    groupsByDocument.set(groupKey, {
                        key: groupKey,
                        title: title,
                        date: date,
                        doc_id: docId,
                        chunk_count: 0,
                        best_score: score,
                        top_snippet: snippet,
                        overlap_counts: new Map(),
                    });
                }

                const group = groupsByDocument.get(groupKey);
                group.chunk_count += 1;
                if (score > group.best_score) {
                    group.best_score = score;
                    group.top_snippet = snippet;
                }
                tokens.forEach(function (token) {
                    if (queryTokens.has(token)) {
                        group.overlap_counts.set(token, (group.overlap_counts.get(token) || 0) + 1);
                    }
                });
            });

            const total = scoreValues.length;
            const average = total
                ? scoreValues.reduce(function (sum, value) { return sum + value; }, 0) / total
                : 0;
            const best = total ? Math.max.apply(null, scoreValues) : 0;
            const lowest = total ? Math.min.apply(null, scoreValues) : 0;

            const topTerms = tokenCountToTopList(aggregatedTermCounts, 8);
            const overlapTerms = tokenCountToTopList(queryOverlapCounts, 6);
            const groupedDocuments = Array.from(groupsByDocument.values())
                .map(function (group) {
                    const overlaps = tokenCountToTopList(group.overlap_counts, 4);
                    return {
                        key: group.key,
                        title: group.title,
                        date: group.date,
                        doc_id: group.doc_id,
                        chunk_count: group.chunk_count,
                        best_score: group.best_score,
                        top_snippet: group.top_snippet,
                        overlaps: overlaps,
                    };
                })
                .sort(function (a, b) {
                    if (b.best_score !== a.best_score) {
                        return b.best_score - a.best_score;
                    }
                    return b.chunk_count - a.chunk_count;
                });

            return {
                total_neighbors: total,
                unique_documents: groupedDocuments.length,
                average_score: average,
                best_score: best,
                lowest_score: lowest,
                strong_count: strongCount,
                medium_count: mediumCount,
                broad_count: broadCount,
                top_terms: topTerms,
                overlap_terms: overlapTerms,
                grouped_documents: groupedDocuments,
            };
        }

        function updateMetaText() {
            const modeLabel = activeView === 'meaning' ? 'Meaning view' : 'Neighbor view';
            meta.textContent = metaPrefix ? (metaPrefix + ' - ' + modeLabel) : modeLabel;
        }

        function setView(viewName) {
            activeView = viewName === 'meaning' ? 'meaning' : 'neighbors';
            const neighborsActive = activeView === 'neighbors';

            resultsContainer.hidden = !neighborsActive;
            meaningContainer.hidden = neighborsActive;

            neighborsViewButton.classList.toggle('is-active', neighborsActive);
            neighborsViewButton.setAttribute('aria-selected', neighborsActive ? 'true' : 'false');
            meaningViewButton.classList.toggle('is-active', !neighborsActive);
            meaningViewButton.setAttribute('aria-selected', neighborsActive ? 'false' : 'true');
            updateMetaText();
        }

        function renderNeighbors(results) {
            resultsContainer.innerHTML = '';

            if (!Array.isArray(results) || results.length === 0) {
                const empty = document.createElement('p');
                empty.className = 'semantic-results__empty';
                empty.textContent = 'No semantic matches found for this query.';
                empty.setAttribute('data-help', 'No nearest-neighbor chunks were returned for this query.');
                resultsContainer.appendChild(empty);
                attachTooltips(empty);
                return;
            }

            results.forEach(function (result, index) {
                const card = document.createElement('article');
                card.className = 'result-card';
                card.setAttribute('data-help', 'Neighbor #' + String(index + 1) + '. This chunk is one of the closest vectors to your query.');

                const rank = document.createElement('div');
                rank.className = 'result-card__rank';
                rank.textContent = '#' + String(index + 1);
                rank.setAttribute('data-help', 'Rank position among nearest semantic neighbors.');
                card.appendChild(rank);

                const scoreWrap = document.createElement('div');
                scoreWrap.className = 'result-card__score';
                scoreWrap.setAttribute('data-help', 'Similarity score and raw vector distance for this neighbor.');

                const scoreBadge = document.createElement('span');
                scoreBadge.className = 'score-badge';
                scoreBadge.textContent = formatScore(result && result.score);
                scoreBadge.setAttribute('data-help', 'Similarity score. Higher values indicate closer semantic proximity.');
                scoreWrap.appendChild(scoreBadge);

                const distance = document.createElement('small');
                distance.className = 'score-distance';
                distance.textContent = 'distance ' + formatScore(result && result.distance);
                distance.setAttribute('data-help', 'Raw distance value returned by the vector store.');
                scoreWrap.appendChild(distance);
                card.appendChild(scoreWrap);

                const body = document.createElement('div');
                body.className = 'result-card__body';

                const heading = document.createElement('h3');
                heading.textContent = safeText(result && result.title) || 'Untitled';
                heading.setAttribute('data-help', 'Document title attached to this neighbor chunk.');
                const date = document.createElement('span');
                date.className = 'result-date';
                date.textContent = safeText(result && result.date);
                date.setAttribute('data-help', 'Document date metadata when available.');
                heading.appendChild(document.createTextNode(' '));
                heading.appendChild(date);
                body.appendChild(heading);

                const snippet = document.createElement('p');
                snippet.className = 'result-snippet';
                snippet.textContent = safeText(result && result.content);
                snippet.setAttribute('data-help', 'Chunk snippet used as evidence for semantic similarity.');
                body.appendChild(snippet);

                if (result && result.doc_id) {
                    const link = document.createElement('a');
                    link.className = 'result-link';
                    link.href = '/document/' + encodeURIComponent(String(result.doc_id));
                    link.textContent = 'Open document ->';
                    link.setAttribute('data-help', 'Open the source document for this neighbor.');
                    body.appendChild(link);
                }

                card.appendChild(body);
                resultsContainer.appendChild(card);
                attachTooltips(card);
            });
        }

        function renderMeaning(model) {
            meaningContainer.innerHTML = '';

            if (!model || !model.total_neighbors) {
                const empty = document.createElement('p');
                empty.className = 'semantic-results__empty';
                empty.textContent = 'Meaning view will appear once neighbors are available.';
                empty.setAttribute('data-help', 'Run a semantic query to build grouped interpretation signals.');
                meaningContainer.appendChild(empty);
                attachTooltips(empty);
                return;
            }

            const summary = document.createElement('section');
            summary.className = 'semantic-meaning__summary';
            summary.setAttribute('data-help', 'High-level interpretation signals derived from the nearest-neighbor set.');
            summary.innerHTML = [
                '<div class="semantic-meaning__stat" data-help="Total chunk neighbors returned by semantic retrieval."><span>Neighbors</span><strong>' + String(model.total_neighbors) + '</strong></div>',
                '<div class="semantic-meaning__stat" data-help="Unique source documents represented by returned neighbors."><span>Documents</span><strong>' + String(model.unique_documents) + '</strong></div>',
                '<div class="semantic-meaning__stat" data-help="Average similarity score across all returned neighbors."><span>Avg score</span><strong>' + formatScore(model.average_score) + '</strong></div>',
                '<div class="semantic-meaning__stat" data-help="Highest similarity score in the current neighbor set."><span>Best score</span><strong>' + formatScore(model.best_score) + '</strong></div>'
            ].join('');
            meaningContainer.appendChild(summary);

            const scoreBands = document.createElement('p');
            scoreBands.className = 'semantic-meaning__bands';
            scoreBands.setAttribute('data-help', 'Simple score bands to show how concentrated the neighbor set is.');
            scoreBands.textContent = 'Score bands: ' +
                String(model.strong_count) + ' strong, ' +
                String(model.medium_count) + ' medium, ' +
                String(model.broad_count) + ' broad matches.';
            meaningContainer.appendChild(scoreBands);

            const interpretation = document.createElement('p');
            interpretation.className = 'semantic-meaning__interpretation';
            interpretation.setAttribute('data-help', 'Narrative interpretation of recurring signals in nearby chunks.');
            if (model.top_terms.length) {
                interpretation.textContent = 'Nearest neighbors repeatedly reference: ' +
                    model.top_terms.slice(0, 5).map(function (item) { return item.term; }).join(', ') + '.';
            } else {
                interpretation.textContent = 'Nearest neighbors are present, but recurring non-query terms are sparse.';
            }
            meaningContainer.appendChild(interpretation);

            const termRow = document.createElement('div');
            termRow.className = 'semantic-meaning__terms';
            termRow.setAttribute('data-help', 'Frequent non-query terms across top semantic neighbors.');
            model.top_terms.forEach(function (item) {
                const chip = document.createElement('span');
                chip.className = 'semantic-term-chip';
                chip.textContent = item.term + ' (' + String(item.count) + ')';
                chip.setAttribute('data-help', 'Recurring term "' + item.term + '" appears in ' + String(item.count) + ' neighbor snippets.');
                termRow.appendChild(chip);
            });
            if (!model.top_terms.length) {
                const none = document.createElement('span');
                none.className = 'semantic-term-chip semantic-term-chip--muted';
                none.textContent = 'No stable recurring terms';
                none.setAttribute('data-help', 'No clear recurring vocabulary emerged from this neighbor set.');
                termRow.appendChild(none);
            }
            meaningContainer.appendChild(termRow);

            const docs = document.createElement('section');
            docs.className = 'semantic-meaning__docs';
            docs.setAttribute('data-help', 'Neighbors grouped by source document to show where meaning clusters.');
            const heading = document.createElement('h3');
            heading.textContent = 'Document clusters';
            docs.appendChild(heading);

            const list = document.createElement('div');
            list.className = 'semantic-meaning__doc-list';
            model.grouped_documents.slice(0, 12).forEach(function (group, index) {
                const card = document.createElement('article');
                card.className = 'semantic-doc-card';
                card.setAttribute('data-help', 'Grouped document signal #' + String(index + 1) + ' with best score and chunk coverage.');

                const topLine = document.createElement('div');
                topLine.className = 'semantic-doc-card__top';
                const titleStrong = document.createElement('strong');
                titleStrong.textContent = group.title || 'Untitled';
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'semantic-doc-card__score';
                scoreSpan.textContent = 'best ' + formatScore(group.best_score);
                topLine.appendChild(titleStrong);
                topLine.appendChild(scoreSpan);
                card.appendChild(topLine);

                const metaLine = document.createElement('p');
                metaLine.className = 'semantic-doc-card__meta';
                metaLine.textContent = String(group.chunk_count) + ' chunk(s) in neighbor set' +
                    (group.date ? ' • ' + group.date : '') +
                    (group.overlaps.length ? ' • overlap: ' + group.overlaps.map(function (entry) { return entry.term; }).join(', ') : '');
                card.appendChild(metaLine);

                const snippet = document.createElement('p');
                snippet.className = 'semantic-doc-card__snippet';
                snippet.textContent = group.top_snippet || '';
                card.appendChild(snippet);

                if (group.doc_id) {
                    const link = document.createElement('a');
                    link.className = 'result-link';
                    link.href = '/document/' + encodeURIComponent(group.doc_id);
                    link.textContent = 'Open document ->';
                    link.setAttribute('data-help', 'Open this clustered source document.');
                    card.appendChild(link);
                }

                list.appendChild(card);
                attachTooltips(card);
            });
            docs.appendChild(list);
            meaningContainer.appendChild(docs);
            attachTooltips(meaningContainer);
        }

        neighborsViewButton.addEventListener('click', function () {
            setView('neighbors');
        });

        meaningViewButton.addEventListener('click', function () {
            setView('meaning');
        });

        form.addEventListener('submit', async function (event) {
            event.preventDefault();
            const query = queryInput.value.trim();
            const topK = parseTopK(topKInput.value);
            if (!query) {
                queryInput.focus();
                return;
            }

            searchButton.disabled = true;
            panel.hidden = true;
            resultsContainer.innerHTML = '';
            meaningContainer.innerHTML = '';
            metaPrefix = '';
            updateMetaText();

            try {
                const response = await fetch('/api/semantic-search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topK }),
                });

                const data = await response.json();
                if (!response.ok || (data && data.error)) {
                    throw new Error((data && data.error) || 'Semantic search failed.');
                }

                const results = Array.isArray(data.results) ? data.results : [];
                metaPrefix = 'Query: "' + query + '" - ' + String(results.length) + ' results';
                panel.hidden = false;
                renderNeighbors(results);
                renderMeaning(buildMeaningModel(query, results));
                setView('neighbors');
            } catch (error) {
                window.alert('Search failed: ' + (error && error.message ? error.message : 'Unknown error'));
            } finally {
                searchButton.disabled = false;
            }
        });

        clearButton.addEventListener('click', function () {
            queryInput.value = '';
            topKInput.value = '20';
            resultsContainer.innerHTML = '';
            meaningContainer.innerHTML = '';
            panel.hidden = true;
            metaPrefix = '';
            setView('neighbors');
            queryInput.focus();
        });
        ensureNativeHelp(document);
        attachTooltips(document);
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSemanticSearch);
    } else {
        initSemanticSearch();
    }
})();
