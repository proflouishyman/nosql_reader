/**
 * network_statistics.js
 *
 * Renders the ERGM-inspired statistical analysis panel at the bottom of
 * /network-analysis, BELOW the existing client-side analytics panel.
 *
 * This panel provides server-side significance tests that complement
 * (not duplicate) the descriptive metrics already computed in network.js.
 *
 * Domain-agnostic: all labels, categories, attribute names come from
 * the API responses. No corpus-specific strings.
 *
 * Dependencies:
 *   - D3.js v7 (already loaded by network-analysis.html)
 *
 * Auto-initializes when #statistics-panel exists in the DOM.
 */

const NetworkStatistics = (() => {

    // Significance markers
    function sigMarker(p) {
        if (p === null || p === undefined) return "\u2014";
        if (p < 0.001) return "\u2726\u2726\u2726";
        if (p < 0.01)  return "\u2726\u2726";
        if (p < 0.05)  return "\u2726";
        return "\u2014";
    }

    function fmt(v, d) {
        d = d || 4;
        if (v === null || v === undefined) return "\u2014";
        return typeof v === "number" ? v.toFixed(d) : String(v);
    }

    // ======================================================================
    // Init
    // ======================================================================

    async function init() {
        var panel = document.getElementById("statistics-panel");
        if (!panel) return;

        var computeBtn = document.getElementById("compute-statistics-btn");
        var resultsContainer = document.getElementById("statistics-results");

        if (computeBtn) {
            computeBtn.addEventListener("click", async function() {
                computeBtn.disabled = true;
                computeBtn.textContent = "Computing\u2026";
                resultsContainer.innerHTML =
                    '<div class="stats-loading">' +
                    '<div class="stats-spinner"></div>' +
                    '<p>Running statistical analysis. This may take 30\u201360 seconds.</p>' +
                    '<p class="stats-loading-detail">Permutation tests, community detection, random graph comparison\u2026</p>' +
                    '</div>';

                try {
                    await loadAndRender(resultsContainer);
                } catch (e) {
                    resultsContainer.innerHTML = '<div class="stats-error">Analysis failed: ' + e.message + '</div>';
                }

                computeBtn.textContent = "Recompute";
                computeBtn.disabled = false;
            });
        }
    }

    async function loadAndRender(container) {
        var params = readFilterParams();
        var url = "/api/network/statistics/summary?min_weight=" + params.min_weight +
            "&person_min_mentions=" + params.person_min_mentions +
            "&n_permutations=1000&n_simulations=100";
        if (params.type_filter) url += "&type_filter=" + params.type_filter;

        var resp = await fetch(url);
        if (!resp.ok) {
            var err = {};
            try { err = await resp.json(); } catch(_) {}
            throw new Error(err.error || "HTTP " + resp.status);
        }
        var data = await resp.json();

        container.innerHTML = "";

        if (data.from_cache) {
            var note = document.createElement("p");
            note.className = "stats-cache-note";
            note.textContent = "Results from cache (computed " + (data.computed_at || "recently") + "). Click Recompute to refresh.";
            container.appendChild(note);
        }

        renderGraphSummary(container, data.graph_summary);
        renderAssortativity(container, data.assortativity);
        renderMixingMatrixSelector(container);
        renderDegreeDistribution(container, data.degree_distribution);
        renderCommunities(container, data.communities);
        renderComparison(container, data.comparison_to_random);
        renderGatekeepers(container, data.gatekeepers);
    }

    /**
     * Read current filter values from the existing network-analysis controls.
     * This coordinates with the controls already rendered by network.js.
     */
    function readFilterParams() {
        var params = { min_weight: 3, person_min_mentions: 3 };

        // Type filter checkboxes (existing on page)
        var typeChecked = document.querySelectorAll('input[type="checkbox"][data-entity-type]:checked, .type-filter-checkbox:checked');
        if (typeChecked.length > 0) {
            params.type_filter = Array.from(typeChecked).map(function(cb) {
                return cb.value || cb.dataset.entityType;
            }).filter(Boolean).join(",");
        }

        // Min weight slider (existing on page)
        var mwSlider = document.getElementById("min-weight-slider") ||
                       document.querySelector('[data-control="min-weight"]');
        if (mwSlider) params.min_weight = parseInt(mwSlider.value, 10) || 3;

        // Person min mentions (existing on page)
        var pmmSlider = document.getElementById("person-min-mentions") ||
                        document.querySelector('[data-control="person-min-mentions"]');
        if (pmmSlider) params.person_min_mentions = parseInt(pmmSlider.value, 10) || 3;

        return params;
    }

    // ======================================================================
    // 1. Graph summary
    // ======================================================================

    function renderGraphSummary(container, data) {
        if (!data) return;
        var section = makeSection(container, "Graph Summary (Server-Side)", "graph-summary");
        section.innerHTML =
            '<div class="stats-metric-row">' +
            '<span class="stats-metric"><strong>' + (data.nodes || 0).toLocaleString() + '</strong> nodes</span>' +
            '<span class="stats-metric"><strong>' + (data.edges || 0).toLocaleString() + '</strong> edges</span>' +
            '<span class="stats-metric">density <strong>' + fmt(data.density, 5) + '</strong></span>' +
            '<span class="stats-metric"><strong>' + (data.components || 0) + '</strong> components</span>' +
            '<span class="stats-metric">largest <strong>' + (data.largest_component_size || 0).toLocaleString() +
            '</strong> (' + ((data.largest_component_fraction || 0) * 100).toFixed(1) + '%)</span>' +
            '</div>' +
            '<p class="stats-note">Computed on full network_edges collection with current filters. ' +
            'Client-side panel above shows the loaded graph slice.</p>';
    }

    // ======================================================================
    // 2. Assortativity
    // ======================================================================

    function renderAssortativity(container, data) {
        if (!data || !Array.isArray(data) || data.length === 0) return;
        var section = makeSection(container, "Assortativity Tests (ERGM nodematch)", "assortativity");

        var valid = data.filter(function(r) { return !r.error; });
        var errors = data.filter(function(r) { return r.error; });

        if (valid.length === 0) {
            section.innerHTML = "<p>No attributes available for assortativity testing.</p>";
            return;
        }

        var html =
            '<p class="stats-description">' +
            'Measures whether entities sharing an attribute connect more (+) or less (\u2212) than random chance. ' +
            'Significance via ' + (valid[0].n_permutations || 1000) + ' permutation trials. ' +
            'Equivalent to ERGM <code>nodematch()</code> term.' +
            '</p>' +
            '<table class="stats-table"><thead><tr>' +
            '<th>Attribute</th><th>Categories</th><th>Coefficient</th>' +
            '<th>p-value</th><th>Sig.</th><th>z-score</th><th>Interpretation</th>' +
            '</tr></thead><tbody>';

        valid.forEach(function(r) {
            var cls = r.significant ? "stats-significant" : "";
            var interp = {
                significant_homophily: "Homophily \u25b2",
                significant_heterophily: "Heterophily \u25bc",
                not_significant: "Random"
            }[r.interpretation] || r.interpretation;

            html += '<tr class="' + cls + '">' +
                '<td><strong>' + r.attribute + '</strong></td>' +
                '<td>' + (r.n_categories || "?") + '</td>' +
                '<td class="stats-num">' + fmt(r.observed) + '</td>' +
                '<td class="stats-num">' + fmt(r.p_value) + '</td>' +
                '<td class="stats-sig">' + sigMarker(r.p_value) + '</td>' +
                '<td class="stats-num">' + fmt(r.z_score, 2) + '</td>' +
                '<td>' + interp + '</td></tr>';
        });

        html += '</tbody></table>';

        if (errors.length > 0) {
            html += '<p class="stats-note">' + errors.length + ' attribute(s) skipped: ' +
                errors.map(function(r) { return r.attribute + " (" + r.error + ")"; }).join("; ") + '</p>';
        }

        section.innerHTML = html;
    }

    // ======================================================================
    // 3. Mixing matrix (on-demand per attribute)
    // ======================================================================

    function renderMixingMatrixSelector(container) {
        var section = makeSection(container, "Mixing Matrix (ERGM nodemix)", "mixing-matrix");
        section.innerHTML =
            '<p class="stats-description">' +
            'Cross-category pairings: which are over- or under-represented vs. random expectation? ' +
            '<span style="color:#b2182b">\u25a0</span> overrepresented, ' +
            '<span style="color:#2166ac">\u25a0</span> underrepresented. ' +
            'Equivalent to ERGM <code>nodemix()</code> term.' +
            '</p>' +
            '<div class="stats-control-row">' +
            '<label for="mixing-attr-select">Attribute:</label>' +
            '<select id="mixing-attr-select"><option value="">Loading\u2026</option></select>' +
            '<button id="mixing-compute-btn" class="stats-btn-small">Show Matrix</button>' +
            '</div>' +
            '<div id="mixing-matrix-output"></div>';

        fetch("/api/network/statistics/available_attributes")
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var select = document.getElementById("mixing-attr-select");
                if (!data.all_attributes || data.all_attributes.length === 0) {
                    select.innerHTML = '<option value="">No attributes available</option>';
                    return;
                }
                select.innerHTML = data.all_attributes
                    .map(function(a) { return '<option value="' + a + '">' + a + '</option>'; })
                    .join("");
            })
            .catch(function() {});

        var btn = document.getElementById("mixing-compute-btn");
        if (btn) {
            btn.addEventListener("click", async function() {
                var attr = document.getElementById("mixing-attr-select").value;
                if (!attr) return;
                var output = document.getElementById("mixing-matrix-output");
                output.innerHTML = '<div class="stats-loading-inline">Computing\u2026</div>';

                var params = readFilterParams();
                var url = "/api/network/statistics/mixing_matrix?attribute=" + attr +
                    "&min_weight=" + params.min_weight +
                    "&person_min_mentions=" + params.person_min_mentions;
                if (params.type_filter) url += "&type_filter=" + params.type_filter;

                try {
                    var resp = await fetch(url);
                    var data = await resp.json();
                    if (data.error) {
                        output.innerHTML = '<p class="stats-note">' + data.error + '</p>';
                        return;
                    }
                    renderHeatmap(output, data);
                } catch (e) {
                    output.innerHTML = '<p class="stats-error">' + e.message + '</p>';
                }
            });
        }
    }

    function renderHeatmap(container, data) {
        container.innerHTML = "";
        var categories = data.categories;
        var residuals = data.residuals;
        var notable = data.notable_pairs;
        if (!categories || !residuals) return;

        var n = categories.length;
        var cellSize = Math.min(50, Math.floor(500 / n));
        var margin = { top: 100, right: 20, bottom: 20, left: 130 };
        var width = margin.left + n * cellSize + margin.right;
        var height = margin.top + n * cellSize + margin.bottom;

        var svg = d3.select(container).append("svg")
            .attr("width", width).attr("height", height);

        var maxAbs = Math.max.apply(null,
            residuals.reduce(function(a, r) { return a.concat(r); }, [])
                .map(Math.abs).filter(isFinite).concat([2])
        );
        var colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([maxAbs, -maxAbs]);

        for (var i = 0; i < n; i++) {
            for (var j = 0; j < n; j++) {
                var val = residuals[i][j];
                svg.append("rect")
                    .attr("x", margin.left + j * cellSize)
                    .attr("y", margin.top + i * cellSize)
                    .attr("width", cellSize - 1)
                    .attr("height", cellSize - 1)
                    .attr("fill", isFinite(val) ? colorScale(val) : "#eee")
                    .attr("rx", 2)
                    .append("title")
                    .text(categories[i] + " \u00d7 " + categories[j] + ": residual " + fmt(val, 2));
            }
        }

        // Row labels
        categories.forEach(function(cat, i) {
            var label = cat.length > 16 ? cat.slice(0, 15) + "\u2026" : cat;
            svg.append("text")
                .attr("x", margin.left - 6)
                .attr("y", margin.top + i * cellSize + cellSize / 2 + 4)
                .attr("text-anchor", "end")
                .attr("font-size", "11px")
                .text(label);
        });

        // Column labels (rotated)
        categories.forEach(function(cat, j) {
            var label = cat.length > 16 ? cat.slice(0, 15) + "\u2026" : cat;
            svg.append("text")
                .attr("x", 0).attr("y", 0)
                .attr("transform", "translate(" + (margin.left + j * cellSize + cellSize / 2) + "," + (margin.top - 6) + ") rotate(-45)")
                .attr("text-anchor", "start")
                .attr("font-size", "11px")
                .text(label);
        });

        // Notable pairs callout
        if (notable && notable.length > 0) {
            var pairsHtml = '<div class="stats-notable-pairs"><strong>Notable pairings:</strong><ul>';
            notable.slice(0, 8).forEach(function(p) {
                var icon = p.direction === "overrepresented" ? "\u25b2" : "\u25bc";
                var color = p.direction === "overrepresented" ? "#b2182b" : "#2166ac";
                pairsHtml += '<li><span style="color:' + color + '">' + icon + '</span> ' +
                    p.category_a + ' \u00d7 ' + p.category_b +
                    ': residual ' + p.residual + ' (obs=' + p.observed + ', exp=' + p.expected + ')</li>';
            });
            pairsHtml += '</ul></div>';
            container.insertAdjacentHTML("beforeend", pairsHtml);
        }
    }

    // ======================================================================
    // 4. Degree distribution
    // ======================================================================

    function renderDegreeDistribution(container, data) {
        if (!data || data.error) return;
        var section = makeSection(container, "Degree Distribution (ERGM degree)", "degree-dist");

        var stats = data.stats;
        var html =
            '<div class="stats-metric-row">' +
            '<span class="stats-metric">range <strong>' + stats.min + '\u2013' + stats.max + '</strong></span>' +
            '<span class="stats-metric">mean <strong>' + stats.mean + '</strong></span>' +
            '<span class="stats-metric">median <strong>' + stats.median + '</strong></span>' +
            '<span class="stats-metric">Gini <strong>' + fmt(data.gini_coefficient) + '</strong> ' +
            '<span class="stats-badge stats-badge-' + data.gini_interpretation + '">' +
            data.gini_interpretation.replace(/_/g, " ") + '</span></span>' +
            '</div>';

        if (data.powerlaw_fit && data.powerlaw_fit.test !== "skipped") {
            html += '<p class="stats-note">Power-law fit: \u03b1=' + (data.powerlaw_fit.alpha_estimate || "?") +
                ', KS p=' + fmt(data.powerlaw_fit.ks_p_value) +
                (data.powerlaw_fit.plausible_powerlaw ? " \u2014 plausible power-law" : " \u2014 not power-law") + '</p>';
        }

        html += '<div id="degree-histogram" style="width:100%;height:200px;"></div>';

        if (data.hubs && data.hubs.length > 0) {
            html += '<h4>Hub Entities (degree \u2265 ' + data.hub_cutoff + ')</h4>' +
                '<table class="stats-table stats-table-compact"><thead><tr>' +
                '<th>Entity</th><th>Type</th><th>Degree</th><th>Weighted</th></tr></thead><tbody>';
            data.hubs.slice(0, 15).forEach(function(h) {
                html += '<tr><td><a href="/network-analysis?entity=' + h.entity_id + '">' + h.name + '</a></td>' +
                    '<td>' + h.type + '</td>' +
                    '<td class="stats-num">' + h.degree + '</td>' +
                    '<td class="stats-num">' + h.weighted_degree + '</td></tr>';
            });
            html += '</tbody></table>';
        }

        section.innerHTML = html;

        if (data.histogram && typeof d3 !== "undefined") {
            drawHistogram(section.querySelector("#degree-histogram"), data.histogram);
        }
    }

    function drawHistogram(el, histogram) {
        if (!el) return;
        var entries = Object.keys(histogram).map(function(k) {
            return { degree: +k, count: histogram[k] };
        }).sort(function(a, b) { return a.degree - b.degree; });

        var margin = { top: 10, right: 20, bottom: 35, left: 50 };
        var width = (el.clientWidth || 600) - margin.left - margin.right;
        var height = 180 - margin.top - margin.bottom;

        var svg = d3.select(el).append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var capped = entries.filter(function(e) { return e.degree <= 50; });

        var x = d3.scaleBand().domain(capped.map(function(d) { return d.degree; })).range([0, width]).padding(0.15);
        var y = d3.scaleLinear().domain([0, d3.max(capped, function(d) { return d.count; })]).nice().range([height, 0]);

        svg.append("g").attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x).tickValues(x.domain().filter(function(_, i) { return i % 5 === 0; })))
            .selectAll("text").attr("font-size", "10px");

        svg.append("g").call(d3.axisLeft(y).ticks(5)).selectAll("text").attr("font-size", "10px");

        svg.selectAll(".bar").data(capped).join("rect")
            .attr("x", function(d) { return x(d.degree); })
            .attr("y", function(d) { return y(d.count); })
            .attr("width", x.bandwidth())
            .attr("height", function(d) { return height - y(d.count); })
            .attr("fill", "#2E75B6")
            .attr("rx", 1)
            .append("title")
            .text(function(d) { return "Degree " + d.degree + ": " + d.count + " nodes"; });

        svg.append("text").attr("x", width / 2).attr("y", height + 30)
            .attr("text-anchor", "middle").attr("font-size", "11px").text("Degree");
        svg.append("text").attr("transform", "rotate(-90)")
            .attr("x", -height / 2).attr("y", -35)
            .attr("text-anchor", "middle").attr("font-size", "11px").text("Count");
    }

    // ======================================================================
    // 5. Community structure
    // ======================================================================

    function renderCommunities(container, data) {
        if (!data || data.error) return;
        var section = makeSection(container, "Community Structure", "communities");

        var html =
            '<div class="stats-metric-row">' +
            '<span class="stats-metric"><strong>' + data.n_communities + '</strong> communities</span>' +
            '<span class="stats-metric">modularity <strong>' + fmt(data.modularity) + '</strong></span>' +
            '<span class="stats-metric">method: ' + (data.method || "?") + '</span>' +
            '</div>';

        // NMI scores
        if (data.nmi_scores && Object.keys(data.nmi_scores).length > 0) {
            html += '<p class="stats-description">Normalized Mutual Information \u2014 ' +
                'how well each attribute predicts community membership (0\u00a0=\u00a0independent, 1\u00a0=\u00a0perfect):</p>' +
                '<div class="stats-nmi-row">';
            Object.keys(data.nmi_scores)
                .sort(function(a, b) { return data.nmi_scores[b] - data.nmi_scores[a]; })
                .forEach(function(attr) {
                    var nmi = data.nmi_scores[attr];
                    var barW = Math.max(nmi * 200, 2);
                    html += '<div class="stats-nmi-item">' +
                        '<span class="stats-nmi-label">' + attr + '</span>' +
                        '<div class="stats-nmi-bar" style="width:' + barW + 'px"></div>' +
                        '<span class="stats-nmi-value">' + fmt(nmi) + '</span></div>';
                });
            html += '</div>';
        }

        // Top communities
        if (data.communities && data.communities.length > 0) {
            html += '<h4>Largest Communities</h4>';
            data.communities.slice(0, 8).forEach(function(c) {
                html += '<div class="stats-community-card">' +
                    '<strong>Community ' + c.community_id + '</strong>: ' +
                    c.size + ' nodes, ' + c.internal_edges + ' internal edges' +
                    '<div class="stats-community-members">Top: ' +
                    c.top_members.map(function(m) {
                        return '<a href="/network-analysis?entity=' + m.entity_id + '">' + m.name + '</a>';
                    }).join(", ") + '</div>';
                if (c.composition) {
                    Object.keys(c.composition).forEach(function(attr) {
                        var top3 = Object.keys(c.composition[attr]).slice(0, 3)
                            .map(function(k) { return k + " (" + c.composition[attr][k] + ")"; }).join(", ");
                        html += '<div class="stats-community-comp">' + attr + ': ' + top3 + '</div>';
                    });
                }
                html += '</div>';
            });
        }

        section.innerHTML = html;
    }

    // ======================================================================
    // 6. Observed vs. random comparison
    // ======================================================================

    function renderComparison(container, data) {
        if (!data || data.error) return;
        var section = makeSection(container, "Observed vs. Random Networks (ERGM simulate/gof)", "comparison");

        var html =
            '<p class="stats-description">' +
            'Compares the observed network to random graphs with the same degree sequence. ' +
            'The client-side panel shows clustering and modularity values; this test shows ' +
            'whether those values are <em>significantly different</em> from random expectation.' +
            '</p>' +
            '<table class="stats-table"><thead><tr>' +
            '<th>Metric</th><th>Observed</th><th>Random (mean \u00b1 std)</th>' +
            '<th>z-score</th><th>Sig.</th><th>Direction</th></tr></thead><tbody>';

        ["clustering", "modularity", "avg_path_length"].forEach(function(key) {
            var m = data[key];
            if (!m || m.error) return;
            var cls = m.significant ? "stats-significant" : "";
            html += '<tr class="' + cls + '">' +
                '<td><strong>' + (m.metric || key).replace(/_/g, " ") + '</strong></td>' +
                '<td class="stats-num">' + fmt(m.observed) + '</td>' +
                '<td class="stats-num">' + fmt(m.null_mean) + ' \u00b1' + fmt(m.null_std) + '</td>' +
                '<td class="stats-num">' + fmt(m.z_score, 2) + '</td>' +
                '<td class="stats-sig">' + (m.significant ? "\u2726" : "\u2014") + '</td>' +
                '<td>' + (m.direction || "\u2014") + ' than random</td></tr>';
        });

        html += '</tbody></table>';
        section.innerHTML = html;
    }

    // ======================================================================
    // 7. Gatekeepers
    // ======================================================================

    function renderGatekeepers(container, data) {
        if (!data || data.error || !data.gatekeepers || data.gatekeepers.length === 0) return;
        var section = makeSection(container, "Gatekeeper Entities", "gatekeepers");

        var html =
            '<p class="stats-description">' +
            'Entities that bridge otherwise separate communities. Bridge score = betweenness\u00a0/\u00a0log(degree+1). ' +
            'High bridge scores identify structural brokers, not just high-degree hubs. ' +
            'Compare with the raw betweenness ranking in the client-side panel above.' +
            '</p>' +
            '<table class="stats-table"><thead><tr>' +
            '<th>Entity</th><th>Type</th><th>Degree</th><th>Betweenness</th><th>Bridge Score</th>' +
            '</tr></thead><tbody>';

        data.gatekeepers.slice(0, 20).forEach(function(g) {
            html += '<tr>' +
                '<td><a href="/network-analysis?entity=' + g.entity_id + '">' + g.name + '</a></td>' +
                '<td>' + g.type + '</td>' +
                '<td class="stats-num">' + g.degree + '</td>' +
                '<td class="stats-num">' + fmt(g.betweenness_centrality, 5) + '</td>' +
                '<td class="stats-num">' + fmt(g.bridge_score, 5) + '</td></tr>';
        });

        html += '</tbody></table>';
        section.innerHTML = html;
    }

    // ======================================================================
    // Utility
    // ======================================================================

    function makeSection(container, title, id) {
        var section = document.createElement("div");
        section.className = "stats-section";
        section.id = "stats-" + id;
        section.innerHTML = '<h3 class="stats-section-title">' + title + '</h3>';
        container.appendChild(section);
        return section;
    }

    return { init: init };
})();

document.addEventListener("DOMContentLoaded", function() {
    NetworkStatistics.init();
});
