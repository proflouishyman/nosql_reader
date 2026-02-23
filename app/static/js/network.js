/**
 * network.js — Network visualization and entity interaction logic.
 *
 * Domain-agnostic: entity types and colors are derived from the data itself,
 * not hard-coded. The color palette assigns colors dynamically based on
 * whatever types the API returns.
 *
 * Dependencies:
 *   - D3.js v7 (loaded via CDN in templates)
 *
 * Usage:
 *   Document detail page: include this script, it auto-initializes popup
 *   handlers and the context panel if the relevant DOM elements exist.
 *
 *   Network explorer page: call NetworkExplorer.init(containerSelector).
 */

// ===========================================================================
// Color palette — maps entity types to colors dynamically
// ===========================================================================
const NetworkColors = (() => {
    // Accessible color palette (colorblind-friendly, high contrast)
    const PALETTE = [
        "#2E75B6", // blue
        "#C0392B", // red
        "#27AE60", // green
        "#8E44AD", // purple
        "#E67E22", // orange
        "#16A085", // teal
        "#D4AC0D", // gold
        "#7F8C8D", // gray
    ];

    const typeColorMap = {};
    let nextIndex = 0;

    return {
        /**
         * Get the color for an entity type. Assigns colors on first encounter.
         * @param {string} type — Entity type string (e.g., "PERSON", "GPE")
         * @returns {string} Hex color
         */
        get(type) {
            if (!typeColorMap[type]) {
                typeColorMap[type] = PALETTE[nextIndex % PALETTE.length];
                nextIndex++;
            }
            return typeColorMap[type];
        },

        /** Get all assigned type→color mappings (for legend rendering). */
        getAll() {
            return { ...typeColorMap };
        },
    };
})();


// ===========================================================================
// API client — thin wrapper around fetch for network endpoints
// ===========================================================================
const NetworkAPI = {
    baseUrl: "/api/network",

    async _fetch(path, params = {}) {
        const url = new URL(this.baseUrl + path, window.location.origin);
        Object.entries(params).forEach(([k, v]) => {
            if (v !== undefined && v !== null && v !== "") {
                url.searchParams.set(k, v);
            }
        });

        try {
            const resp = await fetch(url.toString());
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                console.warn(`Network API ${path} returned ${resp.status}:`, err);
                return null;
            }
            return await resp.json();
        } catch (e) {
            console.error(`Network API ${path} failed:`, e);
            return null;
        }
    },

    /** Ego network for one entity */
    getEgoNetwork(entityId, opts = {}) {
        return this._fetch(`/entity/${entityId}`, opts);
    },

    /** Document context graph */
    getDocumentNetwork(docId, opts = {}) {
        return this._fetch(`/document/${docId}`, opts);
    },

    /** Filtered global network */
    getGlobalNetwork(opts = {}) {
        return this._fetch("/global", opts);
    },

    /** Entity metrics */
    getEntityMetrics(entityId) {
        return this._fetch(`/metrics/${entityId}`);
    },

    /** Related documents */
    getRelatedDocuments(docId, opts = {}) {
        return this._fetch(`/related/${docId}`, opts);
    },

    /** Network stats */
    getStats() {
        return this._fetch("/stats");
    },

    /** Available entity types */
    getTypes() {
        return this._fetch("/types");
    },
};


const HelpTooltips = window.AppHelpTooltips || {
    attach() {
        return null;
    },
    hide() {
        return null;
    },
};


function escapeHtmlAttribute(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("\"", "&quot;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}


// ===========================================================================
// Entity popup — hover/click interaction on document detail page
// ===========================================================================
const EntityPopup = (() => {
    let activePopup = null;
    let cache = {};

    function create(entityId, anchorEl) {
        remove(); // Remove any existing popup

        const popup = document.createElement("div");
        popup.className = "entity-popup";
        popup.innerHTML = '<div class="popup-loading">Loading connections…</div>';

        // Position relative to anchor element
        const rect = anchorEl.getBoundingClientRect();
        popup.style.position = "fixed";
        popup.style.left = `${rect.right + 8}px`;
        popup.style.top = `${rect.top}px`;

        document.body.appendChild(popup);
        activePopup = popup;

        // Reposition if offscreen
        requestAnimationFrame(() => {
            const popupRect = popup.getBoundingClientRect();
            if (popupRect.right > window.innerWidth - 16) {
                popup.style.left = `${rect.left - popupRect.width - 8}px`;
            }
            if (popupRect.bottom > window.innerHeight - 16) {
                popup.style.top = `${window.innerHeight - popupRect.height - 16}px`;
            }
        });

        return popup;
    }

    function remove() {
        if (activePopup) {
            activePopup.remove();
            activePopup = null;
        }
    }

    function render(popup, data) {
        if (!data || !data.entity) {
            popup.innerHTML = '<div class="popup-error">No network data available.</div>';
            return;
        }

        const { entity, edges, metrics } = data;
        const typeColor = NetworkColors.get(entity.type);

        let html = `
            <h4>${entity.name}</h4>
            <span class="entity-type-badge" style="background: ${typeColor}20; color: ${typeColor};">
                ${entity.type}
            </span>
            <span style="margin-left: 8px; color: #666; font-size: 0.85em;">
                ${entity.mention_count} document${entity.mention_count !== 1 ? "s" : ""}
            </span>
        `;

        if (edges && edges.length > 0) {
            html += `
                <div style="margin-top: 8px; font-size: 0.85em; color: #666;">
                    ${metrics.degree} connection${metrics.degree !== 1 ? "s" : ""}
                </div>
                <ul class="connections-list">
            `;
            edges.slice(0, 5).forEach((edge) => {
                const c = NetworkColors.get(edge.type);
                html += `
                    <li>
                        <span style="color: ${c}; font-weight: 500;">●</span>
                        ${edge.name}
                        <span style="float: right; color: #999; font-size: 0.85em;">${edge.weight}</span>
                    </li>
                `;
            });
            html += "</ul>";
        } else {
            html += '<div style="margin-top: 8px; color: #999; font-size: 0.85em;">No connections found.</div>';
        }

        html += `<a class="view-network-link" href="/network-analysis?entity=${entity.id}">View full network →</a>`;

        popup.innerHTML = html;
    }

    async function show(entityId, anchorEl) {
        const popup = create(entityId, anchorEl);

        if (cache[entityId]) {
            render(popup, cache[entityId]);
            return;
        }

        const data = await NetworkAPI.getEgoNetwork(entityId, { limit: 5 });
        if (data) {
            cache[entityId] = data;
        }
        // Popup may have been removed while loading
        if (activePopup === popup) {
            render(popup, data);
        }
    }

    return { show, remove };
})();


// ===========================================================================
// Document detail page initialization
// ===========================================================================
function initDocumentDetailNetwork() {
    // --- Entity popup handlers ---
    document.addEventListener("click", (e) => {
        const entityEl = e.target.closest("[data-entity-id]");
        if (entityEl) {
            e.preventDefault();
            const entityId = entityEl.dataset.entityId;
            EntityPopup.show(entityId, entityEl);
        } else if (!e.target.closest(".entity-popup")) {
            EntityPopup.remove();
        }
    });

    // --- Network context panel ---
    const contextContainer = document.getElementById("network-context-graph");
    const docId = contextContainer?.dataset?.docId;
    if (contextContainer && docId) {
        loadDocumentContextGraph(contextContainer, docId);
    }

    // --- Related documents ---
    const relatedContainer = document.getElementById("network-related-docs");
    const relDocId = relatedContainer?.dataset?.docId;
    if (relatedContainer && relDocId) {
        loadRelatedDocuments(relatedContainer, relDocId);
    }
}


// ===========================================================================
// Document context graph (small inline D3 force graph)
// ===========================================================================
async function loadDocumentContextGraph(container, docId) {
    const data = await NetworkAPI.getDocumentNetwork(docId, { min_weight: 2 });
    if (!data || !data.nodes || data.nodes.length < 2 || data.edges.length === 0) {
        container.closest(".info-section")?.remove();
        return;
    }

    renderForceGraph(container, data.nodes, data.edges, {
        width: container.clientWidth || 600,
        height: 350,
        nodeRadius: (d) => Math.min(4 + Math.sqrt(d.mention_count || 1) * 1.5, 20),
        interactive: true,
    });
}


// ===========================================================================
// Related documents via network
// ===========================================================================
async function loadRelatedDocuments(container, docId) {
    const data = await NetworkAPI.getRelatedDocuments(docId, { limit: 8 });
    if (!data || !data.related || data.related.length === 0) {
        container.closest(".info-section")?.remove();
        return;
    }

    const formatEntity = (entity) => {
        if (typeof entity === "string") return entity;
        const name = entity?.name || entity?.text || entity?.entity_id || "Unknown";
        const type = entity?.type ? ` (${entity.type})` : "";
        return `${name}${type}`;
    };

    let html = '<ul class="related-docs-list">';
    data.related.forEach((doc) => {
        const sharedEntities = Array.isArray(doc.shared_entities) ? doc.shared_entities : [];
        const sharedCount = Number.isFinite(doc.shared_entity_count)
            ? doc.shared_entity_count
            : sharedEntities.length;
        const sharedLabel = sharedCount === 1 ? "shared entity" : "shared entities";
        const sharedText = sharedEntities.length
            ? sharedEntities.map(formatEntity).join(", ")
            : "No direct shared entities listed.";

        const connectors = Array.isArray(doc.network_connector_entities)
            ? doc.network_connector_entities
            : [];
        const connectorText = connectors.length
            ? connectors.slice(0, 4).map(formatEntity).join(", ")
            : "";

        html += `
            <li>
                <a href="/document/${doc.document_id}">${doc.filename}</a>
                <span class="shared-count">${sharedCount} ${sharedLabel} with current document</span>
                <span class="shared-names"><strong>Shared:</strong> ${sharedText}</span>
                ${
                    connectorText
                        ? `<span class="shared-names"><strong>Connected via:</strong> ${connectorText}</span>`
                        : ""
                }
            </li>
        `;
    });
    html += "</ul>";
    container.innerHTML = html;
}


function showGraphMessage(container, message) {
    if (!container) return;
    container.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;padding:16px;text-align:center;">
            ${message}
        </div>
    `;
}


// ===========================================================================
// Force-directed graph renderer (shared by context panel + explorer page)
// ===========================================================================
function renderForceGraph(container, nodes, edges, opts = {}) {
    const {
        width = 800,
        height = 500,
        nodeRadius = (d) => Math.min(4 + Math.sqrt(d.mention_count || 1) * 1.5, 20),
        interactive = true,
        onNodeClick = null,
        onNodeHover = null,
    } = opts;

    // Check for D3
    if (typeof d3 === "undefined") {
        container.innerHTML = '<p style="color: #999;">D3.js not loaded.</p>';
        return;
    }

    // Clear container
    container.innerHTML = "";

    const svg = d3
        .select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height]);

    // Build a node lookup for edge resolution
    const nodeMap = {};
    nodes.forEach((n) => { nodeMap[n.id] = n; });

    // Filter edges to only those with both endpoints in nodes
    const validEdges = edges.filter((e) => nodeMap[e.source] && nodeMap[e.target]);

    // D3 force simulation data (copies, not mutations)
    const simNodes = nodes.map((n) => ({ ...n }));
    const simEdges = validEdges.map((e) => ({
        source: e.source,
        target: e.target,
        weight: e.weight || 1,
    }));

    // Zoom behavior
    const g = svg.append("g");
    if (interactive) {
        svg.call(
            d3.zoom()
                .scaleExtent([0.3, 5])
                .on("zoom", (event) => g.attr("transform", event.transform))
        );
    }

    // Simulation
    const simulation = d3
        .forceSimulation(simNodes)
        .force("link", d3.forceLink(simEdges).id((d) => d.id).distance(80))
        .force("charge", d3.forceManyBody().strength(-120))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20));

    // Edges
    const link = g
        .append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(simEdges)
        .join("line")
        .attr("stroke", "#bbb")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", (d) => Math.min(1 + Math.log(d.weight), 6));

    // Nodes
    const node = g
        .append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(simNodes)
        .join("circle")
        .attr("data-node-id", (d) => d.id)
        .attr("r", nodeRadius)
        .attr("fill", (d) => NetworkColors.get(d.type))
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .style("cursor", "pointer");

    // Labels
    const label = g
        .append("g")
        .attr("class", "labels")
        .selectAll("text")
        .data(simNodes)
        .join("text")
        .attr("data-node-id", (d) => d.id)
        .text((d) => d.name)
        .attr("font-size", "10px")
        .attr("dx", 12)
        .attr("dy", 4)
        .attr("fill", "#333");

    // Tooltip
    const tooltip = d3
        .select(container)
        .append("div")
        .attr("class", "graph-tooltip")
        .style("position", "absolute")
        .style("display", "none")
        .style("background", "white")
        .style("border", "1px solid #ccc")
        .style("border-radius", "4px")
        .style("padding", "6px 10px")
        .style("font-size", "0.85em")
        .style("pointer-events", "none")
        .style("box-shadow", "0 2px 6px rgba(0,0,0,0.1)");

    node.on("mouseover", (event, d) => {
        tooltip
            .style("display", "block")
            .html(`<strong>${d.name}</strong><br/><span style="color: ${NetworkColors.get(d.type)}">${d.type}</span> · ${d.mention_count || "?"} docs`)
            .style("left", `${event.offsetX + 12}px`)
            .style("top", `${event.offsetY - 12}px`);
        if (onNodeHover) onNodeHover(d);
    });

    node.on("mouseout", () => {
        tooltip.style("display", "none");
    });

    node.on("click", (event, d) => {
        if (onNodeClick) {
            onNodeClick(d);
        } else {
            // Default: navigate to ego network page
            window.location.href = `/network-analysis?entity=${d.id}`;
        }
    });

    // Drag behavior
    if (interactive) {
        node.call(
            d3.drag()
                .on("start", (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on("drag", (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on("end", (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                })
        );
    }

    // Tick
    simulation.on("tick", () => {
        link
            .attr("x1", (d) => d.source.x)
            .attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x)
            .attr("y2", (d) => d.target.y);
        node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
        label.attr("x", (d) => d.x).attr("y", (d) => d.y);
    });

    // Return handles for external control
    return {
        simulation,
        svg,
        nodes: simNodes,
        edges: simEdges,
        nodeSelection: node,
        labelSelection: label,
        linkSelection: link,
    };
}


// ===========================================================================
// Network Explorer page (standalone /network-analysis)
// ===========================================================================
const NetworkExplorer = {
    graph: null,
    currentData: null,
    statsData: null,
    activeTypePair: null,
    availableTypes: [],

    async init(containerSelector = "#network-graph") {
        const container = document.querySelector(containerSelector);
        if (!container) return;

        // Load available types for filter controls
        const typesData = await NetworkAPI.getTypes();
        if (typesData?.types) {
            this.availableTypes = typesData.types;
            this.renderTypeFilters(typesData.types);
        }

        // Load stats
        const stats = await NetworkAPI.getStats();
        if (stats) {
            this.statsData = stats;
            this.renderStats(stats);
        }
        if (!stats || stats.exists === false) {
            showGraphMessage(container, "Network data is not available yet. Build network edges to enable this view.");
        }

        const strictToggle = document.getElementById("strict-type-filter");
        if (strictToggle) {
            strictToggle.addEventListener("change", () => {
                this.reloadCurrentView(container);
            });
        }

        const presetSelect = document.getElementById("pair-preset");
        if (presetSelect) {
            presetSelect.addEventListener("change", () => {
                this.reloadCurrentView(container);
            });
        }

        const rankingSelect = document.getElementById("ranking-mode");
        if (rankingSelect) {
            rankingSelect.addEventListener("change", () => {
                this.reloadCurrentView(container);
            });
        }

        const searchBox = document.getElementById("entity-search");
        if (searchBox) {
            let searchDebounce = null;
            searchBox.addEventListener("input", () => {
                if (searchDebounce) {
                    window.clearTimeout(searchDebounce);
                }
                searchDebounce = window.setTimeout(() => {
                    this.reloadCurrentView(container);
                }, 350);
            });
            searchBox.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    event.preventDefault();
                    if (searchDebounce) {
                        window.clearTimeout(searchDebounce);
                        searchDebounce = null;
                    }
                    this.reloadCurrentView(container);
                }
            });
        }

        const applyTemplateBtn = document.getElementById("apply-research-template");
        if (applyTemplateBtn) {
            applyTemplateBtn.addEventListener("click", () => this.applyResearchTemplate(container));
        }

        const resetBtn = document.getElementById("reset-network-controls");
        if (resetBtn) {
            resetBtn.addEventListener("click", () => this.resetControls(container));
        }

        const openViewerBtn = document.getElementById("open-network-viewer");
        if (openViewerBtn) {
            openViewerBtn.addEventListener("click", () => this.openDocumentViewer(openViewerBtn));
        }

        const personMinMentions = document.getElementById("person-min-mentions");
        if (personMinMentions) {
            let personDebounce = null;
            personMinMentions.addEventListener("input", () => {
                if (personDebounce) window.clearTimeout(personDebounce);
                personDebounce = window.setTimeout(() => this.reloadCurrentView(container), 300);
            });
            personMinMentions.addEventListener("change", () => this.reloadCurrentView(container));
        }

        const highlightMetric = document.getElementById("node-highlight-metric");
        if (highlightMetric) {
            highlightMetric.addEventListener("change", () => this.refreshAnalyticsAndHighlights());
        }

        const highlightTopN = document.getElementById("node-highlight-topn");
        if (highlightTopN) {
            highlightTopN.addEventListener("input", () => this.refreshAnalyticsAndHighlights());
            highlightTopN.addEventListener("change", () => this.refreshAnalyticsAndHighlights());
        }

        this.attachStatsInteractions(container);
        this.updateActivePairState();
        this.updateOpenViewerButtonLabel();
        HelpTooltips.attach(document);

        await this.reloadCurrentView(container);
    },

    async reloadCurrentView(container) {
        const params = new URLSearchParams(window.location.search);
        const entityId = params.get("entity");
        if (entityId) {
            await this.loadEgoNetwork(container, entityId);
        } else {
            await this.loadGlobalNetwork(container);
        }
    },

    async resetControls(container) {
        const graphContainer = container || document.querySelector("#network-graph");
        if (!graphContainer) return;

        const minWeight = document.getElementById("min-weight-slider");
        const minWeightDisplay = document.getElementById("min-weight-display");
        if (minWeight) {
            minWeight.value = minWeight.defaultValue || "3";
            if (minWeightDisplay) minWeightDisplay.textContent = minWeight.value;
        }

        const maxEdges = document.getElementById("max-nodes-slider");
        const maxEdgesDisplay = document.getElementById("max-nodes-display");
        if (maxEdges) {
            maxEdges.value = maxEdges.defaultValue || "500";
            if (maxEdgesDisplay) maxEdgesDisplay.textContent = maxEdges.value;
        }

        const strictToggle = document.getElementById("strict-type-filter");
        if (strictToggle) {
            strictToggle.checked = strictToggle.defaultChecked;
        }

        const presetSelect = document.getElementById("pair-preset");
        if (presetSelect) {
            presetSelect.value = presetSelect.defaultValue || "cross_type_only";
        }

        const rankingSelect = document.getElementById("ranking-mode");
        if (rankingSelect) {
            rankingSelect.value = rankingSelect.defaultValue || "most_connected";
        }

        const templateSelect = document.getElementById("research-template");
        if (templateSelect) {
            templateSelect.value = "";
        }

        const searchBox = document.getElementById("entity-search");
        if (searchBox) {
            searchBox.value = "";
        }

        const personMinMentions = document.getElementById("person-min-mentions");
        if (personMinMentions) {
            personMinMentions.value = personMinMentions.defaultValue || "3";
        }

        const highlightMetric = document.getElementById("node-highlight-metric");
        if (highlightMetric) {
            highlightMetric.value = highlightMetric.defaultValue || "harmonic";
        }

        const highlightTopN = document.getElementById("node-highlight-topn");
        const highlightTopNDisplay = document.getElementById("node-highlight-topn-display");
        if (highlightTopN) {
            highlightTopN.value = highlightTopN.defaultValue || "10";
            if (highlightTopNDisplay) highlightTopNDisplay.textContent = highlightTopN.value;
        }

        document.querySelectorAll(".type-filter-checkbox").forEach((cb) => {
            cb.checked = true;
        });

        this.activeTypePair = null;
        this.updatePairBadgeState();
        this.updateActivePairState();
        this.updateOpenViewerButtonLabel();

        // Reset to canonical global view: clear URL params like ?entity=...
        const cleanUrl = new URL(window.location.href);
        cleanUrl.search = "";
        window.history.replaceState({}, "", cleanUrl.pathname);

        // Restore sidebar default prompt.
        const sidebar = document.getElementById("entity-detail-sidebar");
        if (sidebar) {
            sidebar.innerHTML = `
                <div style="color: #999; padding: 16px;">
                    Click a node to see entity details.
                </div>
            `;
        }

        await this.loadGlobalNetwork(graphContainer);
    },

    async loadGlobalNetwork(container) {
        const filters = this.getFilterValues();
        const requestedLimit = Number.parseInt(filters.limit || "500", 10) || 500;
        const apiFilters = { ...filters };
        if (filters.rank_mode && filters.rank_mode !== "most_connected") {
            apiFilters.limit = String(Math.min(requestedLimit * 4, 3000));
        }

        const data = await NetworkAPI.getGlobalNetwork(apiFilters);
        if (!data) {
            showGraphMessage(container, "Network endpoint unavailable. Check feature flag or server logs.");
            this.currentData = { nodes: [], edges: [] };
            this.refreshAnalyticsAndHighlights();
            return;
        }

        const transformed = this.transformGraphData(data, requestedLimit);
        if (!transformed.nodes.length || !transformed.edges.length) {
            showGraphMessage(container, "No edges match the selected filters.");
            this.currentData = { nodes: [], edges: [] };
            this.refreshAnalyticsAndHighlights();
            return;
        }

        this.currentData = transformed;
        this.graph = renderForceGraph(container, transformed.nodes, transformed.edges, {
            width: container.clientWidth || 1000,
            height: 600,
            interactive: true,
            onNodeClick: (d) => this.selectEntity(d),
        });

        this.renderLegend();
        this.refreshAnalyticsAndHighlights();
    },

    async loadEgoNetwork(container, entityId) {
        const filters = this.getFilterValues();
        const data = await NetworkAPI.getEgoNetwork(entityId, {
            ...filters,
            limit: 400,
        });
        if (!data || !data.entity) {
            showGraphMessage(container, "Entity network unavailable.");
            this.currentData = { nodes: [], edges: [] };
            this.refreshAnalyticsAndHighlights();
            return;
        }

        // Convert ego response to graph format
        const nodes = [data.entity];
        const edges = [];
        data.edges.forEach((e) => {
            nodes.push({
                id: e.entity_id,
                name: e.name,
                type: e.type,
                mention_count: 0,
            });
            edges.push({
                source: data.entity.id,
                target: e.entity_id,
                weight: e.weight,
                source_type: data.entity.type,
                target_type: e.type,
            });
        });

        const transformed = this.transformGraphData({ nodes, edges }, 100);
        if (!transformed.nodes.length || !transformed.edges.length) {
            showGraphMessage(container, "No edges match the selected filters.");
            this.currentData = { nodes: [], edges: [] };
            this.refreshAnalyticsAndHighlights();
            return;
        }

        this.currentData = transformed;
        this.graph = renderForceGraph(container, transformed.nodes, transformed.edges, {
            width: container.clientWidth || 1000,
            height: 600,
            interactive: true,
            onNodeClick: (d) => this.selectEntity(d),
        });

        this.selectEntity(data.entity);
        this.renderLegend();
        this.refreshAnalyticsAndHighlights();
    },

    transformGraphData(data, finalLimit = null) {
        const nodes = Array.isArray(data?.nodes) ? data.nodes : [];
        const rawEdges = Array.isArray(data?.edges) ? data.edges : [];
        const pairPreset = this.getPairPreset();
        const rankMode = this.getRankingMode();

        let edges = rawEdges.filter((edge) => this.edgeMatchesPreset(edge, pairPreset));
        edges = edges.filter((edge) => this.edgeMatchesActivePair(edge));

        const pairCounts = {};
        edges.forEach((edge) => {
            const key = `${edge.source_type || "UNKNOWN"}|${edge.target_type || "UNKNOWN"}`;
            pairCounts[key] = (pairCounts[key] || 0) + 1;
        });

        edges.sort((a, b) => this.compareEdges(a, b, rankMode, pairCounts));

        if (finalLimit && Number.isFinite(finalLimit) && finalLimit > 0) {
            edges = edges.slice(0, finalLimit);
        }

        const nodeById = {};
        nodes.forEach((node) => {
            nodeById[node.id] = node;
        });

        const requiredNodeIds = new Set();
        edges.forEach((edge) => {
            requiredNodeIds.add(edge.source);
            requiredNodeIds.add(edge.target);
        });

        const filteredNodes = Array.from(requiredNodeIds)
            .map((nodeId) => nodeById[nodeId])
            .filter(Boolean);

        return { nodes: filteredNodes, edges };
    },

    edgeMatchesPreset(edge, preset) {
        const sourceType = edge.source_type || "";
        const targetType = edge.target_type || "";
        if (preset === "within_type_only") {
            return sourceType === targetType;
        }
        if (preset === "cross_type_only") {
            return sourceType !== targetType;
        }
        return true;
    },

    edgeMatchesActivePair(edge) {
        if (!this.activeTypePair) return true;
        const sourceType = edge.source_type || "";
        const targetType = edge.target_type || "";
        const a = this.activeTypePair.source_type;
        const b = this.activeTypePair.target_type;
        return (
            (sourceType === a && targetType === b) ||
            (sourceType === b && targetType === a)
        );
    },

    compareEdges(a, b, rankMode, pairCounts) {
        const aWeight = Number.parseInt(a.weight || 0, 10);
        const bWeight = Number.parseInt(b.weight || 0, 10);

        const aSource = a.source_type || "UNKNOWN";
        const aTarget = a.target_type || "UNKNOWN";
        const bSource = b.source_type || "UNKNOWN";
        const bTarget = b.target_type || "UNKNOWN";

        const aPairKey = `${aSource}|${aTarget}`;
        const bPairKey = `${bSource}|${bTarget}`;
        const aPairCount = pairCounts[aPairKey] || 1;
        const bPairCount = pairCounts[bPairKey] || 1;
        const aCross = aSource !== aTarget;
        const bCross = bSource !== bTarget;

        if (rankMode === "most_cross_type") {
            if (aCross !== bCross) return aCross ? -1 : 1;
            if (aWeight !== bWeight) return bWeight - aWeight;
            if (aPairCount !== bPairCount) return aPairCount - bPairCount;
            return aPairKey.localeCompare(bPairKey);
        }

        if (rankMode === "rare_but_strong") {
            const aScore = aWeight / aPairCount;
            const bScore = bWeight / bPairCount;
            if (aScore !== bScore) return bScore - aScore;
            if (aWeight !== bWeight) return bWeight - aWeight;
            return aPairKey.localeCompare(bPairKey);
        }

        if (rankMode === "low_frequency_pairs") {
            if (aPairCount !== bPairCount) return aPairCount - bPairCount;
            if (aWeight !== bWeight) return bWeight - aWeight;
            return aPairKey.localeCompare(bPairKey);
        }

        if (aWeight !== bWeight) return bWeight - aWeight;
        if (aPairCount !== bPairCount) return aPairCount - bPairCount;
        return aPairKey.localeCompare(bPairKey);
    },

    getPairPreset() {
        const presetSelect = document.getElementById("pair-preset");
        const value = presetSelect ? presetSelect.value : "cross_type_only";
        if (value === "within_type_only" || value === "all_pairs" || value === "cross_type_only") {
            return value;
        }
        return "cross_type_only";
    },

    getRankingMode() {
        const rankingSelect = document.getElementById("ranking-mode");
        const value = rankingSelect ? rankingSelect.value : "most_connected";
        const allowed = new Set([
            "most_connected",
            "most_cross_type",
            "rare_but_strong",
            "low_frequency_pairs",
        ]);
        return allowed.has(value) ? value : "most_connected";
    },

    setActiveTypePair(sourceType, targetType, syncTypeFilters = true) {
        if (!sourceType || !targetType) {
            this.activeTypePair = null;
        } else if (
            this.activeTypePair &&
            this.activeTypePair.source_type === sourceType &&
            this.activeTypePair.target_type === targetType
        ) {
            this.activeTypePair = null;
        } else {
            this.activeTypePair = {
                source_type: sourceType,
                target_type: targetType,
            };
        }

        if (this.activeTypePair && syncTypeFilters) {
            const allowed = new Set([this.activeTypePair.source_type, this.activeTypePair.target_type]);
            document.querySelectorAll(".type-filter-checkbox").forEach((checkbox) => {
                checkbox.checked = allowed.has(checkbox.value);
            });
            const strictToggle = document.getElementById("strict-type-filter");
            if (strictToggle) strictToggle.checked = true;
            const presetSelect = document.getElementById("pair-preset");
            if (presetSelect) presetSelect.value = "all_pairs";
        }

        this.updatePairBadgeState();
        this.updateActivePairState();
        this.updateOpenViewerButtonLabel();
    },

    updatePairBadgeState() {
        const active = this.activeTypePair;
        document.querySelectorAll(".stat-badge-button[data-source-type][data-target-type]").forEach((badge) => {
            const sourceType = badge.dataset.sourceType;
            const targetType = badge.dataset.targetType;
            const isActive = !!active && (
                (active.source_type === sourceType && active.target_type === targetType) ||
                (active.source_type === targetType && active.target_type === sourceType)
            );
            badge.classList.toggle("stat-badge-active", isActive);
        });
    },

    updateActivePairState() {
        const stateEl = document.getElementById("active-type-pair-state");
        if (!stateEl) return;

        if (!this.activeTypePair) {
            stateEl.innerHTML = "Type pair focus: none";
            return;
        }

        stateEl.innerHTML = `
            Type pair focus: <strong>${this.activeTypePair.source_type}↔${this.activeTypePair.target_type}</strong>
            <button type="button" id="clear-active-type-pair">Clear</button>
        `;
        const clearBtn = document.getElementById("clear-active-type-pair");
        if (clearBtn) {
            clearBtn.addEventListener("click", () => {
                this.activeTypePair = null;
                this.updatePairBadgeState();
                this.updateActivePairState();
                this.updateOpenViewerButtonLabel();
                const graphContainer = document.querySelector("#network-graph");
                if (graphContainer) this.reloadCurrentView(graphContainer);
            });
        }
    },

    attachStatsInteractions(container) {
        const statsEl = document.getElementById("network-stats");
        if (!statsEl) return;
        statsEl.addEventListener("click", (event) => {
            const badge = event.target.closest(".stat-badge-button[data-source-type][data-target-type]");
            if (!badge) return;
            const sourceType = badge.dataset.sourceType;
            const targetType = badge.dataset.targetType;
            this.setActiveTypePair(sourceType, targetType, true);
            if (container) this.reloadCurrentView(container);
        });
    },

    applyResearchTemplate(container) {
        const templateSelect = document.getElementById("research-template");
        if (!templateSelect) return;

        const templates = {
            person_org: {
                types: ["PERSON", "ORG"],
                pair: ["PERSON", "ORG"],
            },
            person_gpe: {
                types: ["PERSON", "GPE"],
                pair: ["PERSON", "GPE"],
            },
            occupation_gpe: {
                types: ["OCCUPATION", "GPE"],
                pair: ["OCCUPATION", "GPE"],
            },
            employee_person_reconciliation: {
                types: ["EMPLOYEE_ID", "PERSON"],
                pair: ["EMPLOYEE_ID", "PERSON"],
            },
        };

        const selected = templates[templateSelect.value];
        if (!selected) return;

        const allowed = new Set(selected.types);
        document.querySelectorAll(".type-filter-checkbox").forEach((checkbox) => {
            checkbox.checked = allowed.has(checkbox.value);
        });

        const strictToggle = document.getElementById("strict-type-filter");
        if (strictToggle) strictToggle.checked = true;

        const presetSelect = document.getElementById("pair-preset");
        if (presetSelect) presetSelect.value = "all_pairs";

        this.setActiveTypePair(selected.pair[0], selected.pair[1], false);
        if (container) this.reloadCurrentView(container);
    },

    openDocumentViewer(buttonEl = null) {
        const launchUrl = new URL("/network/viewer-launch", window.location.origin);
        const filters = this.getFilterValues();
        Object.entries(filters).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== "") {
                launchUrl.searchParams.set(key, value);
            }
        });

        const params = new URLSearchParams(window.location.search);
        const entityId = params.get("entity");
        if (entityId) {
            launchUrl.searchParams.set("entity", entityId);
        }

        if (buttonEl) {
            buttonEl.disabled = true;
            buttonEl.textContent = "Opening...";
        }

        window.location.href = launchUrl.toString();
    },

    updateOpenViewerButtonLabel() {
        const openViewerBtn = document.getElementById("open-network-viewer");
        if (!openViewerBtn) return;
        if (this.activeTypePair) {
            openViewerBtn.textContent = "Open Document Viewer for Pair";
            return;
        }
        openViewerBtn.textContent = "Open Document Viewer";
    },

    selectEntity(entityData) {
        const sidebar = document.getElementById("entity-detail-sidebar");
        if (!sidebar) return;

        sidebar.innerHTML = `
            <h3>${entityData.name}</h3>
            <p>
                <span class="entity-type-badge" style="background: ${NetworkColors.get(entityData.type)}20; color: ${NetworkColors.get(entityData.type)}">
                    ${entityData.type}
                </span>
            </p>
            <p>${entityData.mention_count || "?"} documents</p>
            <p><a href="/network-analysis?entity=${entityData.id}">View ego network</a></p>
            <div id="entity-sidebar-metrics">Loading metrics…</div>
        `;

        // Load metrics async
        NetworkAPI.getEntityMetrics(entityData.id).then((metrics) => {
            const metricsEl = document.getElementById("entity-sidebar-metrics");
            if (!metricsEl || !metrics) return;

            let html = `
                <p><strong>Degree:</strong> ${metrics.degree}</p>
                <p><strong>Weighted degree:</strong> ${metrics.weighted_degree}</p>
            `;

            if (metrics.top_connections?.length) {
                html += "<h4>Top connections</h4><ul>";
                metrics.top_connections.slice(0, 10).forEach((c) => {
                    html += `<li><a href="/network-analysis?entity=${c.entity_id}">${c.name}</a> (${c.weight})</li>`;
                });
                html += "</ul>";
            }

            metricsEl.innerHTML = html;
        });
    },

    getFilterValues() {
        const params = {};
        const typeCheckboxes = document.querySelectorAll(".type-filter-checkbox:checked");
        if (typeCheckboxes.length > 0) {
            params.type_filter = Array.from(typeCheckboxes).map((cb) => cb.value).join(",");
        }

        const minWeight = document.getElementById("min-weight-slider");
        if (minWeight) {
            params.min_weight = minWeight.value;
        }

        const limitSlider = document.getElementById("max-nodes-slider");
        if (limitSlider) {
            params.limit = limitSlider.value;
        }

        const strictToggle = document.getElementById("strict-type-filter");
        if (strictToggle) {
            params.strict_type_filter = strictToggle.checked ? "true" : "false";
        }

        const pairPreset = this.getPairPreset();
        if (pairPreset) {
            params.pair_preset = pairPreset;
        }

        const rankingMode = this.getRankingMode();
        if (rankingMode) {
            params.rank_mode = rankingMode;
        }

        if (this.activeTypePair) {
            params.type_pair = `${this.activeTypePair.source_type}|${this.activeTypePair.target_type}`;
        }

        const searchBox = document.getElementById("entity-search");
        if (searchBox && searchBox.value && searchBox.value.trim()) {
            params.document_term = searchBox.value.trim();
        }

        const personMinMentions = document.getElementById("person-min-mentions");
        if (personMinMentions && personMinMentions.value !== "") {
            params.person_min_mentions = personMinMentions.value;
        }

        return params;
    },

    refreshAnalyticsAndHighlights() {
        const nodes = this.currentData?.nodes || [];
        const edges = this.currentData?.edges || [];
        const analytics = this.computeGraphAnalytics(nodes, edges);
        this.renderWholeNetworkStats(analytics);

        const metricKey = this.getSelectedCentralityMetric();
        const centralityRows = this.rankNodesByCentrality(analytics, metricKey);
        this.renderNodeCentralityList(centralityRows, metricKey);
        this.applyCentralityHighlight(centralityRows);
    },

    getSelectedCentralityMetric() {
        const selector = document.getElementById("node-highlight-metric");
        const value = selector ? selector.value : "harmonic";
        const allowed = new Set(["degree", "weighted_degree", "harmonic", "betweenness"]);
        return allowed.has(value) ? value : "harmonic";
    },

    getTopHighlightCount() {
        const slider = document.getElementById("node-highlight-topn");
        const value = Number.parseInt(slider?.value || "10", 10);
        if (!Number.isFinite(value) || value < 1) return 10;
        return Math.min(value, 100);
    },

    computeGraphAnalytics(nodes, edges) {
        const nodeIds = nodes.map((node) => node.id);
        const nodeSet = new Set(nodeIds);
        const edgeList = [];
        const adjacency = {};
        const degree = {};
        const weightedDegree = {};

        nodeIds.forEach((id) => {
            adjacency[id] = new Map();
            degree[id] = 0;
            weightedDegree[id] = 0;
        });

        edges.forEach((edge) => {
            const source = typeof edge.source === "object" ? edge.source.id : edge.source;
            const target = typeof edge.target === "object" ? edge.target.id : edge.target;
            const weight = Number.parseInt(edge.weight || 1, 10) || 1;
            if (!source || !target || source === target) return;
            if (!nodeSet.has(source) || !nodeSet.has(target)) return;
            if (adjacency[source].has(target)) return;

            adjacency[source].set(target, weight);
            adjacency[target].set(source, weight);
            degree[source] += 1;
            degree[target] += 1;
            weightedDegree[source] += weight;
            weightedDegree[target] += weight;
            edgeList.push({ source, target, weight });
        });

        const nodeCount = nodeIds.length;
        const edgeCount = edgeList.length;
        const density = nodeCount > 1 ? (2 * edgeCount) / (nodeCount * (nodeCount - 1)) : 0;

        const bfsDistances = (start) => {
            const dist = {};
            dist[start] = 0;
            const queue = [start];
            let index = 0;
            while (index < queue.length) {
                const current = queue[index++];
                const currentDist = dist[current];
                adjacency[current].forEach((_w, neighbor) => {
                    if (dist[neighbor] !== undefined) return;
                    dist[neighbor] = currentDist + 1;
                    queue.push(neighbor);
                });
            }
            return dist;
        };

        const components = [];
        const visited = new Set();
        nodeIds.forEach((start) => {
            if (visited.has(start)) return;
            const queue = [start];
            const members = [];
            visited.add(start);
            let index = 0;
            while (index < queue.length) {
                const nodeId = queue[index++];
                members.push(nodeId);
                adjacency[nodeId].forEach((_w, neighbor) => {
                    if (visited.has(neighbor)) return;
                    visited.add(neighbor);
                    queue.push(neighbor);
                });
            }
            components.push(members);
        });

        let largestComponent = [];
        components.forEach((component) => {
            if (component.length > largestComponent.length) largestComponent = component;
        });

        // Average shortest path length (largest component only).
        let pathPairCount = 0;
        let pathDistanceSum = 0;
        if (largestComponent.length > 1) {
            for (let i = 0; i < largestComponent.length; i += 1) {
                const source = largestComponent[i];
                const distances = bfsDistances(source);
                for (let j = i + 1; j < largestComponent.length; j += 1) {
                    const target = largestComponent[j];
                    if (distances[target] === undefined) continue;
                    pathPairCount += 1;
                    pathDistanceSum += distances[target];
                }
            }
        }
        const averagePathLength = pathPairCount > 0 ? pathDistanceSum / pathPairCount : null;

        // Harmonic centrality (normalized by N-1).
        const harmonic = {};
        let harmonicSum = 0;
        nodeIds.forEach((source) => {
            const distances = bfsDistances(source);
            let score = 0;
            Object.entries(distances).forEach(([target, dist]) => {
                if (target === source || dist <= 0) return;
                score += 1 / dist;
            });
            const normalized = nodeCount > 1 ? score / (nodeCount - 1) : 0;
            harmonic[source] = normalized;
            harmonicSum += normalized;
        });
        const meanHarmonicCentrality = nodeCount > 0 ? harmonicSum / nodeCount : 0;

        // Betweenness centrality (Brandes, unweighted).
        const betweenness = {};
        nodeIds.forEach((id) => { betweenness[id] = 0; });
        nodeIds.forEach((source) => {
            const stack = [];
            const predecessors = {};
            const sigma = {};
            const distance = {};
            nodeIds.forEach((id) => {
                predecessors[id] = [];
                sigma[id] = 0;
                distance[id] = -1;
            });
            sigma[source] = 1;
            distance[source] = 0;

            const queue = [source];
            let qIndex = 0;
            while (qIndex < queue.length) {
                const v = queue[qIndex++];
                stack.push(v);
                adjacency[v].forEach((_w, w) => {
                    if (distance[w] < 0) {
                        queue.push(w);
                        distance[w] = distance[v] + 1;
                    }
                    if (distance[w] === distance[v] + 1) {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                });
            }

            const delta = {};
            nodeIds.forEach((id) => { delta[id] = 0; });
            while (stack.length > 0) {
                const w = stack.pop();
                predecessors[w].forEach((v) => {
                    if (sigma[w] > 0) {
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]);
                    }
                });
                if (w !== source) {
                    betweenness[w] += delta[w];
                }
            }
        });
        if (nodeCount > 2) {
            const scale = 1 / ((nodeCount - 1) * (nodeCount - 2) / 2);
            nodeIds.forEach((id) => {
                betweenness[id] *= scale;
            });
        }

        // Clustering / transitivity.
        let clusteringSum = 0;
        let clusteringEligible = 0;
        let triangleAtNodeSum = 0;
        let connectedTriples = 0;
        nodeIds.forEach((nodeId) => {
            const neighbors = Array.from(adjacency[nodeId].keys());
            const k = neighbors.length;
            if (k < 2) return;

            const possible = (k * (k - 1)) / 2;
            connectedTriples += possible;
            clusteringEligible += 1;
            let trianglePairs = 0;

            for (let i = 0; i < neighbors.length; i += 1) {
                for (let j = i + 1; j < neighbors.length; j += 1) {
                    if (adjacency[neighbors[i]].has(neighbors[j])) {
                        trianglePairs += 1;
                    }
                }
            }

            triangleAtNodeSum += trianglePairs;
            clusteringSum += trianglePairs / possible;
        });

        const triangles = triangleAtNodeSum / 3;
        const averageClustering = clusteringEligible > 0 ? clusteringSum / clusteringEligible : 0;
        const transitivity = connectedTriples > 0 ? (3 * triangles) / connectedTriples : 0;

        // Simple community labels via label propagation.
        const labels = {};
        nodeIds.forEach((id) => { labels[id] = id; });
        const maxIterations = 15;
        for (let iteration = 0; iteration < maxIterations; iteration += 1) {
            let changed = false;
            nodeIds.forEach((nodeId) => {
                const counts = {};
                adjacency[nodeId].forEach((_w, neighbor) => {
                    const label = labels[neighbor];
                    counts[label] = (counts[label] || 0) + 1;
                });
                const options = Object.entries(counts);
                if (!options.length) return;
                options.sort((a, b) => {
                    if (a[1] !== b[1]) return b[1] - a[1];
                    return String(a[0]).localeCompare(String(b[0]));
                });
                const nextLabel = options[0][0];
                if (labels[nodeId] !== nextLabel) {
                    labels[nodeId] = nextLabel;
                    changed = true;
                }
            });
            if (!changed) break;
        }

        const communityMap = {};
        nodeIds.forEach((id) => {
            const label = labels[id];
            if (!communityMap[label]) communityMap[label] = [];
            communityMap[label].push(id);
        });
        const communityIds = Object.keys(communityMap);

        let modularity = 0;
        if (edgeCount > 0) {
            const communitySets = {};
            communityIds.forEach((label) => {
                communitySets[label] = new Set(communityMap[label]);
            });
            communityIds.forEach((label) => {
                const set = communitySets[label];
                let l_c = 0;
                let d_c = 0;
                edgeList.forEach((edge) => {
                    if (set.has(edge.source) && set.has(edge.target)) l_c += 1;
                });
                communityMap[label].forEach((nodeId) => {
                    d_c += degree[nodeId] || 0;
                });
                modularity += (l_c / edgeCount) - ((d_c / (2 * edgeCount)) ** 2);
            });
        }

        const centralityByNodeId = {};
        nodeIds.forEach((id) => {
            centralityByNodeId[id] = {
                degree: degree[id] || 0,
                weighted_degree: weightedDegree[id] || 0,
                harmonic: harmonic[id] || 0,
                betweenness: betweenness[id] || 0,
            };
        });

        return {
            nodeCount,
            edgeCount,
            density,
            componentCount: components.length,
            largestComponentSize: largestComponent.length,
            averagePathLength,
            meanHarmonicCentrality,
            averageClustering,
            modularity,
            transitivity,
            directed: false,
            reciprocity: null,
            asymmetry: null,
            centralityByNodeId,
            nodesById: Object.fromEntries(nodes.map((n) => [n.id, n])),
        };
    },

    rankNodesByCentrality(analytics, metricKey) {
        const rows = Object.entries(analytics.centralityByNodeId || {}).map(([nodeId, metrics]) => {
            return {
                nodeId,
                value: Number(metrics[metricKey] || 0),
            };
        });
        rows.sort((a, b) => b.value - a.value || String(a.nodeId).localeCompare(String(b.nodeId)));
        return rows;
    },

    applyCentralityHighlight(rankedRows) {
        if (!this.graph?.nodeSelection || !this.graph?.labelSelection) return;

        const topCount = this.getTopHighlightCount();
        const topIds = new Set(rankedRows.slice(0, topCount).map((row) => row.nodeId));

        this.graph.nodeSelection
            .attr("stroke", (d) => (topIds.has(d.id) ? "#1e3a8a" : "#ffffff"))
            .attr("stroke-width", (d) => (topIds.has(d.id) ? 3 : 1.5))
            .attr("opacity", (d) => (topIds.has(d.id) ? 1 : 0.72))
            .attr("r", (d) => {
                const base = Math.min(4 + Math.sqrt(d.mention_count || 1) * 1.5, 20);
                return topIds.has(d.id) ? Math.min(base * 1.3, 28) : base;
            });

        this.graph.labelSelection
            .attr("fill", (d) => (topIds.has(d.id) ? "#1e3a8a" : "#333"))
            .attr("font-weight", (d) => (topIds.has(d.id) ? 700 : 400))
            .attr("font-size", (d) => (topIds.has(d.id) ? "11px" : "10px"));
    },

    renderWholeNetworkStats(analytics) {
        const container = document.getElementById("whole-network-stats");
        if (!container) return;

        const formatNumber = (value, digits = 3) => {
            if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
            if (!Number.isFinite(value)) return "N/A";
            return Number(value).toFixed(digits);
        };

        const cards = [
            ["Node count", analytics.nodeCount],
            ["Edge count", analytics.edgeCount],
            ["Density", formatNumber(analytics.density, 4)],
            ["Average path length", analytics.averagePathLength === null ? "N/A" : formatNumber(analytics.averagePathLength, 3)],
            ["Mean harmonic centrality", formatNumber(analytics.meanHarmonicCentrality, 4)],
            ["Average clustering", formatNumber(analytics.averageClustering, 4)],
            ["Modularity (label propagation)", formatNumber(analytics.modularity, 4)],
            ["Transitivity", formatNumber(analytics.transitivity, 4)],
            ["Connected components", analytics.componentCount],
            ["Largest component size", analytics.largestComponentSize],
            ["Directed graph metric", analytics.directed ? "Directed" : "Undirected"],
            ["Reciprocity", analytics.directed ? formatNumber(analytics.reciprocity, 4) : "N/A (undirected)"],
            ["Asymmetry", analytics.directed ? formatNumber(analytics.asymmetry, 4) : "N/A (undirected)"],
        ];

        container.innerHTML = cards
            .map(([label, value]) => {
                return `
                    <div class="stat-card">
                        <span class="stat-label">${label}</span>
                        <span class="stat-value">${value}</span>
                    </div>
                `;
            })
            .join("");
    },

    renderNodeCentralityList(rankedRows, metricKey) {
        const container = document.getElementById("node-centrality-list");
        if (!container) return;

        const metricLabels = {
            degree: "degree centrality",
            weighted_degree: "weighted degree",
            harmonic: "harmonic centrality",
            betweenness: "betweenness centrality",
        };
        const topCount = this.getTopHighlightCount();
        const topRows = rankedRows.slice(0, topCount);

        if (!topRows.length) {
            container.innerHTML = '<div class="network-stats-loading">No nodes available for centrality ranking.</div>';
            return;
        }

        const formatValue = (value) => (Number.isFinite(value) ? Number(value).toFixed(4) : "0.0000");
        container.innerHTML = `
            <div style="margin-bottom: 6px; color: #475569; font-size: 0.85em;">
                Highlighting top ${topRows.length} nodes by ${metricLabels[metricKey] || metricKey}.
            </div>
            <ol>
                ${topRows.map((row) => {
                    const node = this.currentData?.nodes?.find((n) => n.id === row.nodeId);
                    const name = node?.name || row.nodeId;
                    const type = node?.type ? ` (${node.type})` : "";
                    return `<li><strong>${name}</strong>${type} — ${formatValue(row.value)}</li>`;
                }).join("")}
            </ol>
        `;
    },

    renderTypeFilters(types) {
        const container = document.getElementById("type-filters");
        if (!container) return;

        container.innerHTML = types
            .map((t) => {
                const color = NetworkColors.get(t);
                return `
            <label style="margin-right: 12px;">
                <input type="checkbox" class="type-filter-checkbox" value="${t}" checked>
                <span style="color: ${color}; font-size: 1.15em; line-height: 1; vertical-align: middle;">●</span>
                <span style="color: ${color}; font-weight: 600;">${t}</span>
            </label>
        `
            })
            .join("");

        // Re-render on change
        container.addEventListener("change", () => {
            const graphContainer = document.querySelector("#network-graph");
            if (graphContainer) {
                this.reloadCurrentView(graphContainer);
            }
        });
        HelpTooltips.attach(container);
    },

    renderStats(stats) {
        const el = document.getElementById("network-stats");
        if (!el || !stats.exists) return;

        const total = stats.total_edges || 0;
        el.innerHTML = `
            <span>${total.toLocaleString()} edges</span>
            ${stats.type_pairs
                ?.map(
                    (tp) => {
                        const percent = total > 0 ? ((tp.count / total) * 100).toFixed(1) : "0.0";
                        const help = `Click to focus ${tp.source_type}↔${tp.target_type}. Count ${tp.count.toLocaleString()}, Avg weight ${tp.avg_weight}, Max weight ${tp.max_weight}, Share ${percent}%.`;
                        return `<button type="button" class="stat-badge stat-badge-button" data-source-type="${tp.source_type}" data-target-type="${tp.target_type}" data-help="${escapeHtmlAttribute(help)}">${tp.source_type}↔${tp.target_type}: ${tp.count.toLocaleString()}</button>`;
                    }
                )
                .join("") || ""}
        `;
        this.updatePairBadgeState();
        HelpTooltips.attach(el);
    },

    renderLegend() {
        const container = document.getElementById("network-legend");
        if (!container) return;

        const colors = NetworkColors.getAll();
        container.innerHTML = Object.entries(colors)
            .map(
                ([type, color]) =>
                    `<span style="margin-right: 14px; color: ${color}; font-weight: 600;"><span style="font-size: 1.2em; line-height: 1; vertical-align: middle;">●</span> ${type}</span>`
            )
            .join("");
    },
};


// ===========================================================================
// Auto-initialize based on page context
// ===========================================================================
document.addEventListener("DOMContentLoaded", () => {
    // Document detail page — entity popups + context panel
    if (document.querySelector("[data-entity-id]") || document.getElementById("network-context-graph")) {
        initDocumentDetailNetwork();
    }

    // Network explorer page
    if (document.getElementById("network-graph")) {
        NetworkExplorer.init("#network-graph");
    }
});
