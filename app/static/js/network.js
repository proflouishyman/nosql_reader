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
    return { simulation, svg, nodes: simNodes, edges: simEdges };
}


// ===========================================================================
// Network Explorer page (standalone /network-analysis)
// ===========================================================================
const NetworkExplorer = {
    graph: null,
    currentData: null,

    async init(containerSelector = "#network-graph") {
        const container = document.querySelector(containerSelector);
        if (!container) return;

        // Load available types for filter controls
        const typesData = await NetworkAPI.getTypes();
        if (typesData?.types) {
            this.renderTypeFilters(typesData.types);
        }

        // Load stats
        const stats = await NetworkAPI.getStats();
        if (stats) {
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

        const resetBtn = document.getElementById("reset-network-controls");
        if (resetBtn) {
            resetBtn.addEventListener("click", () => this.resetControls(container));
        }

        const openViewerBtn = document.getElementById("open-network-viewer");
        if (openViewerBtn) {
            openViewerBtn.addEventListener("click", () => this.openDocumentViewer(openViewerBtn));
        }

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

        const searchBox = document.getElementById("entity-search");
        if (searchBox) {
            searchBox.value = "";
        }

        document.querySelectorAll(".type-filter-checkbox").forEach((cb) => {
            cb.checked = true;
        });

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
        const data = await NetworkAPI.getGlobalNetwork(filters);
        if (!data) {
            showGraphMessage(container, "Network endpoint unavailable. Check feature flag or server logs.");
            return;
        }
        if (!data.nodes?.length || !data.edges?.length) {
            showGraphMessage(container, "No edges match the selected filters.");
            return;
        }

        this.currentData = data;
        this.graph = renderForceGraph(container, data.nodes, data.edges, {
            width: container.clientWidth || 1000,
            height: 600,
            interactive: true,
            onNodeClick: (d) => this.selectEntity(d),
        });

        this.renderLegend();
    },

    async loadEgoNetwork(container, entityId) {
        const filters = this.getFilterValues();
        const data = await NetworkAPI.getEgoNetwork(entityId, {
            ...filters,
            limit: 100,
        });
        if (!data || !data.entity) {
            showGraphMessage(container, "Entity network unavailable.");
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
            });
        });

        this.currentData = { nodes, edges };
        this.graph = renderForceGraph(container, nodes, edges, {
            width: container.clientWidth || 1000,
            height: 600,
            interactive: true,
            onNodeClick: (d) => this.selectEntity(d),
        });

        this.selectEntity(data.entity);
        this.renderLegend();
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

        return params;
    },

    renderTypeFilters(types) {
        const container = document.getElementById("type-filters");
        if (!container) return;

        container.innerHTML = types
            .map(
                (t) => `
            <label style="margin-right: 12px;">
                <input type="checkbox" class="type-filter-checkbox" value="${t}" checked>
                <span style="color: ${NetworkColors.get(t)}; font-weight: 500;">●</span> ${t}
            </label>
        `
            )
            .join("");

        // Re-render on change
        container.addEventListener("change", () => {
            const graphContainer = document.querySelector("#network-graph");
            if (graphContainer) {
                this.reloadCurrentView(graphContainer);
            }
        });
    },

    renderStats(stats) {
        const el = document.getElementById("network-stats");
        if (!el || !stats.exists) return;

        el.innerHTML = `
            <span>${stats.total_edges.toLocaleString()} edges</span>
            ${stats.type_pairs
                ?.map(
                    (tp) =>
                        `<span class="stat-badge">${tp.source_type}↔${tp.target_type}: ${tp.count.toLocaleString()}</span>`
                )
                .join("") || ""}
        `;
    },

    renderLegend() {
        const container = document.getElementById("network-legend");
        if (!container) return;

        const colors = NetworkColors.getAll();
        container.innerHTML = Object.entries(colors)
            .map(
                ([type, color]) =>
                    `<span style="margin-right: 12px;"><span style="color: ${color}; font-weight: bold;">●</span> ${type}</span>`
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
