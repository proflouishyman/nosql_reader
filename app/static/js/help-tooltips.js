(function () {
    const SHOW_DELAY_MS = 2000;
    const TOOLTIP_CLASS = "app-delayed-help-tooltip";
    const TOOLTIP_STYLE_ID = "app-delayed-help-tooltip-style";
    const BOUND_ATTR = "data-help-tooltip-bound";
    const INTERACTIVE_SELECTOR = "a[href], button, input:not([type='hidden']), select, textarea, summary, [role='button'], [role='link']";
    const HELP_SELECTOR = `[data-help], [title], ${INTERACTIVE_SELECTOR}`;

    let tooltipEl = null;
    let showTimer = null;

    function ensureStyle() {
        if (document.getElementById(TOOLTIP_STYLE_ID)) return;
        const style = document.createElement("style");
        style.id = TOOLTIP_STYLE_ID;
        style.textContent = `
            .${TOOLTIP_CLASS} {
                position: fixed;
                z-index: 3000;
                max-width: 340px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.18);
                padding: 8px 10px;
                font-size: 0.82em;
                line-height: 1.35;
                pointer-events: none;
                display: none;
                white-space: normal;
            }
        `;
        document.head.appendChild(style);
    }

    function ensureTooltip() {
        if (tooltipEl) return tooltipEl;
        ensureStyle();
        tooltipEl = document.createElement("div");
        tooltipEl.className = TOOLTIP_CLASS;
        document.body.appendChild(tooltipEl);
        return tooltipEl;
    }

    function clearTimer() {
        if (showTimer) {
            window.clearTimeout(showTimer);
            showTimer = null;
        }
    }

    function hide() {
        clearTimer();
        if (tooltipEl) {
            tooltipEl.style.display = "none";
            tooltipEl.textContent = "";
        }
    }

    function isRangeInput(el) {
        return el.tagName === "INPUT" && String(el.getAttribute("type") || "").toLowerCase() === "range";
    }

    function isNumericString(text) {
        return /^-?\d+(\.\d+)?$/.test(text);
    }

    function cleanText(value) {
        return String(value || "").replace(/\s+/g, " ").trim();
    }

    function clipText(value, maxLen = 90) {
        const text = cleanText(value);
        if (text.length <= maxLen) return text;
        return `${text.slice(0, maxLen - 1)}…`;
    }

    function getLabelText(target) {
        const explicitAria = cleanText(target.getAttribute("aria-label"));
        if (explicitAria) return explicitAria;

        const id = target.getAttribute("id");
        if (id) {
            try {
                const escapedId = window.CSS && window.CSS.escape ? window.CSS.escape(id) : id.replace(/([#.;?+*~':"!^$[\]()=>|\/@])/g, "\\$1");
                const explicitLabel = document.querySelector(`label[for="${escapedId}"]`);
                if (explicitLabel) {
                    const labelText = cleanText(explicitLabel.textContent);
                    if (labelText) return labelText;
                }
            } catch (_) {
                // Ignore selector errors and continue with fallback heuristics.
            }
        }

        const wrappedLabel = target.closest("label");
        if (wrappedLabel) {
            const labelText = cleanText(wrappedLabel.textContent);
            if (labelText) return labelText;
        }

        const placeholder = cleanText(target.getAttribute("placeholder"));
        if (placeholder) return placeholder;

        return "";
    }

    function autoHelpMessage(target) {
        if (!(target instanceof HTMLElement)) return "";

        const tag = target.tagName.toLowerCase();
        const label = clipText(getLabelText(target));
        const text = clipText(target.textContent);

        if (tag === "a") {
            const linkText = text || label;
            if (!linkText) return "";
            return `Open ${linkText}.`;
        }

        if (tag === "button") {
            const buttonText = text || label;
            if (!buttonText) return "";
            return `Activate ${buttonText}.`;
        }

        if (tag === "select") {
            const selectLabel = label || "this option";
            return `Choose ${selectLabel}.`;
        }

        if (tag === "textarea") {
            const areaLabel = label || "text";
            return `Enter ${areaLabel}.`;
        }

        if (tag === "summary") {
            const summaryText = text || label;
            if (!summaryText) return "";
            return `Expand ${summaryText}.`;
        }

        if (tag === "input") {
            const type = String(target.getAttribute("type") || "text").toLowerCase();
            const inputLabel = label || "value";
            if (type === "checkbox" || type === "radio") {
                return `Toggle ${inputLabel}.`;
            }
            if (type === "range") {
                return `Adjust ${inputLabel}.`;
            }
            if (type === "file") {
                const multi = target.hasAttribute("multiple");
                return multi ? `Choose files for ${inputLabel}.` : `Choose a file for ${inputLabel}.`;
            }
            if (type === "submit" || type === "button") {
                const valueText = cleanText(target.getAttribute("value"));
                if (valueText) return `Activate ${clipText(valueText)}.`;
            }
            return `Enter ${inputLabel}.`;
        }

        if (target.getAttribute("role") === "button") {
            const roleText = text || label;
            if (!roleText) return "";
            return `Activate ${roleText}.`;
        }

        if (target.getAttribute("role") === "link") {
            const roleText = text || label;
            if (!roleText) return "";
            return `Open ${roleText}.`;
        }

        return "";
    }

    function getHelpMessage(target) {
        if (!(target instanceof HTMLElement)) return "";

        const explicit = (target.getAttribute("data-help") || "").trim();
        if (explicit) return explicit;

        const title = (target.getAttribute("title") || "").trim();
        if (title && !(isRangeInput(target) || isNumericString(title))) {
            // Promote title to data-help so browser-native tooltip does not fight ours.
            target.setAttribute("data-help", title);
            if (!target.getAttribute("aria-label")) {
                target.setAttribute("aria-label", title);
            }
            target.removeAttribute("title");
            return title;
        }

        const generated = autoHelpMessage(target);
        if (generated) {
            target.setAttribute("data-help", generated);
            return generated;
        }

        return "";
    }

    function positionTooltip(target, tooltip) {
        const rect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
        let top = rect.bottom + 10;

        if (left < 8) left = 8;
        if (left + tooltipRect.width > window.innerWidth - 8) {
            left = window.innerWidth - tooltipRect.width - 8;
        }
        if (top + tooltipRect.height > window.innerHeight - 8) {
            top = rect.top - tooltipRect.height - 10;
        }
        if (top < 8) top = 8;

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    }

    function show(target) {
        const message = getHelpMessage(target);
        if (!message) return;

        const tooltip = ensureTooltip();
        tooltip.textContent = message;
        tooltip.style.display = "block";
        positionTooltip(target, tooltip);
    }

    function scheduleShow(target) {
        const message = getHelpMessage(target);
        if (!message) return;
        clearTimer();
        showTimer = window.setTimeout(() => show(target), SHOW_DELAY_MS);
    }

    function bindTarget(target) {
        if (!(target instanceof HTMLElement)) return;
        if (target.getAttribute(BOUND_ATTR) === "1") return;
        target.setAttribute(BOUND_ATTR, "1");

        target.addEventListener("mouseenter", () => scheduleShow(target));
        target.addEventListener("mouseleave", hide);
        target.addEventListener("focus", () => scheduleShow(target));
        target.addEventListener("blur", hide);
        target.addEventListener("click", hide);
    }

    function attach(root = document) {
        if (!root) return;

        if (root instanceof HTMLElement && root.matches(HELP_SELECTOR)) {
            bindTarget(root);
        }

        const scope = root.querySelectorAll ? root : document;
        const targets = scope.querySelectorAll(HELP_SELECTOR);
        targets.forEach((target) => bindTarget(target));
    }

    function observeMutations() {
        if (!("MutationObserver" in window)) return;
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node instanceof HTMLElement) {
                        attach(node);
                    }
                });
            });
        });
        observer.observe(document.documentElement, { childList: true, subtree: true });
    }

    function init() {
        attach(document);
        observeMutations();
        document.addEventListener("scroll", hide, true);
        window.addEventListener("resize", hide);
    }

    window.AppHelpTooltips = { attach, hide, init };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
