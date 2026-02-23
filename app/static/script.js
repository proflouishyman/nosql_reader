document.addEventListener('DOMContentLoaded', function() {
    // ===============================
    // Initialization
    // ===============================
    const searchForm = document.getElementById('searchForm');
    if (!searchForm) {
        return;
    }

    const resultsDiv = document.getElementById('results');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const cancelButton = document.getElementById('cancelSearch');
    const totalResultsDiv = document.getElementById('totalResults');
    const exportSelectedCsvButton = document.getElementById('exportSelectedCsv');

    // Pagination and Search Variables
    let controller;
    let page = 1;
    let totalPages = 1;
    const perPage = 50;  // Fixed number of results per request
    let totalResults = 0;
    let isLoading = false;
    let hasMore = true;
    let currentQuery = {};
    let prefetchedData = null;

    // Selection Management
    let selectedDocuments = new Set();

    // Variable to store current search_id
    let searchId = null;

    // ===============================
    // Utility Functions
    // ===============================

    /**
     * Debounce function to limit the rate at which a function can fire.
     * @param {Function} func - The function to debounce.
     * @param {number} delay - The delay in milliseconds.
     * @returns {Function} - The debounced function.
     */
    function debounce(func, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }

    /**
     * Toggles the selection of a document.
     * @param {string} docId - The ID of the document.
     * @param {boolean} isSelected - Whether the document is selected.
     */
    function toggleDocumentSelection(docId, isSelected) {
        if (isSelected) {
            selectedDocuments.add(docId);
        } else {
            selectedDocuments.delete(docId);
        }
        updateExportButtonVisibility();
        saveSelectedDocuments();
    }

    /**
     * Updates the visibility of the Export Selected button based on selections.
     */
    function updateExportButtonVisibility() {
        if (!exportSelectedCsvButton) return;
        if (selectedDocuments.size > 0) {
            exportSelectedCsvButton.style.display = 'inline-block';
        } else {
            exportSelectedCsvButton.style.display = 'none';
        }
    }

    /**
     * Saves the selected documents to localStorage.
     */
    function saveSelectedDocuments() {
        localStorage.setItem('selectedDocuments', JSON.stringify(Array.from(selectedDocuments)));
    }

    /**
     * Loads the selected documents from localStorage.
     */
    function loadSelectedDocuments() {
        const savedSelectedDocuments = JSON.parse(localStorage.getItem('selectedDocuments') || '[]');
        savedSelectedDocuments.forEach(id => selectedDocuments.add(id));
        updateExportButtonVisibility();
    }

    /**
     * Shows the loading indicator.
     */
    function showLoadingIndicator() {
        if (!loadingIndicator) return;
        loadingIndicator.hidden = false;
        if (loadingIndicator.style) {
            loadingIndicator.style.display = 'block';
        }
    }

    /**
     * Hides the loading indicator.
     */
    function hideLoadingIndicator() {
        if (!loadingIndicator) return;
        loadingIndicator.hidden = true;
        if (loadingIndicator.style) {
            loadingIndicator.style.display = 'none';
        }
    }

    /**
     * Resets the search parameters and UI elements.
     */
    function resetSearch() {
        page = 1;
        hasMore = true;
        if (resultsDiv) resultsDiv.innerHTML = '';
        if (totalResultsDiv) totalResultsDiv.textContent = '';
        searchId = null;  // Reset search_id for new search
    }

    /**
     * Gathers search parameters from the form.
     */
    function gatherSearchParameters() {
        const formData = new FormData(searchForm);
        currentQuery = {};
        for (let i = 1; i <= 3; i++) {
            currentQuery[`field${i}`] = formData.get(`field${i}`);
            currentQuery[`operator${i}`] = formData.get(`operator${i}`);
            currentQuery[`searchTerm${i}`] = formData.get(`searchTerm${i}`);
        }
    }

    /**
     * Validates the search parameters.
     * @returns {boolean} True if valid, else false.
     */
    function validateSearchParameters() {
        if (!currentQuery['field1'] || !currentQuery['searchTerm1']) {
            console.error('Please enter a valid search term in the first field.');
            alert('Please enter a valid search term in the first field.');
            return false;
        }
        return true;
    }

    /**
     * Handles the scroll event for infinite scrolling.
     */
    function handleScroll() {
        const scrollPosition = window.innerHeight + window.scrollY;
        const threshold = document.body.offsetHeight - 100;

        if (scrollPosition >= threshold) {
            if (prefetchedData) {
                // Use prefetched data
                appendResults(prefetchedData.documents);
                page += 1;
                totalPages = prefetchedData.total_pages;
                totalResults = prefetchedData.total_count;
                updateTotalResults();

                // Clear prefetched data and prefetch the next page
                prefetchedData = null;
                if (page <= totalPages) {
                    prefetchNextPage();
                } else {
                    hasMore = false;
                }
            } else {
                performSearch();
            }
        } else if (prefetchedData === null && (scrollPosition >= threshold / 2)) {
            // Start prefetching when user scrolls halfway
            prefetchNextPage();
        }
    }

    // ===============================
    // Event Listeners
    // ===============================

    // Handle form submission for search
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            resetSearch();
            gatherSearchParameters();
            if (validateSearchParameters()) {
                performSearch(true);
            }
        });
    }

    // Handle checkbox changes using event delegation
    if (resultsDiv) {
        resultsDiv.addEventListener('change', function(e) {
            if (e.target && e.target.matches('.select-document')) {
                const docId = e.target.getAttribute('data-doc-id');
                toggleDocumentSelection(docId, e.target.checked);
            }
        });
    }

    // Handle "Select All" functionality
    if (resultsDiv) {
        resultsDiv.addEventListener('change', function(e) {
            if (e.target && e.target.matches('#selectAll')) {
                const checkboxes = resultsDiv.querySelectorAll('.select-document');
                const isChecked = e.target.checked;
                checkboxes.forEach(cb => {
                    cb.checked = isChecked;
                    const docId = cb.getAttribute('data-doc-id');
                    if (isChecked) {
                        selectedDocuments.add(docId);
                    } else {
                        selectedDocuments.delete(docId);
                    }
                });
                updateExportButtonVisibility();
                saveSelectedDocuments();
            }
        });
    }

    // Handle Export Selected to CSV Button Click
    if (exportSelectedCsvButton) {
        exportSelectedCsvButton.addEventListener('click', function() {
            exportSelectedDocuments();
        });
    }

    // Handle Cancel Search Button Click
    if (cancelButton) {
        cancelButton.addEventListener('click', function() {
            cancelSearch();
        });
    }

    // Handle Infinite Scroll with Debounce
    window.addEventListener('scroll', debounce(handleScroll, 200));

    // ===============================
    // Search Functionality
    // ===============================

    /**
     * Performs the search by sending a POST request to the server.
     * @param {boolean} isNewSearch - Indicates if it's a new search.
     */
    function performSearch(isNewSearch = false) {
        if (isLoading || !hasMore) return;
        isLoading = true;

        if (isNewSearch) {
            prefetchedData = null;
            searchId = null;  // Reset search_id for new search
        }

        showLoadingIndicator();
        if (cancelButton) {
            cancelButton.hidden = false;
            cancelButton.style.display = 'inline-block';
        }

        // Add page and perPage to currentQuery
        currentQuery.page = page;
        currentQuery.per_page = perPage;

        controller = new AbortController();
        const signal = controller.signal;

        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentQuery),
            signal: signal
        })
        .then(response => response.json())
        .then(data => {
            hideLoadingIndicator();
            if (cancelButton) {
                cancelButton.hidden = true;
                cancelButton.style.display = 'none';
            }
            if (data.documents && data.documents.length > 0) {
                // Store search_id if it's a new search
                if (isNewSearch) {
                    searchId = data.search_id;
                }

                appendResults(data.documents);
                totalPages = data.total_pages;
                totalResults = data.total_count;

                // Prefetch the next page if there are more pages
                if (page < totalPages) {
                    page += 1;
                    prefetchNextPage();
                } else {
                    hasMore = false;
                }
            } else {
                hasMore = false;
                if (isNewSearch && resultsDiv && resultsDiv.innerHTML === '') {
                    resultsDiv.innerHTML = '<p>No results found.</p>';
                }
            }
            updateTotalResults();
            isLoading = false;
        })
        .catch(error => {
            hideLoadingIndicator();
            if (cancelButton) {
                cancelButton.hidden = true;
                cancelButton.style.display = 'none';
            }
            isLoading = false;
            if (error.name === 'AbortError') {
                console.log('Search was cancelled');
            } else {
                console.error('Error:', error);
                alert('An error occurred during the search. Please try again.');
            }
        });
    }

    /**
     * Appends search results to the resultsDiv.
     * @param {Array} documents - List of document objects.
     */
    function appendResults(documents) {
        if (!resultsDiv) return;

        let table = document.getElementById('resultsTable');
        let tbody;

        // If the table doesn't exist yet, create it
        if (!table) {
            table = document.createElement('table');
            table.id = 'resultsTable';

            // Create table headers with a Select All checkbox
            const thead = document.createElement('thead');
            thead.innerHTML = `
                <tr>
                    <th><input type="checkbox" id="selectAll" /></th>
                    <th>File</th>
                    <th>Summary</th>
                </tr>
            `;
            table.appendChild(thead);

            // Create table body
            tbody = document.createElement('tbody');
            table.appendChild(tbody);

            // Append the table to the results div
            resultsDiv.appendChild(table);
        } else {
            // If the table exists, get its tbody
            tbody = table.querySelector('tbody');
        }

        // Append new rows to the table body
        documents.forEach(doc => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><input type="checkbox" class="select-document" data-doc-id="${doc._id}" /></td>
                <td><a href="/document/${doc._id}?search_id=${searchId}">${doc.filename || 'No file name'}</a></td>
                <td>${doc.summary || 'No summary available.'}</td>
            `;
            // If the document is already selected, check the box
            if (selectedDocuments.has(doc._id)) {
                row.querySelector('.select-document').checked = true;
            }
            tbody.appendChild(row);
        });
    }

    /**
     * Updates the total results display.
     */
    function updateTotalResults() {
        if (totalResultsDiv) {
            totalResultsDiv.textContent = `Total results: ${totalResults}`;
        }
    }

    /**
     * Prefetches the next page of results.
     */
    function prefetchNextPage() {
        if (prefetchedData || !hasMore) return;

        const prefetchQuery = { ...currentQuery, page: page };

        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(prefetchQuery),
        })
        .then(response => response.json())
        .then(data => {
            if (data.documents && data.documents.length > 0) {
                prefetchedData = data;
            } else {
                hasMore = false;
            }
        })
        .catch(error => {
            console.error('Error during prefetching:', error);
        });
    }

    // ===============================
    // Selection Management
    // ===============================

    /**
     * Exports the selected documents to CSV.
     */
    function exportSelectedDocuments() {
        if (selectedDocuments.size === 0) {
            alert('No documents selected.');
            return;
        }

        // Prepare the list of selected document IDs
        const selectedIds = Array.from(selectedDocuments);

        // Show a loading modal or notification
        showExportModal('Exporting selected documents...');

        // Send the list to the server via POST
        fetch('/export_selected_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ document_ids: selectedIds }),
        })
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                return response.json().then(err => { throw err; });
            }
        })
        .then(blob => {
            // Create a link to download the blob
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'selected_documents.csv';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

            // Hide the export modal and show success message
            hideExportModal();
            showExportModal('Export successful!', 'success');

            // Reset the export button
            exportSelectedCsvButton.disabled = false;
            exportSelectedCsvButton.textContent = 'Export Selected to CSV';

            // Optionally, clear the selectedDocuments set
            selectedDocuments.clear();
            updateExportButtonVisibility();

            // Uncheck all checkboxes
            const checkboxes = resultsDiv.querySelectorAll('.select-document');
            checkboxes.forEach(cb => cb.checked = false);

            // Uncheck "Select All" checkbox
            const selectAllCheckbox = document.getElementById('selectAll');
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = false;
            }

            // Remove success message after a short delay
            setTimeout(() => {
                hideExportModal();
            }, 3000);
        })
        .catch(error => {
            console.error('Error exporting selected documents:', error);
            hideExportModal();
            alert('Error exporting selected documents.');

            // Reset the export button
            exportSelectedCsvButton.disabled = false;
            exportSelectedCsvButton.textContent = 'Export Selected to CSV';
        });
    }

    /**
     * Cancels the ongoing search.
     */
    function cancelSearch() {
        if (controller) {
            controller.abort();
            hideLoadingIndicator();
            isLoading = false;
            hasMore = false;
            if (cancelButton) {
                cancelButton.hidden = true;
                cancelButton.style.display = 'none';
            }
        }
    }

    // ===============================
    // Export Feedback Mechanism
    // ===============================

    /**
     * Creates and displays an export modal for feedback.
     * @param {string} message - The message to display.
     * @param {string} type - The type of message ('info', 'success', 'error').
     */
    function showExportModal(message, type = 'info') {
        // Check if modal already exists
        let modal = document.getElementById('exportModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'exportModal';
            modal.innerHTML = `
                <div class="modal-content">
                    <span class="close-button">&times;</span>
                    <p id="exportMessage">${message}</p>
                </div>
            `;
            document.body.appendChild(modal);

            // Style the modal
            const style = document.createElement('style');
            style.textContent = `
                #exportModal {
                    display: block; 
                    position: fixed; 
                    z-index: 1000; 
                    left: 0;
                    top: 0;
                    width: 100%; 
                    height: 100%; 
                    overflow: auto; 
                    background-color: rgba(0,0,0,0.4); 
                }
                .modal-content {
                    background-color: #fefefe;
                    margin: 15% auto; 
                    padding: 20px;
                    border: 1px solid #888;
                    width: 300px; 
                    text-align: center;
                    border-radius: 5px;
                }
                .close-button {
                    color: #aaa;
                    float: right;
                    font-size: 28px;
                    font-weight: bold;
                    cursor: pointer;
                }
                .close-button:hover,
                .close-button:focus {
                    color: black;
                    text-decoration: none;
                }
                .modal-content.success {
                    border-color: #28a745;
                }
                .modal-content.error {
                    border-color: #dc3545;
                }
            `;
            document.head.appendChild(style);

            // Handle close button click
            modal.querySelector('.close-button').addEventListener('click', hideExportModal);
        }

        // Update the message and style based on type
        const exportMessage = modal.querySelector('#exportMessage');
        exportMessage.textContent = message;
        modal.querySelector('.modal-content').className = 'modal-content'; // Reset classes

        if (type === 'success') {
            modal.querySelector('.modal-content').classList.add('success');
        } else if (type === 'error') {
            modal.querySelector('.modal-content').classList.add('error');
        }

        // Display the modal
        modal.style.display = 'block';
    }

    /**
     * Hides the export modal.
     */
    function hideExportModal() {
        const modal = document.getElementById('exportModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    // ===============================
    // Initial Load
    // ===============================

    /**
     * Applies URL prefill parameters and automatically runs a search once.
     * Supported params:
     * - prefill_field
     * - prefill_term
     */
    function applyPrefillFromUrl() {
        const params = new URLSearchParams(window.location.search);
        const prefillField = (params.get('prefill_field') || '').trim();
        const prefillTerm = (params.get('prefill_term') || '').trim();

        if (!prefillField || !prefillTerm) {
            return;
        }

        const fieldSelect = document.getElementById('field1');
        if (!fieldSelect) {
            return;
        }

        const hasFieldOption = Array.from(fieldSelect.options).some(option => option.value === prefillField);
        if (!hasFieldOption) {
            return;
        }

        // Trigger existing field-change behavior first because some fields swap text input to dropdown.
        fieldSelect.value = prefillField;
        fieldSelect.dispatchEvent(new Event('change', { bubbles: true }));

        setTimeout(() => {
            const firstTermInput = document.getElementById('searchTerm1');
            if (!firstTermInput) {
                return;
            }

            if (firstTermInput.tagName === 'SELECT') {
                const hasTermOption = Array.from(firstTermInput.options).some(option => option.value === prefillTerm);
                if (!hasTermOption) {
                    return;
                }
                firstTermInput.value = prefillTerm;
            } else {
                firstTermInput.value = prefillTerm;
            }

            // Remove prefill params to avoid rerunning automatically on refresh/back nav.
            window.history.replaceState({}, document.title, window.location.pathname);
            searchForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
        }, 120);
    }

    // Load selected documents from localStorage
    loadSelectedDocuments();
    applyPrefillFromUrl();
});
