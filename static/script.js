// File: static/script.js

document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const resultsDiv = document.getElementById('results');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const cancelButton = document.getElementById('cancelSearch');
    const totalResultsDiv = document.getElementById('totalResults');

    // Add console warnings for missing elements
    if (!searchForm) console.warn('Search form not found');
    if (!resultsDiv) console.warn('Results div not found');
    if (!loadingIndicator) console.warn('Loading indicator not found');
    if (!cancelButton) console.warn('Cancel button not found');
    if (!totalResultsDiv) console.warn('Total results div not found');

    let controller;
    let page = 1;
    let totalPages = 1;
    const perPage = 50;  // Fixed number of results per request
    let totalResults = 0;
    let isLoading = false;
    let hasMore = true;
    let currentQuery = {};
    let prefetchedData = null; // Declare prefetchedData

    // Handle form submission
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            // Reset variables
            page = 1;
            hasMore = true;
            if (resultsDiv) resultsDiv.innerHTML = '';
            if (totalResultsDiv) totalResultsDiv.textContent = '';
            
            // Get search parameters
            const formData = new FormData(searchForm);
            currentQuery = {};
            for (let i = 1; i <= 3; i++) {
                currentQuery[`field${i}`] = formData.get(`field${i}`);
                currentQuery[`operator${i}`] = formData.get(`operator${i}`);
                currentQuery[`searchTerm${i}`] = formData.get(`searchTerm${i}`);
            }

            // Check if form fields have valid values before proceeding
            if (!currentQuery['field1'] || !currentQuery['searchTerm1']) {
                console.error('Please enter a valid search term in the first field.');
                return;
            }

            performSearch(true);
        });
    }

    // Cancel search functionality
    if (cancelButton) {
        cancelButton.addEventListener('click', function() {
            if (controller) {
                controller.abort();
                hideLoadingIndicator();
                isLoading = false;
                hasMore = false;
                if (cancelButton) cancelButton.style.display = 'none';
            }
        });
    }

    // Function to fetch results
    function performSearch(isNewSearch = false) {
        if (isLoading || !hasMore) return;
        isLoading = true;

        if (isNewSearch) {
            // Clear prefetched data on new search
            prefetchedData = null;
        }

        showLoadingIndicator();
        if (cancelButton) cancelButton.style.display = 'inline-block';

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
            if (cancelButton) cancelButton.style.display = 'none';
            if (data.documents && data.documents.length > 0) {
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
            if (cancelButton) cancelButton.style.display = 'none';
            isLoading = false;
            if (error.name === 'AbortError') {
                console.log('Search was cancelled');
            } else {
                console.error('Error:', error);
            }
        });
    }

    function showLoadingIndicator() {
        if (loadingIndicator && loadingIndicator.style) {
            loadingIndicator.style.display = 'block';
        }
    }

    function hideLoadingIndicator() {
        if (loadingIndicator && loadingIndicator.style) {
            loadingIndicator.style.display = 'none';
        }
    }

    // Function to prefetch the next page of results
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

    // Infinite scroll logic with debounce
    let debounceTimeout;
    window.addEventListener('scroll', function() {
        clearTimeout(debounceTimeout);
        debounceTimeout = setTimeout(function() {
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
        }, 200);
    });

    // Function to append results to the page
    function appendResults(documents) {
        if (!resultsDiv) return;

        let table = document.getElementById('resultsTable');
        let tbody;

        // If the table doesn't exist yet, create it
        if (!table) {
            table = document.createElement('table');
            table.id = 'resultsTable';

            // Create table headers
            const thead = document.createElement('thead');
            thead.innerHTML = `
                <tr>
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
                <td><a href="/document/${doc._id}">${doc.filename || 'No file name'}</a></td>
                <td>${doc.summary || 'No summary available.'}</td>
            `;
            tbody.appendChild(row);
        });
    }

    // Function to update total results display
    function updateTotalResults() {
        if (totalResultsDiv) {
            totalResultsDiv.textContent = `Total results: ${totalResults}`;
        }
    }

    // Export to CSV functionality
    const exportCsvButton = document.getElementById('exportCsv');
    if (exportCsvButton) {
        exportCsvButton.addEventListener('click', function() {
            if (!searchForm) return;

            const formData = new FormData(searchForm);
            const searchData = {
                fields: []
            };

            for (let i = 1; i <= 3; i++) {
                const field = formData.get(`field${i}`);
                const operator = formData.get(`operator${i}`);
                const searchTerm = formData.get(`searchTerm${i}`);
                
                if (field && operator && searchTerm) {
                    searchData.fields.push({
                        field: field,
                        operator: operator,
                        searchTerm: searchTerm
                    });
                }
            }

            fetch('/export_csv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(searchData)
            })
            .then(response => response.blob())
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'search_results.csv';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => console.error('Error:', error));
        });
    }
});