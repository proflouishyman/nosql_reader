$(document).ready(function() {
    let currentPage = 1;
    const perPage = 10;

    // Add field
    $('#add-field').click(function() {
        var newField = $('.search-field').first().clone();
        newField.find('input').val('');
        $('#search-fields').append(newField);
    });

    // Remove field
    $('#search-fields').on('click', '.remove-field', function() {
        if ($('.search-field').length > 1) {
            $(this).closest('.search-field').remove();
        }
    });

    // Search
    $('#search').click(function() {
        currentPage = 1;
        performSearch();
    });

    function performSearch() {
        var searchData = [];
        $('.search-field').each(function() {
            searchData.push({
                table: $(this).find('select[name="field"]').val(),
                operator: $(this).find('select[name="operator"]').val(),
                searchTerm: $(this).find('input[name="query"]').val()
            });
        });

        $.ajax({
            url: '/advanced_search',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                fields: searchData,
                page: currentPage,
                per_page: perPage
            }),
            success: function(response) {
                displayResults(response.results);
                displayPagination(response.total, response.pages, response.current_page);
            },
            error: function(error) {
                console.error('Error:', error);
                $('#search-results').html('<p>An error occurred during the search.</p>');
            }
        });
    }

    function displayResults(results) {
        var resultsHtml = '<table class="table"><thead><tr><th>ID</th><th>File</th><th>Summary</th></tr></thead><tbody>';
        results.forEach(function(result) {
            resultsHtml += '<tr>';
            resultsHtml += '<td>' + result.id + '</td>';
            resultsHtml += '<td>' + result.file + '</td>';
            resultsHtml += '<td>' + result.summary + '</td>';
            resultsHtml += '</tr>';
        });
        resultsHtml += '</tbody></table>';
        $('#search-results').html(resultsHtml);
    }

    function displayPagination(total, pages, currentPage) {
        var paginationHtml = '<nav><ul class="pagination">';
        for (var i = 1; i <= pages; i++) {
            paginationHtml += '<li class="page-item ' + (i === currentPage ? 'active' : '') + '">';
            paginationHtml += '<a class="page-link" href="#" data-page="' + i + '">' + i + '</a>';
            paginationHtml += '</li>';
        }
        paginationHtml += '</ul></nav>';
        $('#pagination').html(paginationHtml);
    }

    // Handle pagination clicks
    $(document).on('click', '.pagination a', function(e) {
        e.preventDefault();
        currentPage = parseInt($(this).data('page'));
        performSearch();
    });
});
