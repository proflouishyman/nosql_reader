# Historical Document Reader


##Needed FEatures
1.  Better scrolling.
Ability to look at sequential documents in a directory rather than the original
2. Ability to mark documents for followup
3. Highlighting of search terms in produced text [attempted to code but difficult]
4. Preloading of next document somehow
5.
6. 
7. mouseover wikipedia search



### Installation Instructions
1. Install the requirements:
pip install -r requirements.txt


2. Run the Application
python app.py

3. Click on the server which usually looks like "Running on http://127.0.0.1:5000"




## Files
G:.
├───app.py
├───models.py
├───routes.py
├───static/
│   ├───script.js
│   ├───search.js
│   ├───style.css
│   └───search_terms/
│       ├───ocr_text_terms.json
│       ├───summary_terms.json
│       ├───named_entities_terms.json
│       ├───dates_terms.json
│       ├───monetary_amounts_terms.json
│       ├───relationships_terms.json
│       ├───metadata_terms.json
│       ├───translation_terms.json
│       └───file_info_terms.json
└───templates/
    ├───document-detail.html
    ├───document-list.html
    ├───error.html
    ├───index.html
    ├───search-terms.html
    ├───database-info.html
    ├───style.css
    ├───settings.html
    └───base.html
    └───login.html



## Overview
The Historical Document Reader is a web application that provides an interface to perform complex SQL searches on a database containing information extracted from historical documents. The application allows users to construct advanced queries using boolean logic, specify search criteria across multiple tables, and retrieve relevant documents and data. The application runs locally and is accessed through a web browser.

## Current Features
- Advanced search functionality with multiple filters
- Pagination for search results
- Sorting options for search results

- Improved user interface with responsive design

## Features Currently Being Debugged/Implemented
1. Complex search queries across multiple tables
2. Document preview functionality
3. Performance optimization for large result sets
4. Caching of frequent searches
5. User-friendly error handling and feedback

## Database Schema
The database schema consists of the following tables:
Table: ocr_text
Columns:
- id (TEXT)
- file (TEXT)
- text (TEXT)

Table: summary
Columns:
- id (TEXT)
- file (TEXT)
- text (TEXT)

Table: named_entities
Columns:
- id (TEXT)
- file (TEXT)
- entity (TEXT)
- type (TEXT)

Table: dates
Columns:
- id (TEXT)
- file (TEXT)
- date (TEXT)

Table: monetary_amounts
Columns:
- id (TEXT)
- file (TEXT)
- amount (TEXT)
- category (TEXT)

Table: relationships
Columns:
- id (TEXT)
- file (TEXT)
- entity1 (TEXT)
- relationship (TEXT)
- entity2 (TEXT)

Table: metadata
Columns:
- id (TEXT)
- file (TEXT)
- document_type (TEXT)
- period (TEXT)
- context (TEXT)
- sentiment (TEXT)

Table: translation
Columns:
- id (TEXT)
- file (TEXT)
- french_text (TEXT)
- english_translation (TEXT)

Table: file_info
Columns:
- id (TEXT)
- file (TEXT)
- original_filepath (TEXT)



## Backend
- Built using the Flask web framework
- Uses SQLAlchemy for database interactions and query building
- Implements caching with Flask-Caching for improved performance
- Handles routing and processing of HTTP requests from the frontend
- Constructs complex SQL queries based on user-defined search criteria

## Frontend
- Built with HTML, CSS, and JavaScript
- Uses custom CSS for styling (potential future upgrade to Bootstrap or another UI framework)
- Communicates with the backend using AJAX requests
- Provides an intuitive interface for constructing complex search queries
- Displays search results in a clear and organized manner

## Search Functionality
- Supports advanced searches with multiple boolean criteria
- Allows filtering by text content, named entities, document type, date range, and monetary amounts, that is, all the tables
- Implements server-side pagination and sorting
- Supports boolean logic in search queries
- Allows users to specify search conditions across different tables and columns

## Local Development and Deployment
- Developed and run locally using Python and Flask
- Uses a SQLite database for data storage
- Accessed by running the Flask development server and opening the application in a web browser
- Local deployment eliminates the need for external hosting services

## Current Focus
- Debugging and refining the advanced search functionality
- Improving the user interface for a more intuitive search experience
- Optimizing database queries for better performance with large datasets
- Implement the database information feature, including query terms
- Developing the document preview functionality

## Next Steps

- Add data visualization features (e.g., charts for monetary amounts, network graphs for entity relationships)
- Enhance error handling and user feedback mechanisms
- Explore options for full-text search optimization
- Consider integrating a frontend framework like React or Vue.js for more dynamic user interactions


## Challenges and Ongoing Debugging Efforts
1. Optimizing complex queries that join multiple tables for better performance
2. Ensuring accurate results when combining different search criteria
3. Handling edge cases in date and monetary amount searches
4. Improving the responsiveness of the UI when dealing with large result sets
5. Refining the caching strategy to balance performance and data freshness
6. Developing an efficient document preview feature that doesn't overload the server
7. Ensuring consistent behavior across different browsers and devices

This comprehensive overview reflects the current state of the Historical Document Reader project, including its structure, features, database schema, and ongoing development efforts. It serves as a reference point for the project's current status and future direction.
