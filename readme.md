# Historical Document Reader


Organize data ingestion from 
database_setup      creates the db
json_validator      checks texts for inclusion
database_processing ingests the data


##Needed FEatures

2. Ability to mark documents for followup
3. Highlighting of search terms in produced text [attempted to code but difficult]
4. Preloading of next document somehow
5.
6. 
7. mouseover wikipedia search


Architecture:

The application is built using Flask, a Python web framework.
It uses MongoDB as the NoSQL database, accessed via the PyMongo library.
The frontend is created using HTML, CSS, and JavaScript, with some dynamic content rendering.


Main Components:

app.py: The main Flask application file that initializes the app and includes configurations.
routes.py: Contains all the route handlers for different endpoints.
database_setup.py: Manages database connections and operations.
data_processing.py: Handles the ingestion of JSON files into the MongoDB database.
static/script.js: Contains client-side JavaScript for handling user interactions and AJAX requests.
static/style.css: Defines the styling for the application.
Various HTML templates in the templates folder for rendering different pages.


Functionality:

Document Search: Users can search for documents using multiple fields and operators (AND, OR, NOT).
Pagination: Search results are paginated for better performance and user experience.
Document Viewing: Users can view individual documents with their details and associated images.
Database Information: Provides an overview of the database structure and field counts.
Search Terms Analysis: Allows users to explore the frequency of terms in different fields.
Settings: Users can customize UI settings like fonts, colors, and spacing.
Authentication: Basic login functionality is implemented for accessing certain features.


Data Flow:

JSON files are processed and inserted into MongoDB using data_processing.py.
The application dynamically discovers and adapts to the structure of the documents in the database.
Search queries are constructed based on user input and executed against MongoDB.
Results are returned to the frontend and displayed using a combination of server-side rendering and client-side JavaScript.


Notable Features:

Dynamic field structure discovery and adaptation.
Infinite scrolling for search results.
AJAX-based search to improve responsiveness.
Prefetching of next page results for smoother scrolling.
Export to CSV functionality for search results.
Image viewing with zoom and pan capabilities for document images.


Security Measures:

Password protection for certain routes.
CAPTCHA implementation to prevent automated login attempts.
Session management for maintaining user state.



This application provides a flexible and efficient way to search and view historical documents stored in a NoSQL database, with a focus on performance and user experience.