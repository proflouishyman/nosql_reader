# Historical Document Reader


# put all data in data subdirectory

cd ~/Desktop/coding/nosql_reader
source .dbenv/bin/activate

# SETUP
You need to setup MongoDB
https://thelinuxforum.com/articles/912-how-to-install-mongodb-on-ubuntu-24-04

Using Docker
DOCKER:
Run run_docker.sh and it should install
if there are problems, install dos2unix and then convert the file.
everything runs better in WSL

client = MongoClient('mongodb://admin:secret@localhost:27017/')

to do
0 shift other pc to docker
0.1 create flag for list of hidden fields
-searchable fields
-result fields
implement the pictures

1 convertsetup process to a part of settings or a new page. it shoukd be able to add to thr db
2. create login splash page

4. address weirdness of base file and index html. it is unseemly 
5. reorganize routes py
6. add cross referencing of named entities
7. restore adding export from list
8. add Select all for export
9. clean file description to remove extension in title. 
10. create a way to do a search and then add that result to thr db. 
11. backup and restore database
12. remove blank fields from json expansion
13. color code sections of json expansion



## Description
The Historical Document Reader is a Flask-based web application designed to manage, search, and display historical documents stored in a MongoDB database. It provides an intuitive interface for researchers and historians to access and analyze digitized historical records.

## Features

### Document Management
- Data ingestion from JSON files
- Dynamic field structure discovery and adaptation
- Storage of documents in MongoDB

### Search Functionality
- Advanced search with multiple fields and logical operators (AND, OR, NOT)
- Infinite scrolling for search results
- AJAX-based search for improved responsiveness
- Prefetching of next page results for smoother scrolling

### Document Viewing
- Detailed view of individual documents
- Image viewing with zoom and pan capabilities for document images
- Navigation between documents in search results

### Data Analysis
- Search terms analysis with word and phrase frequency
- Database structure information display

### User Interface
- Customizable UI settings (fonts, colors, spacing)
- Responsive design for various screen sizes

### Data Export
- Export search results to CSV

### Security
- Basic authentication system
- CAPTCHA implementation to prevent automated login attempts

### Additional Features
- Error handling and user feedback
- Logging system for debugging and monitoring

## File Structure
```
historical_document_reader/
│
├── app.py
├── routes.py
├── models.py
├── database_setup.py
├── data_processing.py
├── json_validator.py
├── json_validator_multi.py
├── generate_password.py
├── requirements.txt
├── config.json
├── secret_key.txt
├── README.md
│
├── static/
│   ├── script.js
│   └── style.css
│
└── templates/
    ├── base.html
    ├── index.html
    ├── document-detail.html
    ├── document-list.html
    ├── search-terms.html
    ├── database-info.html
    ├── settings.html
    ├── login.html
    └── error.html
```

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/historical-document-reader.git
   ```

2. Navigate to the project directory:
   ```
   cd historical-document-reader
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up MongoDB:
   - Install MongoDB if you haven't already.
   - Start the MongoDB service.
   - Update the connection string in `database_setup.py` if necessary:
     ```python
     client = MongoClient('mongodb://localhost:27017/')
     ```

5. Initialize the database structure:
   ```
   python database_setup.py
   ```
   This script will create necessary indexes and initialize the field structure.

6. Prepare your JSON data:
   - Ensure your historical document data is in JSON format.
   - If your data is in .txt files, use the JSON validator to convert and validate them:
     ```
     python json_validator.py
     ```
   - This script will process .txt files in the specified directory, convert them to valid JSON, and save them with a .json extension.

7. Ingest data into the database:
   - Update the `data_directory` path in `data_processing.py` to point to your JSON files:
     ```python
     data_directory = r'path/to/your/json/files'
     ```
   - Run the data processing script:
     ```
     python data_processing.py
     ```
   - This script will read the JSON files, insert them into the MongoDB database, update the field structure, and compute unique terms for search functionality.

8. Generate a password for admin access (optional):
   ```
   python generate_password.py
   ```
   - This will generate a hashed password. Copy the output and update the `ADMIN_PASSWORD_HASH` in `routes.py`.

9. Run the application:
   ```
   python app.py
   ```

## Usage

1. Access the application through a web browser at `http://localhost:5000`.

2. If you set up admin access, log in using the password you generated.

3. Use the search interface to find documents:
   - Select fields from the dropdown menus.
   - Choose operators (AND, OR, NOT) to combine search criteria.
   - Enter search terms and click "Search".

4. View search results:
   - Click on a document title to view its details.
   - Use infinite scrolling to load more results.

5. Analyze search terms:
   - Navigate to the "Search Terms" page.
   - Select a field to see word and phrase frequencies.

6. View database information:
   - Go to the "Database Info" page to see the structure and record counts.

7. Customize the interface:
   - Use the "Settings" page to adjust fonts, colors, and spacing.

8. Export results:
   - Use the "Export to CSV" button to download search results.

## Maintenance and Updates

- To add new documents, place their JSON files in the data directory and run `data_processing.py` again.
- If the structure of your documents changes, the system will automatically adapt when you process new files.
- Regularly backup your MongoDB database to prevent data loss.

## Troubleshooting

- If you encounter issues with data processing, check the JSON format of your files and ensure they are valid.
- For database connection issues, verify your MongoDB service is running and the connection string is correct.
- Check the application logs for any error messages or unexpected behavior.

## Python Files Description

Here's a breakdown of the main Python files in the project and their functions:

### `app.py`
- Main application file
- Initializes the Flask app and configures it
- Sets up caching, logging, and session management
- Loads the UI configuration
- Defines context processors for injecting data into templates

### `routes.py`
- Contains all the route handlers for different endpoints
- Implements the main functionality of the web application:
  - Search
  - Document viewing
  - Database information display
  - Search terms analysis
  - User authentication
  - Settings management
  - CSV export

### `database_setup.py`
- Establishes connection to MongoDB
- Defines functions for CRUD operations on documents
- Implements dynamic field structure discovery and updates
- Creates necessary indexes for performance optimization

### `data_processing.py`
- Handles the ingestion of JSON files into the MongoDB database
- Processes documents to extract and store unique terms
- Updates the field structure based on ingested documents
- Implements multiprocessing for faster data ingestion

### `json_validator.py`
- Validates and cleans JSON files
- Converts .txt files to proper JSON format
- Implements multiprocessing for efficient file processing

### `json_validator_multi.py`
- Similar to `json_validator.py` but optimized for handling multiple files
- Uses multiprocessing to speed up the validation and conversion process

### `generate_password.py`
- Utility script to generate a hashed password for admin access
- Uses Werkzeug's password hashing function

### `models.py`
- This file is currently not in use as the project uses a NoSQL database
- Kept for potential future use if SQL models are needed

## Contributing

Contributions to the Historical Document Reader are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

## License

[Insert your chosen license here]

## Contact

[Your Name or Organization]
[Contact Information]

