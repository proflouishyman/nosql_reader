# Historical Document Reader

## Description
The Historical Document Reader is a Flask-based web application designed to manage, search, and display historical documents stored in a MongoDB database. It provides an intuitive interface for researchers and historians to access and analyze digitized historical records.

## Repository
The project is hosted on GitHub at:
https://github.com/proflouishyman/nosql_reader

## Features

- Document Management: Data ingestion from JSON files, dynamic field structure discovery
- Search Functionality: Advanced search with multiple fields and logical operators
- Document Viewing: Detailed view with image zoom and pan capabilities
- Data Analysis: Search terms analysis with word and phrase frequency
- User Interface: Customizable UI settings, responsive design
- Data Export: Export search results to CSV
- Security: Basic authentication system with CAPTCHA

## To Do

0. Shift other PC to Docker
0.1. Create flag for list of hidden fields
1. Convert setup process to a part of settings or a new page. It should be able to add to the DB
2. Create login splash page
3. Need to adjust unique search terms to move from a pickle file to part of the MongoDB
4. Address weirdness of base file and index.html. It is unseemly 
5. Reorganize routes.py
6. Add cross-referencing of named entities
7. Restore adding export from list
8. Add "Select all" for export
9. Clean file description to remove extension in title
10. Create a way to do a search and then add that result to the DB
11. Backup and restore database
12. Remove blank fields from JSON expansion
13. Color code sections of JSON expansion

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
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── document-detail.html
│   ├── document-list.html
│   ├── search-terms.html
│   ├── database-info.html
│   ├── settings.html
│   ├── login.html
│   └── error.html
│
└── archives/
    └── [Your archival files go here]
```

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/proflouishyman/nosql_reader.git
   ```

2. Navigate to the project directory and install dependencies:
   ```
   cd nosql_reader
   pip install -r requirements.txt
   ```

3. Set up MongoDB:
   - Install MongoDB if you haven't already.
   - Start the MongoDB service.
   - Update the connection string in `database_setup.py` if necessary:
     ```python
     client = MongoClient('mongodb://localhost:27017/')
     ```

4. Initialize the database structure:
   ```
   python database_setup.py
   ```

5. Prepare your JSON data:
   - Ensure your historical document data is in JSON format.
   - Place all archival files in the `archives` subdirectory.
   - If your data is in .txt files, use the JSON validator to convert and validate them:
     ```
     python json_validator.py
     ```

6. Ingest data into the database:
   - Update the `data_directory` path in `data_processing.py` to point to your JSON files in the `archives` subdirectory.
   - Run the data processing script:
     ```
     python data_processing.py
     ```

7. Generate a password for admin access (optional):
   ```
   python generate_password.py
   ```

8. Run the application:
   ```
   python app.py
   ```

## Usage

1. Access the application at `http://localhost:5000`.
2. Log in using the generated admin password.
3. Use the search interface to find documents.
4. View search results and document details.
5. Analyze search terms and view database information.
6. Customize the interface in the Settings page.
7. Export results to CSV as needed.

## Docker Setup

Run `run_docker.sh` to install and set up using Docker. If there are problems, install `dos2unix` and then convert the file. Everything runs better in WSL.

MongoDB connection string for Docker setup:
```python
client = MongoClient('mongodb://admin:secret@localhost:27017/')
```

## Maintenance and Updates

- Add new documents by placing JSON files in the `archives` subdirectory and running `data_processing.py`.
- The system automatically adapts to changes in document structure.
- Regularly backup your MongoDB database.

## Troubleshooting

- Verify JSON format for data processing issues.
- Check MongoDB service and connection string for database issues.
- Consult application logs for error messages.

## Main Python Files

- `app.py`: Main application file, initializes Flask app
- `routes.py`: Contains all route handlers and main functionality
- `database_setup.py`: Manages MongoDB connection and CRUD operations
- `data_processing.py`: Handles data ingestion and processing
- `json_validator.py` and `json_validator_multi.py`: Validate and clean JSON files
- `generate_password.py`: Utility for generating admin password

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make changes and commit with descriptive messages.
4. Push changes to your fork.
5. Submit a pull request to the main repository.

## License

[Insert your chosen license here]

## Contact

[Your Name or Organization]
[Contact Information]