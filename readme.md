# Historical Document Reader
v0.1

## Description
The Historical Document Reader is a Flask-based web application designed to manage, search, and display historical documents stored in a MongoDB database. It provides an intuitive interface for researchers and historians to access and analyze digitized historical records.

## Repository
The project is hosted on GitHub at:
https://github.com/proflouishyman/nosql_reader

## Features

- Document Management: Data ingestion from JSON files, dynamic field structure discovery
- Search Functionality: Advanced search with multiple fields and logical operators
- Document Viewing: Detailed view with image zoom, pan, and enhancement capabilities
- Data Analysis: Search terms analysis with word and phrase frequency
- User Interface: Customizable UI settings, responsive design
- Data Export: Export search results to CSV
- Security: Basic authentication system with CAPTCHA
- Docker Support: Containerized application setup

## File Structure

```
railroad_documents_project/
├── app/
│   ├── app.py
│   ├── routes.py
│   ├── models.py
│   ├── database_setup.py
│   ├── data_processing.py
│   ├── json_validator.py
│   ├── json_validator_multi.py
│   ├── generate_password.py
│   ├── chunk_utils.py
│   ├── generate_unique_terms.py
│   ├── test_mongo_connection.py
│   ├── requirements.txt
│   ├── config.json
│   ├── secret_key.txt
│   ├── README.md
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── document-detail.html
│       ├── document-list.html
│       ├── search-terms.html
│       ├── database-info.html
│       ├── settings.html
│       ├── login.html
│       └── error.html
├── mongo-init/
│   └── init_script.js
├── entrypoint.sh
├── Dockerfile
├── docker-compose.yml
├── .env
├── .dockerignore
└── README.md
└── archives/
    └── [Your document files go here]


```

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/proflouishyman/nosql_reader.git
   ```

2. Set up the environment:
   - Copy the `.env.example` file to `.env` and update the variables as needed.
   - Ensure Docker and Docker Compose are installed on your system.

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Access the application at `http://localhost:5000`.

## Usage

1. Log in using the generated admin password (see `generate_password.py`).
2. Use the search interface to find documents.
3. View search results and document details.
4. Analyze search terms and view database information.
5. Customize the interface in the Settings page.
6. Export results to CSV as needed.

## Docker Setup

The application is containerized using Docker. The `docker-compose.yml` file defines the services:
- `flask_app`: The main Flask application
- `mongodb`: The MongoDB database
- `mongo-express`: A web-based MongoDB admin interface (optional)

To start the application:
```
docker-compose up
```

To rebuild the containers after making changes:
```
docker-compose up --build
```

## Maintenance and Updates

- Add new documents by placing JSON files in the `app/archives` directory and running `data_processing.py` within the container.
- The system automatically adapts to changes in document structure.
- Use `generate_unique_terms.py` to update the unique terms collection.
- Regularly backup your MongoDB database using MongoDB's backup tools.

## Troubleshooting

- For database connection issues, use `test_mongo_connection.py` to verify the connection.
- Check Docker logs for each service:
  ```
  docker-compose logs flask_app
  docker-compose logs mongodb
  ```
- Verify JSON format using `json_validator.py` or `json_validator_multi.py` for batch processing.
-to access the shell
'''
docker compose exec -it flask_app /bin/bash
'''

-for mongodb
'''
docker compose exec -it mongodb /bin/bash
mongosh mongodb://admin:secret@localhost:27017/admin
'''


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