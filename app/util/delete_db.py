from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Connect to MongoDB
mongo_uri = os.environ.get('MONGO_URI')
client = MongoClient(mongo_uri)

# Specify the database name
db_name = 'railroad_documents'

# Drop the database
client.drop_database(db_name)
print(f'Deleted database: {db_name}')
