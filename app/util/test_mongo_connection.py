# Date: 2024-10-07
# Purpose: Test connection to MongoDB using PyMongo and print connection details.

import os
from pymongo import MongoClient
from urllib.parse import urlparse
import logging
from dotenv import load_dotenv



def test_connection():
    env_file_path  = '../.env'

    # Load environment variables from .env file
    env_path = load_dotenv(env_file_path)  # Load .env file

    print(os.path.abspath(env_file_path))

    if env_path:
        print(f"Using .env file: {env_path}")
    else:
        print("No .env file found.")

    # Retrieve the MONGO_URI from environment variables
    mongo_uri = os.environ.get('MONGO_URI')
    print(mongo_uri)




    
    # Set a default value for testing if environment variable is not set
    #mongo_uri = "mongodb://admin:secret@mongodb:27017/admin"
    print(f"Using default MONGO_URI: {mongo_uri}")

    print(mongo_uri)
    


    # # Print all environment variables for debugging
    # print("Current Environment Variables:")
    # for key, value in os.environ.items():
    #     print(f"{key}={value}")


    # Parse the MongoDB URI to extract its components
    parsed_uri = urlparse(mongo_uri)
    
    # Extract components with masking for sensitive information
    scheme = parsed_uri.scheme
    username = parsed_uri.username
    password = parsed_uri.password 
    hostname = parsed_uri.hostname
    port = parsed_uri.port
    auth_database = parsed_uri.path[1:] if parsed_uri.path else None  # Remove leading '/'
    query = parsed_uri.query

    # Display the parsed components
    print("🔍 **MongoDB Connection Details:**")
    print(f"• **Scheme:** {scheme}")
    print(f"• **Username:** {username}")
    print(f"• **Password:** {password}")
    print(f"• **Hostname:** {hostname}")
    print(f"• **Port:** {port}")
    print(f"• **Authentication Database:** {auth_database}")
    print(f"• **Options:** {query}\n")

    print(f"MONGO_URI: mongodb://{username}:****@{hostname}:{port}/{auth_database}")

    try:
        # Initialize the MongoDB client with increased timeout for reliability
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Attempt to ping the MongoDB server to test the connection
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB.")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")

if __name__ == "__main__":
    print("🚀 Testing MongoDB connection...\n")
    test_connection()
