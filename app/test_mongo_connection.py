import os
from pymongo import MongoClient

def test_connection():
    mongo_uri = os.environ.get('MONGO_URI')
    if not mongo_uri:
        print("MONGO_URI environment variable not set.")
        return
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")

if __name__ == "__main__":
    print("testing connection")
    test_connection()
