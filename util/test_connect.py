from pymongo import MongoClient
from pprint import pprint
from tabulate import tabulate


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Adjust the connection string as needed
db = client['railroad_employees']  # Replace with your actual database name
employees = db['employees']  # Replace with your actual collection name

# Query to fetch a few records
records = employees.find().limit(1)  # Adjust the limit to fetch more or fewer records

# Print out the records in a structured way
for record in records:
    pprint({
        'ID': record.get('_id'),
        'Source': record.get('source'),
        'OCR Text': record.get('ocr_text', 'N/A'),
        'Summary': record.get('summary', 'N/A'),
        'Personal Information': {
            'Name': record.get('sections', [{}])[0].get('fields', [{}])[0].get('linked_information', {}).get('personal_information', {}).get('name', 'N/A'),
            'Date of Birth': record.get('sections', [{}])[0].get('fields', [{}])[0].get('linked_information', {}).get('personal_information', {}).get('date_of_birth', 'N/A'),
            'Social Security No.': record.get('sections', [{}])[0].get('fields', [{}])[0].get('linked_information', {}).get('personal_information', {}).get('social_security_no', 'N/A'),
        },
        'Employment Record': {
            'Document Type': record.get('sections', [{}])[0].get('fields', [{}])[0].get('linked_information', {}).get('metadata', {}).get('document_type', 'N/A'),
            'Period': record.get('sections', [{}])[0].get('fields', [{}])[0].get('linked_information', {}).get('metadata', {}).get('period', 'N/A'),
            'Context': record.get('sections', [{}])[0].get('fields', [{}])[0].get('linked_information', {}).get('metadata', {}).get('context', 'N/A'),
        }
    })

# Close the connection
client.close()

#Second test


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['railroad_employees']
employees = db['employees']

# Recursive function to print all key-value pairs in a document
def recursive_print(document, indent=0):
    """Recursively prints key-value pairs in a document."""
    for key, value in document.items():
        # Print the key with appropriate indentation
        print(' ' * indent + str(key) + ':', end=' ')
        
        if isinstance(value, dict):
            print()  # Print a new line before going deeper
            recursive_print(value, indent + 4)  # Recur for dictionaries
        elif isinstance(value, list):
            print()  # Print a new line before going deeper
            for i, item in enumerate(value):
                print(' ' * (indent + 4) + f'[{i}]:', end=' ')
                if isinstance(item, dict):
                    print()  # Print a new line before going deeper
                    recursive_print(item, indent + 8)
                else:
                    print(item)
        else:
            print(value)  # Print the value directly if it's neither dict nor list

# Query to fetch a few records
records = employees.find().limit(1)  # Adjust the limit as needed

# Print out the records recursively
for record in records:
    print(f'Record ID: {record.get("_id")}')
    recursive_print(record)
    print('-' * 40)  # Separator between records

# Close the connection
client.close()



