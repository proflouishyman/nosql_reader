# datetime: 2024-10-16
# Purpose: To load and print all environment variables from a .env file using dotenv

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Print all environment variables
for key, value in os.environ.items():
    print(f"{key}: {value}")
