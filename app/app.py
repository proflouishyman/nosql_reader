# File: app.py
# Path: railroad_documents_project/app.py

import os
import json
from flask import Flask
from flask_caching import Cache
from flask_session import Session
from database_setup import get_client, get_db, get_collections, get_field_structure
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)



# Setup console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the app's logger
app.logger.addHandler(console_handler)

# # Setup file-based logging
# if not app.debug:
#     file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
#     file_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     app.logger.addHandler(file_handler)

app.logger.setLevel(logging.DEBUG)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.json')

try:
    with open(config_path) as config_file:
        config = json.load(config_file)
    app.config['UI_CONFIG'] = config
except FileNotFoundError:
    app.logger.error(f"Configuration file not found at {config_path}.")
    config = {}
except json.JSONDecodeError as e:
    app.logger.error(f"Error decoding JSON from {config_path}: {e}")
    config = {}


# Add config to app config
app.config['UI_CONFIG'] = config

# Print out the template folder path for debugging
print(f"Template folder path: {app.template_folder}")

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'historical_document_reader'

def get_secret_key():
    secret_file = os.path.join(app.root_path, 'secret_key.txt')
    if os.path.exists(secret_file):
        with open(secret_file, 'r') as f:
            return f.read().strip()
    else:
        import secrets
        generated_key = secrets.token_hex(16)
        with open(secret_file, 'w') as f:
            f.write(generated_key)
        return generated_key

# Set the secret key
app.secret_key = get_secret_key()

# Initialize extensions
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)
Session(app)

def load_config():
    """
    Load the configuration from the JSON file.
    This function is called on each request to allow for dynamic UI configuration.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path) as config_file:
        return json.load(config_file)

@app.context_processor
def inject_ui_config():
    """
    Inject the UI configuration and field structure into all templates.
    This allows for dynamic UI customization without needing to pass the config to each template.
    """
    app.config['UI_CONFIG'] = load_config()
    client = get_client()
    field_struct = get_field_structure(client)
    return dict(ui_config=app.config['UI_CONFIG'], field_structure=field_struct)

# Import routes after initializing app to avoid circular imports
from routes import *

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
