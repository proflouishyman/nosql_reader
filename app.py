import os
from flask import Flask
from flask_caching import Cache
from flask_session import Session
import json
from database_setup import client, db, documents

app = Flask(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path) as config_file:
    config = json.load(config_file)

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
    Inject the UI configuration into all templates.
    This allows for dynamic UI customization without needing to pass the config to each template.
    """
    app.config['UI_CONFIG'] = load_config()
    return dict(ui_config=app.config['UI_CONFIG'])

# Define table options
table_options = [
    ('ocr_text', 'OCR Text'),
    ('summary', 'Summary'),
    ('named_entities', 'Named Entities'),
    ('dates', 'Dates'),
    ('monetary_amounts', 'Monetary Amounts'),
    ('relationships', 'Relationships'),
    ('metadata', 'Metadata'),
    ('translation', 'Translation'),
    ('file_info', 'File Info')
]

# Define operator options
operator_options = ['AND', 'OR', 'NOT']

# Import routes after initializing app to avoid circular imports
from routes import *

if __name__ == '__main__':
    # Run the app
    app.run(debug=False)