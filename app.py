import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_caching import Cache
from flask_session import Session
import json

app = Flask(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path) as config_file:
    config = json.load(config_file)

# Add config to app config
app.config['UI_CONFIG'] = config

# Print out the template folder path for debugging
print(f"Template folder path: {app.template_folder}")

# Database Configuration
if os.getenv('GAE_ENV', '').startswith('standard'):
    # Production configuration for Google App Engine
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        'mysql+pymysql://{user}:{password}@/{database}?unix_socket=/cloudsql/{connection_name}'
        .format(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            connection_name=os.getenv('CLOUDSQL_CONNECTION_NAME')
        )
    )
else:
    # Local development configuration
    base_dir = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, 'create_db', 'mcd_index_updated.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

# SQLAlchemy configuration
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
        # If the file doesn't exist, generate a random key and save it
        import secrets
        generated_key = secrets.token_hex(16)
        with open(secret_file, 'w') as f:
            f.write(generated_key)
        return generated_key

# Set the secret key
app.secret_key = get_secret_key()
# Initialize extensions
db = SQLAlchemy(app)
ma = Marshmallow(app)
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

# Import routes after initializing db to avoid circular imports
from routes import *

if __name__ == '__main__':
    with app.app_context():
        # Check if the database file exists and create it if it doesn't
        if not os.path.exists(db_path):
            print(f"Database file not found at {db_path}")
            print("Creating new database file...")
            db.create_all()
        else:
            print(f"Database file found at {db_path}")
    
    # Run the app
    app.run(debug=False)