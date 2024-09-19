# File: routes.py
# Path: railroad_documents_project/routes.py

from flask import request, jsonify, render_template, redirect, url_for, flash, session, abort, Response
from functools import wraps
from app import app, cache
from database_setup import documents, find_document_by_id, find_documents, insert_document, update_document, delete_document, get_field_structure
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import math
import json
import re
import logging
import time
from datetime import datetime, timedelta
import random
import csv
from io import StringIO
from logging.handlers import RotatingFileHandler


# Setup file-based logging
if not app.debug:
    file_handler = RotatingFileHandler('logs/app_routes.log', maxBytes=10240, backupCount=10)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

app.logger.setLevel(logging.DEBUG)

# Hashed password (generate this using generate_password_hash('your_actual_password'))
ADMIN_PASSWORD_HASH = 'pbkdf2:sha256:260000$uxZ1Fkjt9WQCHwuN$ca37dfb41ebc26b19daf24885ebcd09f607cab85f92dcab13625627fd9ee902a'

# Login attempt tracking
MAX_ATTEMPTS = 5
LOCKOUT_TIME = 15 * 60  # 15 minutes in seconds
login_attempts = {}

def is_locked_out(ip):
    if ip in login_attempts:
        attempts, last_attempt_time = login_attempts[ip]
        if attempts >= MAX_ATTEMPTS:
            if datetime.now() - last_attempt_time < timedelta(seconds=LOCKOUT_TIME):
                return True
            else:
                login_attempts[ip] = (0, datetime.now())
    return False

def update_login_attempts(ip, success):
    if ip in login_attempts:
        attempts, _ = login_attempts[ip]
        if success:
            login_attempts[ip] = (0, datetime.now())
        else:
            login_attempts[ip] = (attempts + 1, datetime.now())
    else:
        login_attempts[ip] = (1, datetime.now()) if not success else (0, datetime.now())

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    app.logger.info('Handling request to index')
    num_search_fields = 3  # Number of search fields to display
    return render_template('index.html', num_search_fields=num_search_fields)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        ip = request.remote_addr

        if is_locked_out(ip):
            flash('Too many failed attempts. Please try again later.')
            return render_template('login.html')

        # Verify CAPTCHA
        user_captcha = request.form.get('captcha')
        correct_captcha = request.form.get('captcha_answer')
        if user_captcha != correct_captcha:
            flash('Incorrect CAPTCHA')
            return redirect(url_for('login'))

        if check_password_hash(ADMIN_PASSWORD_HASH, request.form['password']):
            session['logged_in'] = True
            update_login_attempts(ip, success=True)
            flash('You were successfully logged in')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            update_login_attempts(ip, success=False)
            time.sleep(2)  # Add a delay after failed attempt
            flash('Invalid password')

    # Generate CAPTCHA for GET requests
    captcha_num1 = random.randint(1, 10)
    captcha_num2 = random.randint(1, 10)
    captcha_answer = str(captcha_num1 + captcha_num2)

    return render_template('login.html', captcha_num1=captcha_num1, captcha_num2=captcha_num2, captcha_answer=captcha_answer)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('index'))



@app.route('/search', methods=['GET', 'POST'])
def search():
    try:
        if request.method == 'POST':
            # Log incoming request data
            data = request.get_json()
            logging.debug(f"Received POST search request: {data}")

            # Extract pagination details
            page = int(data.get('page', 1))
            per_page = int(data.get('per_page', 50))
            logging.debug(f"Pagination details - Page: {page}, Per Page: {per_page}")

            # Build query based on user input
            query = build_query(data)
            logging.debug(f"Constructed MongoDB query: {query}")

            # Query MongoDB for the matching documents
            total_count = documents.count_documents(query)
            search_results = documents.find(query).skip((page - 1) * per_page).limit(per_page)
            documents_list = list(search_results)

            logging.debug(f"Total matching documents: {total_count}, Returned documents: {len(documents_list)}")

            # Convert ObjectId to string for each document
            for doc in documents_list:
                doc['_id'] = str(doc['_id'])

            # Calculate total pages
            total_pages = math.ceil(total_count / per_page)

            logging.debug(f"Total pages: {total_pages}, Current page: {page}")

            # Render the search results page
            return render_template('search_results.html',
                                   documents=documents_list,
                                   total_count=total_count,
                                   current_page=page,
                                   total_pages=total_pages,
                                   per_page=per_page)

        else:
            # For GET requests, simply render the search form
            num_search_fields = 3
            logging.debug(f"Rendering search page with {num_search_fields} fields")
            return render_template('search.html', num_search_fields=num_search_fields)

    except Exception as e:
        logging.error(f"An error occurred during search: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

def build_query(data):
    """
    Build a MongoDB query from the search criteria provided by the user.
    """
    query = {}
    criteria_list = []

    logging.debug(f"Building query from search data: {data}")

    for i in range(1, 4):
        field = data.get(f'field{i}')
        search_term = data.get(f'searchTerm{i}')
        operator = data.get(f'operator{i}')

        if field and search_term:
            condition = {}
            if operator == 'NOT':
                condition[field] = {'$not': {'$regex': search_term, '$options': 'i'}}
            else:
                condition[field] = {'$regex': search_term, '$options': 'i'}
            
            criteria_list.append((operator, condition))
            logging.debug(f"Processed field {field} with search term '{search_term}' and operator '{operator}'")

    # Build the query based on operators (AND, OR, NOT)
    if criteria_list:
        and_conditions = []
        or_conditions = []

        for operator, condition in criteria_list:
            if operator == 'AND' or operator == 'NOT':
                and_conditions.append(condition)
            elif operator == 'OR':
                or_conditions.append(condition)

        # Add AND conditions to the query
        if and_conditions:
            query['$and'] = and_conditions

        # Add OR conditions to the query
        if or_conditions:
            if '$or' not in query:
                query['$or'] = or_conditions
            else:
                query['$or'].extend(or_conditions)

    logging.debug(f"Final query: {query}")
    return query


@app.route('/document/<string:doc_id>')
@login_required
def document_detail(doc_id):
    try:
        document = find_document_by_id(doc_id)
        if not document:
            abort(404)

        document['_id'] = str(document['_id'])

        prev_doc = documents.find_one({'_id': {'$lt': ObjectId(doc_id)}}, sort=[('_id', -1)])
        next_doc = documents.find_one({'_id': {'$gt': ObjectId(doc_id)}}, sort=[('_id', 1)])

        prev_id = str(prev_doc['_id']) if prev_doc else None
        next_id = str(next_doc['_id']) if next_doc else None

        return render_template('document-detail.html', document=document, prev_id=prev_id, next_id=next_id)
    except Exception as e:
        logger.error(f"Error in document_detail: {str(e)}")
        abort(500)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    if request.method == 'POST':
        new_config = request.form.to_dict()

        for key in ['fonts', 'sizes', 'colors', 'spacing']:
            if key in new_config:
                new_config[key] = json.loads(new_config[key])

        with open(config_path, 'w') as config_file:
            json.dump(new_config, config_file, indent=4)

        app.config['UI_CONFIG'] = new_config

        flash('Settings updated successfully', 'success')
        return redirect(url_for('settings'))

    with open(config_path) as config_file:
        config = json.load(config_file)

    return render_template('settings.html', config=config)

@app.route('/search-terms')
@login_required
def search_terms():
    field = request.args.get('field', None)
    if not field:
        return jsonify({"error": "No field specified"}), 400

    pipeline = [
        {'$unwind': f'${field}'},
        {'$group': {'_id': f'${field}', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}}
    ]

    terms = list(documents.aggregate(pipeline))

    unique_terms = len(terms)
    total_records = documents.count_documents({})

    data = {
        'terms': [{'term': str(term['_id']), 'count': term['count']} for term in terms],
        'unique_terms': unique_terms,
        'total_records': total_records
    }

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(data)

    return render_template('search-terms.html', **data)

@app.route('/database-info')
@login_required
def database_info():
    field_struct = get_field_structure()
    collection_info = []

    def count_documents_with_field(field_path):
        count = documents.count_documents({field_path: {'$exists': True}})
        return count

    def traverse_structure(structure, current_path=''):
        for field, value in structure.items():
            path = f"{current_path}.{field}" if current_path else field
            if isinstance(value, dict):
                traverse_structure(value, current_path=path)
            else:
                count = count_documents_with_field(path)
                collection_info.append({
                    'name': path,
                    'count': count
                })

    traverse_structure(field_struct)

    return render_template('database-info.html', collection_info=collection_info)

@app.route('/export_csv', methods=['POST'])
@login_required
def export_csv():
    data = request.get_json()
    query = build_query(data)

    results = documents.find(query)

    output = StringIO()
    writer = csv.writer(output)
    # Get field names from field_structure
    field_struct = get_field_structure()
    field_names = []

    def get_field_names(structure, prefix=''):
        for key, value in structure.items():
            if isinstance(value, dict):
                get_field_names(value, prefix=prefix + key + '.')
            else:
                field_names.append(prefix + key)

    get_field_names(field_struct)

    writer.writerow(['ID'] + field_names)

    for result in results:
        row = [str(result.get('_id', ''))]
        for field in field_names:
            # Use dot notation to get nested fields
            keys = field.split('.')
            value = result
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key, '')
                else:
                    value = ''
                    break
            row.append(value)
        writer.writerow(row)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=search_results.csv"}
    )

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message='An unexpected error has occurred'), 500
