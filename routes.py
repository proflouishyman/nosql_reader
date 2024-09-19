# File: routes.py
# Path: routes.py

from flask import request, jsonify, render_template, redirect, url_for, flash, session, abort, Response
from functools import wraps
from app import app, cache
from database_setup import (
    documents,
    find_document_by_id,
    find_documents,
    insert_document,
    update_document,
    delete_document,
    get_field_structure,
    unique_terms_collection
)
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
import os

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
        login_attempts[ip] = (0, datetime.now()) if success else (1, datetime.now())

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
# @login_required
def index():
    app.logger.info('Handling request to index')
    num_search_fields = 3  # Number of search fields to display
    field_structure = get_field_structure()
    return render_template('index.html', num_search_fields=num_search_fields, field_structure=field_structure)

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

@app.route('/search', methods=['POST'])
# @login_required
def search():
    try:
        data = request.get_json()
        app.logger.debug(f"Received search request: {data}")

        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 50))

        query = build_query(data)
        app.logger.debug(f"Constructed MongoDB query: {query}")

        total_count = documents.count_documents(query)
        search_results = list(documents.find(query).skip((page - 1) * per_page).limit(per_page))

        for doc in search_results:
            doc['_id'] = str(doc['_id'])

        total_pages = math.ceil(total_count / per_page) if per_page else 1

        app.logger.debug(f"Found {total_count} documents, returning page {page} of {total_pages}")

        return jsonify({
            "documents": search_results,
            "total_count": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page
        })

    except Exception as e:
        app.logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

def build_query(data):
    query = {}
    criteria_list = []

    app.logger.debug(f"Building query from search data: {data}")

    for i in range(1, 4):
        field = data.get(f'field{i}')
        search_term = data.get(f'searchTerm{i}')
        operator = data.get(f'operator{i}')

        if field and search_term:
            condition = {}
            if operator == 'NOT':
                condition[field] = {'$not': {'$regex': re.escape(search_term), '$options': 'i'}}
            else:
                condition[field] = {'$regex': re.escape(search_term), '$options': 'i'}
            
            criteria_list.append((operator, condition))
            app.logger.debug(f"Processed field {field} with search term '{search_term}' and operator '{operator}'")

    if criteria_list:
        and_conditions = []
        or_conditions = []

        for operator, condition in criteria_list:
            if operator == 'AND' or operator == 'NOT':
                and_conditions.append(condition)
            elif operator == 'OR':
                or_conditions.append(condition)

        if and_conditions:
            query['$and'] = and_conditions

        if or_conditions:
            if '$or' not in query:
                query['$or'] = or_conditions
            else:
                query['$or'].extend(or_conditions)

    app.logger.debug(f"Final query: {query}")
    return query

@app.route('/document/<string:doc_id>')
# @login_required
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
        app.logger.error(f"Error in document_detail: {str(e)}")
        abort(500)



@app.route('/search-terms', methods=['GET'])
# @login_required
def search_terms():
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Handle AJAX request
        field = request.args.get('field')
        app.logger.debug(f"AJAX request for field: {field}")
        if not field:
            return jsonify({"error": "No field specified"}), 400

        # Fetch unique terms for the specified field from unique_terms_collection
        unique_terms_doc = unique_terms_collection.find_one({"field": field})
        if not unique_terms_doc:
            app.logger.debug(f"No terms found for field '{field}'.")
            return jsonify({
                "words": [],
                "phrases": [],
                "unique_words": 0,
                "unique_phrases": 0,
                "total_records": documents.count_documents({}),
                "message": f"No terms found for field '{field}'."
            }), 200  # Changed status to 200

        words = unique_terms_doc.get('words', {})
        phrases = unique_terms_doc.get('phrases', {})
        unique_words_count = len(words)
        unique_phrases_count = len(phrases)
        total_records = documents.count_documents({})

        # Convert words and phrases to lists of dictionaries
        words_list = [{'word': word, 'count': count} for word, count in sorted(words.items())]
        phrases_list = [{'phrase': phrase, 'count': count} for phrase, count in sorted(phrases.items())]

        data = {
            'words': words_list,
            'phrases': phrases_list,
            'unique_words': unique_words_count,
            'unique_phrases': unique_phrases_count,
            'total_records': total_records
        }

        return jsonify(data)
    else:
        # Render the HTML template
        field_structure = get_field_structure()
        return render_template('search-terms.html', field_structure=field_structure)

    

@app.route('/database-info')
# @login_required
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

@app.route('/settings', methods=['GET', 'POST'])
# @login_required
def settings():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    if request.method == 'POST':
        new_config = request.form.to_dict()

        for key in ['fonts', 'sizes', 'colors', 'spacing']:
            if key in new_config:
                try:
                    new_config[key] = json.loads(new_config[key])
                except json.JSONDecodeError:
                    flash(f"Invalid JSON format for {key}.", 'danger')
                    return redirect(url_for('settings'))

        try:
            with open(config_path, 'w') as config_file:
                json.dump(new_config, config_file, indent=4)
            app.config['UI_CONFIG'] = new_config
            flash('Settings updated successfully', 'success')
        except Exception as e:
            app.logger.error(f"Error updating settings: {str(e)}")
            flash('Failed to update settings.', 'danger')
        return redirect(url_for('settings'))

    try:
        if os.path.exists(config_path):
            with open(config_path) as config_file:
                config = json.load(config_file)
        else:
            config = {}
    except json.JSONDecodeError:
        config = {}
        flash('Configuration file is corrupted. Using default settings.', 'warning')

    return render_template('settings.html', config=config)

@app.route('/export_csv', methods=['POST'])
# @login_required
def export_csv():
    data = request.get_json()
    query = build_query(data)

    results = documents.find(query)

    output = StringIO()
    writer = csv.writer(output)
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
            keys = field.split('.')
            value = result
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key, '')
                else:
                    value = ''
                    break
            # Convert lists and dictionaries to JSON strings for better readability
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
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
