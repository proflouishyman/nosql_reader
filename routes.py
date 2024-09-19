from flask import request, jsonify, render_template, redirect, url_for, flash, session, abort, Response
from functools import wraps
from app import app, cache, table_options, operator_options
from database_setup import documents, find_document_by_id, find_documents, insert_document, update_document, delete_document
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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    num_search_fields = 3  # Number of search fields to display
    return render_template('index.html',
                           table_options=table_options,
                           operator_options=operator_options,
                           num_search_fields=num_search_fields)

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
@login_required
def search():
    try:
        data = request.get_json()
        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 50))
        
        query = {}
        for i in range(1, 4):
            table = data.get(f'table{i}')
            search_term = data.get(f'searchTerm{i}')
            operator = data.get(f'operator{i}')
            
            if table and search_term:
                if operator == 'NOT':
                    query[table] = {'$not': {'$regex': search_term, '$options': 'i'}}
                else:
                    if table not in query:
                        query[table] = {'$regex': search_term, '$options': 'i'}
                    elif operator == 'AND':
                        query[table] = {'$all': [query[table], {'$regex': search_term, '$options': 'i'}]}
                    elif operator == 'OR':
                        if '$or' not in query:
                            query['$or'] = []
                        query['$or'].append({table: {'$regex': search_term, '$options': 'i'}})

        total_count = documents.count_documents(query)
        search_results = find_documents(query, limit=per_page).skip((page - 1) * per_page)
        
        documents_list = list(search_results)
        for doc in documents_list:
            doc['_id'] = str(doc['_id'])

        total_pages = math.ceil(total_count / per_page)

        return jsonify({
            "documents": documents_list,
            "total_count": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page
        })

    except Exception as e:
        logger.error("An error occurred during search", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

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
    table = request.args.get('table', 'named_entities')
    
    pipeline = [
        {'$unwind': f'${table}'},
        {'$group': {'_id': f'${table}', 'count': {'$sum': 1}}},
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
    
    return render_template('search-terms.html', table_options=table_options, **data)

@app.route('/database-info')
@login_required
def database_info():
    collection_info = []
    for name, _ in table_options:
        count = documents.count_documents({name: {'$exists': True}})
        collection_info.append({
            'name': name,
            'count': count
        })
    return render_template('database-info.html', collection_info=collection_info)

@app.route('/export_csv', methods=['POST'])
@login_required
def export_csv():
    data = request.get_json()
    query = {}
    
    for field in data['fields']:
        table = field['table']
        search_term = field['searchTerm']
        operator = field['operator']
        
        if operator == 'AND':
            query[table] = {'$regex': search_term, '$options': 'i'}
        elif operator == 'OR':
            if '$or' not in query:
                query['$or'] = []
            query['$or'].append({table: {'$regex': search_term, '$options': 'i'}})
        elif operator == 'NOT':
            query[table] = {'$not': {'$regex': search_term, '$options': 'i'}}

    results = list(find_documents(query))

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'File', 'OCR Text', 'Summary'])

    for result in results:
        writer.writerow([str(result['_id']), result.get('file', ''), result.get('ocr_text', ''), result.get('summary', '')])

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