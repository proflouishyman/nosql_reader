#app.py

from flask import request, jsonify, render_template, abort, Response, session, send_file, redirect, url_for
from app import app, db, cache
from models import OCRText, Summary, NamedEntity, Date, MonetaryAmount, Relationship, DocumentMetadata, Translation, FileInfo
from sqlalchemy import text, func, distinct
import os
from io import StringIO
import csv
import logging
import math
import traceback
import json
from sqlalchemy.exc import SQLAlchemyError
import re



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# File: routes.py
# Updated: 2024-08-12 10:45

from flask import request, jsonify, render_template, redirect, url_for, flash, session, abort, Response
from functools import wraps
from app import app, db, cache
from models import OCRText, Summary, NamedEntity, Date, MonetaryAmount, Relationship, DocumentMetadata, Translation, FileInfo
from sqlalchemy import or_, and_, not_, func, text
from sqlalchemy.orm import aliased
from werkzeug.security import generate_password_hash, check_password_hash
import math
import json
import re
import logging
import time
from datetime import datetime, timedelta
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Hashed password (generate this using generate_password_hash('your_actual_password'))
#@TOTO replace this hardcore with a file
ADMIN_PASSWORD_HASH = 'pbkdf2:sha256:260000$uxZ1Fkjt9WQCHwuN$ca37dfb41ebc26b19daf24885ebcd09f607cab85f92dcab13625627fd9ee902a'

# Login attempt tracking
MAX_ATTEMPTS = 5
LOCKOUT_TIME = 15 * 60  # 15 minutes in seconds
login_attempts = {}

def is_locked_out(ip):
    """
    Check if the given IP is locked out due to too many failed login attempts.
    """
    if ip in login_attempts:
        attempts, last_attempt_time = login_attempts[ip]
        if attempts >= MAX_ATTEMPTS:
            if datetime.now() - last_attempt_time < timedelta(seconds=LOCKOUT_TIME):
                return True
            else:
                # Reset attempts after lockout period
                login_attempts[ip] = (0, datetime.now())
    return False

def update_login_attempts(ip, success):
    """
    Update the login attempts for the given IP.
    Reset attempts on successful login, increment on failed login.
    """
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
    # Define options to pass to the template
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
    operator_options = ['AND', 'OR', 'NOT']
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


def get_model_and_column(table_name):
    model_map = {
        'ocr_text': (OCRText, 'text'),
        'summary': (Summary, 'text'),
        'named_entities': (NamedEntity, 'entity'),
        'dates': (Date, 'date'),
        'monetary_amounts': (MonetaryAmount, 'amount'),
        'relationships': (Relationship, None),
        'metadata': (DocumentMetadata, None),
        'translation': (Translation, 'french_text'),
        'file_info': (FileInfo, 'original_filepath')
    }
    return model_map.get(table_name, (None, None))

def generate_search_terms(table_name):
    try:
        model, column = get_model_and_column(table_name)
        if model is None:
            raise ValueError(f"Invalid table name: {table_name}")

        yield "Starting search terms generation..."

        if table_name == 'relationships':
            query = db.session.query(
                func.concat(Relationship.entity1, ' ', Relationship.relationship, ' ', Relationship.entity2).label('term'),
                func.count().label('count')
            ).group_by(Relationship.entity1, Relationship.relationship, Relationship.entity2)
        elif table_name == 'metadata':
            query = db.session.query(
                DocumentMetadata.document_type.label('term'),
                func.count().label('count')
            ).group_by(DocumentMetadata.document_type).union_all(
                db.session.query(
                    DocumentMetadata.period.label('term'),
                    func.count().label('count')
                ).group_by(DocumentMetadata.period)
            ).union_all(
                db.session.query(
                    DocumentMetadata.context.label('term'),
                    func.count().label('count')
                ).group_by(DocumentMetadata.context)
            ).union_all(
                db.session.query(
                    DocumentMetadata.sentiment.label('term'),
                    func.count().label('count')
                ).group_by(DocumentMetadata.sentiment)
            )
        else:
            query = db.session.query(
                getattr(model, column).label('term'),
                func.count().label('count')
            ).group_by(getattr(model, column))

        yield "Executing database query..."
        results = query.all()
        yield f"Query complete. Processing {len(results)} results..."

        terms = []
        for i, row in enumerate(results):
            terms.append({'term': row.term, 'count': row.count})
            if i % 1000 == 0:  # Update progress every 1000 items
                yield f"Processed {i+1} of {len(results)} terms..."

        yield "Calculating unique terms and total records..."
        unique_terms = db.session.query(func.count(distinct(model.id))).scalar()
        total_records = db.session.query(func.count(model.id)).scalar()

        data = {
            'terms': terms,
            'unique_terms': unique_terms,
            'total_records': total_records
        }

        yield f"Generation complete. {len(terms)} terms processed."
        return data

    except Exception as e:
        app.logger.error(f"Error generating search terms for {table_name}: {str(e)}")
        app.logger.error(traceback.format_exc())
        yield f"Error: {str(e)}"
        raise

@app.route('/generate-search-terms')
def generate_search_terms_route():
    table_name = request.args.get('table', 'named_entities')
    
    def generate():
        for update in generate_search_terms(table_name):
            yield f"data: {json.dumps({'message': update})}\n\n"
        
        # After generation is complete, yield the final data
        static_terms_dir = os.path.join(app.static_folder, 'search_terms')
        json_file = os.path.join(static_terms_dir, f'{table_name}_terms.json')
        with open(json_file, 'r') as f:
            final_data = json.load(f)
        yield f"data: {json.dumps({'final_data': final_data})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/search-terms', methods=['GET'])
@login_required
def search_terms():
    try:
        selected_table = request.args.get('table', 'named_entities')
        app.logger.info(f"Requested search terms for table: {selected_table}")

        # Path to the static JSON files
        static_terms_dir = os.path.join(app.static_folder, 'search_terms')
        json_file = os.path.join(static_terms_dir, f'{selected_table}_terms.json')
        
        if not os.path.exists(json_file):
            app.logger.error(f"Search terms file not found for {selected_table}")
            return jsonify({"error": "Search terms not available for this table"}), 404
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Return JSON response for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(data)
        
        # Render the template for initial page load
        return render_template('search-terms.html', **data)
    except Exception as e:
        app.logger.error(f"Error in search_terms route: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/database-info')
@login_required
def database_info():
    table_info = []
    for table in [OCRText, Summary, NamedEntity, Date, MonetaryAmount, Relationship, DocumentMetadata, Translation, FileInfo]:
        count = db.session.query(func.count()).select_from(table).scalar()
        table_info.append({
            'name': table.__tablename__,
            'count': count
        })
    return render_template('database-info.html', table_info=table_info)


@app.route('/search', methods=['POST'])
@login_required
def search():
    try:
        logger.debug("Received search request")

        data = request.get_json()
        logger.debug(f"Request data: {data}")

        # Extract parameters with defaults to avoid KeyError
        table1 = data.get('table1', '')
        table2 = data.get('table2', '')
        table3 = data.get('table3', '')
        search_term1 = data.get('searchTerm1', '')
        operator1 = data.get('operator1', 'AND')
        search_term2 = data.get('searchTerm2', '')
        operator2 = data.get('operator2', 'AND')
        search_term3 = data.get('searchTerm3', '')
        operator3 = data.get('operator3', 'AND')
        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 50))

        logger.debug(f"Parameters - Page: {page}, Per Page: {per_page}")

        # Define the columns to search in each table based on the schema
        table_columns = {
            'ocr_text': ['text'],
            'summary': ['text'],
            'named_entities': ['entity', 'type'],
            'dates': ['date'],
            'monetary_amounts': ['amount', 'category'],
            'relationships': ['entity1', 'relationship', 'entity2'],
            'metadata': ['document_type', 'period', 'context', 'sentiment'],
            'translation': ['french_text', 'english_translation'],
            'file_info': ['original_filepath']
        }

        # Define the join conditions for each table
        table_joins = {
            'ocr_text': {'alias': 't1', 'join_condition': None},
            'summary': {'alias': 's', 'join_condition': 't1.id = s.file'},
            'named_entities': {'alias': 'ne', 'join_condition': 't1.id = ne.file'},
            'dates': {'alias': 'd', 'join_condition': 't1.id = d.file'},
            'monetary_amounts': {'alias': 'ma', 'join_condition': 't1.id = ma.file'},
            'relationships': {'alias': 'r', 'join_condition': 't1.id = r.file'},
            'metadata': {'alias': 'm', 'join_condition': 't1.id = m.file'},
            'translation': {'alias': 'tr', 'join_condition': 't1.id = tr.file'},
            'file_info': {'alias': 'fi', 'join_condition': 't1.id = fi.file'}
        }

        def build_search_condition(table, alias, search_term):
            columns = table_columns.get(table, [])
            conditions = []

            # Extract quoted phrases
            quoted_phrases = re.findall(r'"([^"]*)"', search_term)
            # Remove quoted phrases from the original search term
            remaining_terms = re.sub(r'"[^"]*"', '', search_term).split()

            for col in columns:
                # Add conditions for quoted phrases (exact match)
                for phrase in quoted_phrases:
                    param_name = f"{alias}_{col}_phrase_{phrase}"
                    conditions.append(f"{alias}.{col} LIKE :{param_name}")
                    params[param_name] = f"%{phrase}%"
                # Add conditions for remaining terms (partial match)
                for term in remaining_terms:
                    term = term.strip()
                    if term:
                        param_name = f"{alias}_{col}_term_{term}"
                        conditions.append(f"{alias}.{col} LIKE :{param_name}")
                        params[param_name] = f"%{term}%"

            return f"({' OR '.join(conditions)})" if conditions else None

        tables = [
            (table1, search_term1, operator1),
            (table2, search_term2, operator2),
            (table3, search_term3, operator3)
        ]

        query = "SELECT DISTINCT t1.id, t1.file, s.text as summary FROM ocr_text t1 "
        query += "LEFT JOIN summary s ON t1.id = s.file "
        table_conditions = []
        params = {}

        # Keep track of the aliases used to avoid duplicates
        used_aliases = set(['t1', 's'])

        for i, (table, search_term, operator) in enumerate(tables):
            if table and search_term:
                alias = table_joins[table]['alias']
                join_condition = table_joins[table]['join_condition']

                # Add the join if not already joined
                if alias not in used_aliases and join_condition:
                    query += f"LEFT JOIN {table} {alias} ON {join_condition} "
                    used_aliases.add(alias)

                condition = build_search_condition(table, alias, search_term)
                if condition:
                    if operator == 'NOT':
                        condition = f"NOT {condition}"
                    elif operator == 'OR' and table_conditions:
                        condition = f"OR {condition}"
                    else:
                        condition = f"AND {condition}"

                    table_conditions.append(condition)

        # Build the WHERE clause
        if table_conditions:
            combined_condition = ' '.join(table_conditions).lstrip('AND ')
            query += f" WHERE {combined_condition}"
        query += " ORDER BY t1.id"

        logger.debug(f"Constructed query: {query}")
        logger.debug(f"Parameters: {params}")

        # Count total results without pagination
        count_query = f"SELECT COUNT(DISTINCT t1.id) FROM ocr_text t1 "
        count_query += "LEFT JOIN summary s ON t1.id = s.file "
        used_aliases = set(['t1', 's'])

        for i, (table, search_term, operator) in enumerate(tables):
            if table and search_term:
                alias = table_joins[table]['alias']
                join_condition = table_joins[table]['join_condition']
                if alias not in used_aliases and join_condition:
                    count_query += f"LEFT JOIN {table} {alias} ON {join_condition} "
                    used_aliases.add(alias)
        if table_conditions:
            count_query += f" WHERE {combined_condition}"

        logger.debug(f"Constructed count query: {count_query}")

        # Execute the count query
        total_count_result = db.session.execute(text(count_query), params)
        total_count = total_count_result.scalar()
        logger.debug(f"Total count: {total_count}")

        # Apply pagination
        offset = (page - 1) * per_page
        paginated_query = query + f" LIMIT {per_page} OFFSET {offset}"
        logger.debug(f"Paginated query: {paginated_query}")

        # Execute the paginated query
        result = db.session.execute(text(paginated_query), params)
        documents = [{"id": row.id, "file": row.file, "summary": row.summary} for row in result]
        logger.debug(f"Retrieved {len(documents)} documents")

        total_pages = math.ceil(total_count / per_page)

        # Return JSON response
        response = jsonify({
            "documents": documents,
            "total_count": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page
        })
        logger.debug(f"Returning response: {response.get_data(as_text=True)}")
        return response

    except Exception as e:
        logger.error("An error occurred during search", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500



@app.route('/document/<string:doc_id>')
@login_required
def document_detail(doc_id):
    try:
        ocr_text = OCRText.query.get_or_404(doc_id)
        summary = Summary.query.filter_by(file=doc_id).first()
        named_entities = NamedEntity.query.filter_by(file=doc_id).all()
        dates = Date.query.filter_by(file=doc_id).all()
        monetary_amounts = MonetaryAmount.query.filter_by(file=doc_id).all()
        relationships = Relationship.query.filter_by(file=doc_id).all()
        document_metadata = DocumentMetadata.query.filter_by(file=doc_id).first()
        translation = Translation.query.filter_by(file=doc_id).first()
        file_info = FileInfo.query.filter_by(file=doc_id).first()

        results = session.get('search_results', [])
        current_index = next((index for (index, d) in enumerate(results) if d["id"] == doc_id), None)
        
        prev_id = next_id = prev_file_id = next_file_id = None

        if current_index is not None:
            if current_index > 0:
                prev_id = results[current_index - 1]["id"]
            if current_index < len(results) - 1:
                next_id = results[current_index + 1]["id"]

        # New code to get next/previous JSON file in directory
        if file_info and file_info.original_filepath:
            try:
                # Get the directory and current file name
                directory = os.path.dirname(file_info.original_filepath)
                current_file = os.path.basename(file_info.original_filepath)
                
                # Change the extension to .json
                current_json = os.path.splitext(current_file)[0] + '.json'
                
                # Get all JSON files in the directory and sort them
                json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
                
                app.logger.debug(f"Directory: {directory}")
                app.logger.debug(f"Current file: {current_file}")
                app.logger.debug(f"Current JSON: {current_json}")
                app.logger.debug(f"JSON files in directory: {json_files}")
                
                if current_json in json_files:
                    current_file_index = json_files.index(current_json)
                    app.logger.debug(f"Current file index: {current_file_index}")
                    
                    prev_file = json_files[current_file_index - 1] if current_file_index > 0 else None
                    next_file = json_files[current_file_index + 1] if current_file_index < len(json_files) - 1 else None
                    
                    app.logger.debug(f"Previous file: {prev_file}")
                    app.logger.debug(f"Next file: {next_file}")
                    
                    if prev_file:
                        prev_file_info = FileInfo.query.filter_by(original_filepath=os.path.join(directory, prev_file)).first()
                        prev_file_id = prev_file_info.file if prev_file_info else prev_file  # Use filename if not in database
                    
                    if next_file:
                        next_file_info = FileInfo.query.filter_by(original_filepath=os.path.join(directory, next_file)).first()
                        next_file_id = next_file_info.file if next_file_info else next_file  # Use filename if not in database

                    app.logger.debug(f"Previous file ID: {prev_file_id}")
                    app.logger.debug(f"Next file ID: {next_file_id}")
                else:
                    app.logger.warning(f"Current JSON file {current_json} not found in directory {directory}")
            except OSError as e:
                app.logger.error(f"Error accessing directory: {str(e)}")
            except Exception as e:
                app.logger.error(f"Unexpected error in file navigation: {str(e)}")

        return render_template('document-detail.html',
                           ocr_text=ocr_text,
                           summary=summary,
                           named_entities=named_entities,
                           dates=dates,
                           monetary_amounts=monetary_amounts,
                           relationships=relationships,
                           document_metadata=document_metadata,
                           translation=translation,
                           file_info=file_info,
                           prev_id=prev_id,
                           next_id=next_id,
                           prev_file_id=prev_file_id,
                           next_file_id=next_file_id)
    except SQLAlchemyError as e:
        app.logger.error(f"Database error: {str(e)}")
        abort(500)
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        abort(500)




@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if request.method == 'POST':
        # Update the configuration
        new_config = request.form.to_dict()
        
        # Convert nested dictionaries
        for key in ['fonts', 'sizes', 'colors', 'spacing']:
            if key in new_config:
                new_config[key] = json.loads(new_config[key])
        
        # Write the new configuration to the file
        with open(config_path, 'w') as config_file:
            json.dump(new_config, config_file, indent=4)
        
        # Update the app config
        app.config['UI_CONFIG'] = new_config
        
        return redirect(url_for('settings'))
    
    # Read the current configuration
    with open(config_path) as config_file:
        config = json.load(config_file)
    
    return render_template('settings.html', config=config)


@app.route('/image/<path:filename>')
@login_required
def serve_image(filename):
    try:
        _, extension = os.path.splitext(filename)
        mimetype = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif'
        }.get(extension.lower(), 'application/octet-stream')
        return send_file(filename, mimetype=mimetype)
    except FileNotFoundError:
        abort(404)

@app.route('/export_csv', methods=['POST'])
@login_required
def export_csv():
    data = request.get_json()
    query = db.session.query(OCRText.id, OCRText.file, OCRText.text, Summary.text.label('summary'))

    table_models = {
        'ocr_text': OCRText,
        'summary': Summary,
        'named_entities': NamedEntity,
        'dates': Date,
        'monetary_amounts': MonetaryAmount,
        'relationships': Relationship,
        'metadata': DocumentMetadata,
        'translation': Translation,
        'file_info': FileInfo
    }

    for field in data['fields']:
        table_name = field['table']
        if table_name not in table_models:
            continue

        model = table_models[table_name]
        if field['operator'] == 'AND':
            query = query.filter(model.text.ilike(f'%{field["searchTerm"]}%'))
        elif field['operator'] == 'OR':
            query = query.filter(or_(model.text.ilike(f'%{field["searchTerm"]}%')))
        elif field['operator'] == 'NOT':
            query = query.filter(not_(model.text.ilike(f'%{field["searchTerm"]}%')))

    query = query.join(Summary, OCRText.id == Summary.file)

    results = query.all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'File', 'OCR Text', 'Summary'])

    for result in results:
        writer.writerow([result.id, result.file, result.text, result.summary])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=search_results.csv"}
    )

@app.route('/documents')
@login_required
def list_documents():
    documents = OCRText.query.all()
    return render_template('document-list.html', documents=documents)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message='An unexpected error has occurred'), 500



