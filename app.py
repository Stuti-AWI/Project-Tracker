from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config, DevelopmentConfig, ProductionConfig, ReplitConfig
import os
from flask_mail import Mail, Message
import secrets
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
import re
from urllib.parse import urlparse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Force reload of environment variables

# Initialize OpenAI client with API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")
client = OpenAI(api_key=openai_api_key)

app = Flask(__name__)
# Database configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Force reload of environment variables

# Use Replit PostgreSQL connection if available
if 'REPLIT_DB_URL' in os.environ:
    # Use Replit's database URL if available
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('REPLIT_DB_URL')
elif 'DATABASE_URL' in os.environ:
    # Parse the DATABASE_URL to add SSL mode if not present
    db_url = os.environ['DATABASE_URL']
    parsed_url = urlparse(db_url)
    
    # If SSL mode is not specified in the URL, add it
    if 'sslmode=' not in db_url:
        if '?' in db_url:
            db_url += '&sslmode=require'
        else:
            db_url += '?sslmode=require'
    
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    # Use local development database as fallback
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project_tracker.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change this to a secure secret key
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_ADDRESS')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')
db = SQLAlchemy(app)
mail = Mail(app)

# Import Supabase config for additional features
supabase = None
try:
    from supabase_config import supabase
except ImportError:
    print("Supabase configuration not found, continuing without Supabase features")
    pass

# Add this decorator definition at the top of your file, after imports
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin', False):
            flash('This operation requires admin privileges', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Add User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Define the Sample model
class Sample(db.Model):
    __tablename__ = 'sample'
    id = db.Column(db.String(100), primary_key=True)
    company_name = db.Column(db.String(200), nullable=False)
    ERB = db.Column(db.Text, nullable=True)
    ERB_description = db.Column(db.Text, nullable=True)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    am_pm = db.Column(db.String(2), nullable=False)
    recipe_front = db.Column(db.String(200), nullable=False)
    recipe_back = db.Column(db.String(200), nullable=False)
    glass_type = db.Column(db.String(100), nullable=False)
    length = db.Column(db.Integer, nullable=False)
    thickness = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    cleaning = db.Column(db.String(1), default='N')
    coating = db.Column(db.String(1), default='N')
    annealing = db.Column(db.String(1), default='N')
    done = db.Column(db.String(1), default='N')
    # Add relationship to Experiment
    experiment = db.relationship('Experiment', backref='sample', uselist=False)

# Define the Experiment model
class Experiment(db.Model):
    __tablename__ = 'experiment'
    id = db.Column(db.String(100), db.ForeignKey('sample.id'), primary_key=True)
    transmittance = db.Column(db.String(500))
    reflectance = db.Column(db.String(500))
    absorbance = db.Column(db.String(500))
    plqy = db.Column(db.String(500))
    sem = db.Column(db.String(500))
    edx = db.Column(db.String(500))
    xrd = db.Column(db.String(500))

# Add this with your other models
class Prefix(db.Model):
    __tablename__ = 'prefix'
    prefix = db.Column(db.String(10), primary_key=True)
    full_form = db.Column(db.String(200), nullable=False)

# Modify the trash models
class SampleTrash(db.Model):
    __tablename__ = 'sample_trash'
    id = db.Column(db.String, primary_key=True)
    company_name = db.Column(db.String(200), nullable=False)
    ERB = db.Column(db.Text, nullable=True)
    ERB_description = db.Column(db.Text, nullable=True)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    am_pm = db.Column(db.String(2), nullable=False)
    recipe_front = db.Column(db.String(200), nullable=False)
    recipe_back = db.Column(db.String(200), nullable=False)
    glass_type = db.Column(db.String(100), nullable=False)
    length = db.Column(db.Integer, nullable=False)
    thickness = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    cleaning = db.Column(db.String(1), default='N')
    coating = db.Column(db.String(1), default='N')
    annealing = db.Column(db.String(1), default='N')
    done = db.Column(db.String(1), default='N')
    deleted_at = db.Column(db.DateTime, default=datetime.utcnow)
    deleted_by = db.Column(db.String(80))
    # Add relationship to ExperimentTrash
    experiment = db.relationship('ExperimentTrash', backref='sample_trash', uselist=False)

class ExperimentTrash(db.Model):
    __tablename__ = 'experiment_trash'
    id = db.Column(db.String(100), db.ForeignKey('sample_trash.id'), primary_key=True)
    transmittance = db.Column(db.String(500))
    reflectance = db.Column(db.String(500))
    absorbance = db.Column(db.String(500))
    plqy = db.Column(db.String(500))
    sem = db.Column(db.String(500))
    edx = db.Column(db.String(500))
    xrd = db.Column(db.String(500))
    deleted_at = db.Column(db.DateTime, default=datetime.utcnow)
    deleted_by = db.Column(db.String(80))

# Add these routes at the beginning of your app
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        # Debug prints
        print(f"Login attempt for username: {username}")
        if user:
            print(f"User found, stored password hash: {user.password}")
        else:
            print("User not found")
        
        try:
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['username'] = user.username
                session['is_admin'] = user.is_admin
                print("Password check successful")
                return redirect(url_for('index'))
            else:
                print("Password check failed")
                flash('Invalid username or password', 'error')
        except Exception as e:
            print(f"Error during password check: {str(e)}")
            flash('An error occurred during login', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Modify your existing routes to require login
@app.route('/')
@login_required
def index():
    # Use func.lower() for case-insensitive sorting of company names
    samples = Sample.query\
        .order_by(db.func.lower(Sample.company_name).asc())\
        .all()
    # Additional Python-level sorting for sequence numbers
    samples.sort(key=lambda x: (
        x.company_name.lower(),
        int(x.id.split('-')[-1]) if x.id.split('-')[-1].isdigit() else float('inf')
    ))
    return render_template('index.html', samples=samples)

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_sample():
    # Get all prefixes for the dropdown
    prefixes = Prefix.query.order_by(Prefix.full_form).all()
    
    if request.method == 'POST':
        # Get form data for sample
        cleaning = 'Y' if request.form.get('cleaning') else 'N'
        coating = 'Y' if request.form.get('coating') else 'N'
        annealing = 'Y' if request.form.get('annealing') else 'N'
        done = 'Y' if all([cleaning == 'Y', coating == 'Y', annealing == 'Y']) else 'N'
        
        # Get the selected prefix and ERB number
        selected_prefix = request.form.get('company_prefix')
        erb_number = request.form.get('ERB')
        
        # Get the company name from the prefix table
        prefix_obj = Prefix.query.get(selected_prefix)
        if not prefix_obj:
            flash('Invalid company selected', 'error')
            return render_template('add.html', prefixes=prefixes)
            
        company_name = prefix_obj.full_form
        
        if not erb_number:
            flash('Please select a company and enter an ERB number', 'error')
            return render_template('add.html', prefixes=prefixes)

        # Find all samples for this company prefix
        samples = Sample.query.filter(
            Sample.id.like(f"{selected_prefix}-%")
        ).all()

        # Also check the trash table
        trash_samples = SampleTrash.query.filter(
            SampleTrash.id.like(f"{selected_prefix}-%")
        ).all()

        # Get all sequence numbers from both tables
        sequence_numbers = []
        for sample in samples + trash_samples:
            try:
                seq = int(sample.id.split('-')[-1])
                sequence_numbers.append(seq)
            except (IndexError, ValueError):
                continue

        # Get the next sequence number
        sequence_number = 1
        if sequence_numbers:
            sequence_number = max(sequence_numbers) + 1
        
        # Generate the ID in the format PREFIX-ExERB-SEQUENCE
        sample_id = f"{selected_prefix}-Ex{erb_number}-{sequence_number}"
        
        # Create new sample
        new_sample = Sample(
            id=sample_id,
            company_name=company_name,
            ERB=erb_number,
            ERB_description=request.form.get('ERB_description'),
            date=request.form['date'],
            time=request.form['time'],
            am_pm=request.form['am_pm'],
            recipe_front=request.form['recipe_front'],
            recipe_back=request.form['recipe_back'],
            glass_type=request.form['glass_type'],
            length=int(request.form['length']),
            thickness=int(request.form['thickness']),
            height=int(request.form['height']),
            cleaning=cleaning,
            coating=coating,
            annealing=annealing,
            done=done
        )
        db.session.add(new_sample)

        # Create experiment if any experiment data is provided
        if any(request.form.get(field) for field in ['transmittance', 'reflectance', 'absorbance', 
                                                    'plqy', 'sem', 'edx', 'xrd']):
            experiment = Experiment(
                id=sample_id,
                transmittance=request.form.get('transmittance'),
                reflectance=request.form.get('reflectance'),
                absorbance=request.form.get('absorbance'),
                plqy=request.form.get('plqy'),
                sem=request.form.get('sem'),
                edx=request.form.get('edx'),
                xrd=request.form.get('xrd')
            )
            db.session.add(experiment)

        db.session.commit()
        return redirect(url_for('index'))
    return render_template('add.html', prefixes=prefixes)

@app.route('/edit/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_sample(id):
    sample = Sample.query.get_or_404(id)
    prefixes = Prefix.query.order_by(Prefix.full_form).all()
    
    if request.method == 'POST':
        # Get the selected prefix and company name
        selected_prefix = request.form.get('company_prefix')
        prefix_obj = Prefix.query.get(selected_prefix)
        if not prefix_obj:
            flash('Invalid company selected', 'error')
            return render_template('edit.html', sample=sample, prefixes=prefixes)
            
        company_name = prefix_obj.full_form
        
        # Update sample fields
        sample.company_name = company_name
        sample.id = request.form['id']
        sample.date = request.form['date']
        sample.time = request.form['time']
        sample.am_pm = request.form['am_pm']
        sample.ERB = request.form.get('ERB')
        sample.ERB_description = request.form.get('ERB_description')
        sample.recipe_front = request.form['recipe_front']
        sample.recipe_back = request.form['recipe_back']
        sample.glass_type = request.form['glass_type']
        sample.length = int(request.form['length'])
        sample.thickness = int(request.form['thickness'])
        sample.height = int(request.form['height'])
        sample.cleaning = 'Y' if request.form.get('cleaning') else 'N'
        sample.coating = 'Y' if request.form.get('coating') else 'N'
        sample.annealing = 'Y' if request.form.get('annealing') else 'N'
        sample.done = 'Y' if all([sample.cleaning == 'Y', sample.coating == 'Y', sample.annealing == 'Y']) else 'N'
        
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('edit.html', sample=sample, prefixes=prefixes)

@app.route('/delete/<string:id>')
@login_required
def delete_sample(id):
    try:
        # Get the sample and its experiment
        sample = Sample.query.get_or_404(id)
        experiment = Experiment.query.get(id)
        
        # Create trash records
        sample_trash = SampleTrash(
            id=sample.id,
            company_name=sample.company_name,
            ERB=sample.ERB,
            ERB_description=sample.ERB_description,
            date=sample.date,
            time=sample.time,
            am_pm=sample.am_pm,
            recipe_front=sample.recipe_front,
            recipe_back=sample.recipe_back,
            glass_type=sample.glass_type,
            length=sample.length,
            thickness=sample.thickness,
            height=sample.height,
            cleaning=sample.cleaning,
            coating=sample.coating,
            annealing=sample.annealing,
            done=sample.done,
            deleted_by=session.get('username')
        )
        
        # First add the sample trash record
        db.session.add(sample_trash)
        
        if experiment:
            experiment_trash = ExperimentTrash(
                id=experiment.id,
                transmittance=experiment.transmittance,
                reflectance=experiment.reflectance,
                absorbance=experiment.absorbance,
                plqy=experiment.plqy,
                sem=experiment.sem,
                edx=experiment.edx,
                xrd=experiment.xrd,
                deleted_by=session.get('username')
            )
            db.session.add(experiment_trash)
            db.session.delete(experiment)
            
        # Delete the original sample
        db.session.delete(sample)
        db.session.commit()
        flash('Record moved to trash successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting record: {str(e)}', 'error')
        print(f"Error: {str(e)}")
        
    return redirect(url_for('index'))

@app.route('/experiments')
@login_required
def experiments():
    experiments = Experiment.query.all()
    return render_template('experiments.html', experiments=experiments)

@app.route('/add_experiment/<string:sample_id>', methods=['GET', 'POST'])
@login_required
def add_experiment(sample_id):
    sample = Sample.query.get_or_404(sample_id)
    if request.method == 'POST':
        def process_data(file_data):
            if not file_data:
                return None
                
            try:
                content = file_data.read().decode('utf-8')
                lines = content.strip().split('\n')
                data = []
                for line in lines:
                    values = line.strip().split(',')
                    if len(values) >= 2:
                        try:
                            x = float(values[0])
                            y = float(values[1])
                            data.append([x, y])
                        except ValueError:
                            continue
                return json.dumps(data)
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                return None

        # Process each measurement type
        experiment = Experiment(
            id=sample_id,
            transmittance=process_data(request.files.get('transmittance_file')),
            reflectance=process_data(request.files.get('reflectance_file')),
            absorbance=process_data(request.files.get('absorbance_file')),
            plqy=process_data(request.files.get('plqy_file')),
            sem=request.form.get('sem'),
            edx=request.form.get('edx'),
            xrd=request.form.get('xrd')
        )
        
        db.session.add(experiment)
        db.session.commit()
        return redirect(url_for('experiments'))
        
    return render_template('add_experiment.html', sample=sample)

@app.route('/edit_experiment/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_experiment(id):
    experiment = Experiment.query.get_or_404(id)
    if request.method == 'POST':
        experiment.transmittance = request.form['transmittance']
        experiment.reflectance = request.form['reflectance']
        experiment.absorbance = request.form['absorbance']
        experiment.plqy = request.form['plqy']
        experiment.sem = request.form['sem']
        experiment.edx = request.form['edx']
        experiment.xrd = request.form['xrd']
        db.session.commit()
        return redirect(url_for('experiments'))
    return render_template('edit_experiment.html', experiment=experiment)

@app.route('/combined_view')
@login_required
def combined_view():
    # Join Sample and Experiment tables with proper ordering
    results = db.session.query(Sample, Experiment)\
        .outerjoin(Experiment, Sample.id == Experiment.id)\
        .order_by(db.func.lower(Sample.company_name).asc())\
        .all()
    # Additional Python-level sorting for sequence numbers
    results.sort(key=lambda x: (
        x[0].company_name.lower(),
        int(x[0].id.split('-')[-1]) if x[0].id.split('-')[-1].isdigit() else float('inf')
    ))
    return render_template('combined_view.html', results=results)

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    results = None
    query = None
    error = None
    response = None
    selected_columns = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            response = get_help_message()
        else:
            try:
                # Get all data first
                all_data = db.session.query(Sample, Experiment)\
                    .outerjoin(Experiment, Sample.id == Experiment.id)\
                    .all()
                
                # Process the query dynamically
                query_type, conditions = analyze_query(query)
                
                if query_type == 'help':
                    response = get_help_message()
                elif query_type == 'error':
                    error = conditions  # In this case, conditions contains the error message
                else:
                    results = process_dynamic_query(query_type, conditions, all_data)
                    if isinstance(results, tuple):
                        results, selected_columns = results
                    
                    if not results or (isinstance(results, list) and len(results) == 0):
                        error = "No matching records found."
                        results = None
                    
            except Exception as e:
                error = f"An error occurred while processing your query: {str(e)}"
                print(f"Error in chatbot: {str(e)}")
    else:
        response = get_help_message()
    
    return render_template('chatbot.html', 
                         results=results, 
                         query=query, 
                         error=error, 
                         response=response,
                         selected_columns=selected_columns)

@app.route('/chatbot_new', methods=['GET', 'POST'])
@login_required
def chatbot_new():
    results = None
    query = None
    error = None
    response = None
    selected_columns = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            response = get_help_message()
        else:
            try:
                print(f"Processing query: {query}")  # Debug print
                
                # Get all data first
                all_data = db.session.query(Sample, Experiment)\
                    .outerjoin(Experiment, Sample.id == Experiment.id)\
                    .all()
                
                print(f"Found {len(all_data)} total records")  # Debug print
                
                # Process the query dynamically
                query_type, conditions = analyze_query(query)
                print(f"Query type: {query_type}, Conditions: {conditions}")  # Debug print
                
                if query_type == 'help':
                    response = get_help_message()
                elif query_type == 'error':
                    error = conditions  # In this case, conditions contains the error message
                else:
                    results = process_dynamic_query(query_type, conditions, all_data)
                    if isinstance(results, tuple):
                        results, selected_columns = results
                        print(f"Selected columns: {selected_columns}")  # Debug print
                    
                    print(f"Found {len(results) if results else 0} matching records")  # Debug print
                    
                    if not results or (isinstance(results, list) and len(results) == 0):
                        error = "No matching records found."
                        results = None
                    
            except Exception as e:
                import traceback
                print(f"Error in chatbot: {str(e)}")
                print(traceback.format_exc())  # Print full traceback
                error = f"An error occurred while processing your query: {str(e)}"
    else:
        response = get_help_message()
    
    return render_template('chatbot_new.html', 
                         results=results, 
                         query=query, 
                         error=error, 
                         response=response,
                         selected_columns=selected_columns)

def validate_sql_query(query):
    """Validate SQL query for security."""
    # Convert to lowercase for easier checking
    query_lower = query.lower().strip()
    
    # List of forbidden SQL commands
    forbidden_commands = [
        'drop', 'truncate', 'delete', 'update', 'insert', 'alter', 'create',
        'grant', 'revoke', 'commit', 'rollback', 'savepoint', 'lock',
        'set', 'execute', 'call', 'merge', 'rename'
    ]
    
    # Check for forbidden commands
    for command in forbidden_commands:
        if query_lower.startswith(command) or f' {command} ' in query_lower:
            raise ValueError(f"SQL command '{command}' is not allowed for security reasons")
    
    # Ensure query is SELECT only
    if not query_lower.startswith('select'):
        raise ValueError("Only SELECT queries are allowed")
    
    # Check for multiple statements
    if ';' in query_lower[:-1]:  # Allow semicolon at the end
        raise ValueError("Multiple SQL statements are not allowed")
    
    # Check for comments
    if '--' in query_lower or '/*' in query_lower:
        raise ValueError("SQL comments are not allowed")
    
    return True

def sanitize_table_access(query):
    """Ensure query only accesses allowed tables."""
    allowed_tables = {'sample', 'experiment'}
    # Simple SQL parser to extract table names
    # This is a basic implementation and might need to be enhanced
    query_lower = query.lower()
    
    # Remove any quoted strings to avoid false positives
    in_quote = False
    quote_char = None
    cleaned_query = ''
    for char in query_lower:
        if char in ["'", '"']:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char:
                in_quote = False
            continue
        if not in_quote:
            cleaned_query += char
    
    # Extract table names from FROM and JOIN clauses
    parts = cleaned_query.split()
    tables_found = set()
    
    for i, word in enumerate(parts):
        if word in ('from', 'join'):
            if i + 1 < len(parts):
                table_name = parts[i + 1].strip('()')
                if table_name not in allowed_tables:
                    raise ValueError(f"Access to table '{table_name}' is not allowed")
                tables_found.add(table_name)
    
    return True

@app.route('/chatbot_llm', methods=['GET', 'POST'])
@login_required
def chatbot_llm():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            flash('Please enter a query', 'error')
            return render_template('chatbot_llm.html')

        try:
            # Rate limiting
            if not check_rate_limit(request):
                raise ValueError("Too many requests. Please wait before trying again.")

            # Call OpenAI to convert natural language to SQL
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a SQL expert. Convert the following natural language query to a PostgreSQL query.
                    The database has these tables:
                    - sample (id, company_name, ERB, ERB_description, date, time, am_pm, recipe_front, recipe_back, glass_type, length, thickness, height, cleaning, coating, annealing, done)
                    - experiment (id [foreign key to sample.id], transmittance, reflectance, absorbance, plqy, sem, edx, xrd)
                    
                    Important security rules:
                    1. ONLY generate SELECT queries
                    2. DO NOT use multiple statements (no semicolons except at the end)
                    3. DO NOT include any comments
                    4. ONLY access the tables listed above
                    5. DO NOT use subqueries that might expose sensitive data
                    6. Use proper SQL injection prevention practices
                    7. ALWAYS use explicit column names (no SELECT *)
                    
                    Only output the SQL query, nothing else."""},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1  # Lower temperature for more consistent SQL generation
            )
            
            generated_sql = completion.choices[0].message.content.strip()
            
            # Validate and sanitize the generated SQL
            validate_sql_query(generated_sql)
            sanitize_table_access(generated_sql)
            
            # Log the query for audit purposes
            log_query(user_input, generated_sql, session.get('user_id'))
            
            try:
                # Execute the generated SQL query with timeout
                with timeout(seconds=5):  # 5 seconds timeout
                    result = db.session.execute(generated_sql)
                    results = [dict(row) for row in result]
                
                # Limit the number of returned rows
                max_rows = 1000
                if len(results) > max_rows:
                    results = results[:max_rows]
                    flash(f'Results limited to {max_rows} rows', 'warning')
                
                return render_template('chatbot_llm.html', 
                                    generated_sql=generated_sql,
                                    results=results)
                                    
            except TimeoutError:
                raise ValueError("Query execution timed out. Please simplify your query.")
                
        except ValueError as e:
            flash(f'Validation Error: {str(e)}', 'error')
            return render_template('chatbot_llm.html')
        except Exception as e:
            # Log the error but don't expose details to user
            app.logger.error(f"Error in chatbot_llm: {str(e)}")
            flash('An error occurred while processing your query. Please try again.', 'error')
            return render_template('chatbot_llm.html')
            
    return render_template('chatbot_llm.html')

# Rate limiting implementation
from datetime import datetime, timedelta
from collections import defaultdict

# Store request counts per user
request_counts = defaultdict(list)

def check_rate_limit(request):
    """
    Implement rate limiting: 10 requests per minute per user
    """
    user_id = session.get('user_id', request.remote_addr)
    now = datetime.now()
    
    # Clean up old requests
    request_counts[user_id] = [
        timestamp for timestamp in request_counts[user_id]
        if timestamp > now - timedelta(minutes=1)
    ]
    
    # Check rate limit
    if len(request_counts[user_id]) >= 10:
        return False
    
    # Add new request
    request_counts[user_id].append(now)
    return True

def log_query(user_input, generated_sql, user_id):
    """
    Log queries for audit purposes
    """
    try:
        # You might want to create a new table for this
        app.logger.info(
            f"Query Log - User: {user_id}, "
            f"Input: {user_input}, "
            f"SQL: {generated_sql}, "
            f"Timestamp: {datetime.now()}"
        )
    except Exception as e:
        app.logger.error(f"Error logging query: {str(e)}")

# Timeout context manager
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError("Query execution timed out")

    # Set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)