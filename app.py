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

@app.route('/chatbot_llm', methods=['GET', 'POST'])
@login_required
def chatbot_llm():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            flash('Please enter a query', 'error')
            return render_template('chatbot_llm.html')

        try:
            # Call OpenAI to convert natural language to SQL
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a SQL expert. Convert the following natural language query to a PostgreSQL query. 
                    The database has these tables:
                    - sample (id, company_name, ERB, ERB_description, date, time, am_pm, recipe_front, recipe_back, glass_type, length, thickness, height, cleaning, coating, annealing, done)
                    - experiment (id [foreign key to sample.id], transmittance, reflectance, absorbance, plqy, sem, edx, xrd)
                    Only output the SQL query, nothing else."""},
                    {"role": "user", "content": user_input}
                ]
            )
            
            generated_sql = completion.choices[0].message.content.strip()
            
            # Execute the generated SQL query
            result = db.session.execute(generated_sql)
            results = [dict(row) for row in result]
            
            return render_template('chatbot_llm.html', 
                                generated_sql=generated_sql,
                                results=results)
                                
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return render_template('chatbot_llm.html')
            
    return render_template('chatbot_llm.html')

def analyze_query(query):
    query = query.lower().strip()
    
    # Check for help-related queries
    help_patterns = ['help', 'what can you do', 'show commands', 'available commands']
    if any(pattern in query for pattern in help_patterns):
        return 'help', None
    
    # Handle natural language patterns for company name queries
    company_patterns = [
        r'(?:show|find|display|get).*(?:from|by|for|where|with)\s+(?:company\s+name\s+(?:is|=)\s*["\']?([^"\']+?)["\']?|company\s+name\s*["\']?([^"\']+?)["\']?|(?:company\s+)?([^"\']+?)(?:\s+(?:where|with|and|or)|$))',
        r'company\s*name\s*(?:is|=)\s*["\']?([^"\']+?)["\']?(?:\s|$)',
    ]
    
    for pattern in company_patterns:
        match = re.search(pattern, query)
        if match:
            # Take the first non-None group
            company_name = next((g for g in match.groups() if g is not None), '').strip()
            print(f"Extracted company name: {company_name}")  # Debug print
            conditions = {'_conditions': {'company_name': company_name}, '_operator': 'AND'}
            
            # Check for additional status conditions
            status_fields = ['cleaning', 'coating', 'annealing', 'done']
            for field in status_fields:
                status_match = re.search(rf'{field}\s*(?:is|=)\s*["\']?([YNyn])["\']?', query)
                if status_match:
                    conditions['_conditions'][field] = status_match.group(1).upper()
                    if ' or ' in query:
                        conditions['_operator'] = 'OR'
            
            # Check if specific columns are requested
            if 'show' in query and any(word in query for word in ['column', 'columns']):
                columns = extract_columns(query)
                return 'combined', {'columns': columns or [], 'conditions': conditions}
            return 'status', conditions
    
    # Handle natural language patterns for ERB queries
    erb_patterns = [
        r'(?:show|find|display|get).*erb\s*(?:number|#)?\s*["\']?(\d+)["\']?',
        r'erb\s*(?:is|=)\s*["\']?(\d+)["\']?',
    ]
    
    for pattern in erb_patterns:
        match = re.search(pattern, query)
        if match:
            erb_number = match.group(1)
            conditions = {'_conditions': {'erb': erb_number}, '_operator': 'AND'}
            if 'show' in query and any(word in query for word in ['column', 'columns']):
                columns = extract_columns(query)
                return 'combined', {'columns': columns or [], 'conditions': conditions}
            return 'status', conditions
    
    # Handle natural language patterns for status queries
    status_fields = ['cleaning', 'coating', 'annealing', 'done']
    status_patterns = [
        r'(?:show|find|display|get).*(?:with|where)?\s+(\w+)\s*(?:is|=)\s*["\']?([YNyn])["\']?',
        r'(\w+)\s*(?:is|=)\s*["\']?([YNyn])["\']?',
    ]
    
    status_conditions = {}
    for pattern in status_patterns:
        matches = re.finditer(pattern, query)
        for match in matches:
            field = match.group(1).lower()
            if field in status_fields:
                value = match.group(2).upper()
                status_conditions[field] = value
    
    if status_conditions:
        operator = 'OR' if ' or ' in query else 'AND'
        conditions = {'_conditions': status_conditions, '_operator': operator}
        if 'show' in query and any(word in query for word in ['column', 'columns']):
            columns = extract_columns(query)
            return 'combined', {'columns': columns or [], 'conditions': conditions}
        return 'status', conditions
    
    # Handle column selection queries
    if 'show' in query and any(word in query for word in ['column', 'columns']):
        columns = extract_columns(query)
        if columns:
            return 'columns', columns
    
    # Handle ID queries
    id_patterns = [
        r'(?:show|find|display|get).*id\s*[="\']([^"\']+)["\']',
        r'id\s*[="\']([^"\']+)["\']',
        r'(?:show|find|display|get).*(?:with|where)?\s+id\s+(?:is|=)\s*["\']?([^"\']+?)["\']?(?:\s|$)',
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, query)
        if match:
            return 'id', {'id': match.group(1).strip()}
    
    # Check for combined queries as a last resort
    combined = analyze_combined_query(query)
    if combined:
        return 'combined', combined
    
    # If no pattern matches, check if it's a simple column request
    words = query.split()
    if len(words) <= 3 and not any(word in ['where', 'with', 'and', 'or'] for word in words):
        potential_columns = [word.strip(',') for word in words if word not in ['show', 'me', 'the', 'columns']]
        if potential_columns:
            return 'columns', potential_columns
    
    return 'error', "I couldn't understand your query. Please try using one of the example formats or ask for help."

def extract_columns(query):
    # More flexible pattern matching for column names
    patterns = [
        r'show\s+([\w\s,]+?)(?:\s+(?:for|where|$))',  # matches "show col1, col2 for..."
        r'display\s+([\w\s,]+?)(?:\s+(?:for|where|$))', # matches "display col1, col2 for..."
        r'get\s+([\w\s,]+?)(?:\s+(?:for|where|$))',   # matches "get col1, col2 for..."
        r'([\w\s,]+?)\s+columns?(?:\s+(?:for|where|$))', # matches "col1, col2 columns for..."
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            # Split by comma and clean up each column name
            columns = [col.strip().replace(' ', '_') for col in match.group(1).split(',')]
            # Remove empty strings and common words
            columns = [col for col in columns if col and col not in ['columns', 'column', 'show', 'me', 'the']]
            return columns
    
    # If no pattern matches but query contains column names, try to extract them
    words = query.lower().split()
    if 'show' in words or 'display' in words or 'get' in words:
        potential_columns = []
        for word in words:
            word = word.strip(',').replace(' ', '_')
            if word and word not in ['show', 'display', 'get', 'me', 'the', 'columns', 'column']:
                potential_columns.append(word)
        if potential_columns:
            return potential_columns
    
    return None

def analyze_combined_query(query):
    combined = {
        'columns': [],
        'conditions': {}
    }
    
    # Extract columns if specified
    if 'show' in query:
        cols = re.findall(r'show\s+([\w\s,]+?)(?:\s+for|where|$)', query)
        if cols:
            combined['columns'] = [col.strip() for col in cols[0].split(',')]
    
    # Extract conditions
    id_match = extract_id(query)
    if id_match:
        combined['conditions']['id'] = id_match
    
    status_match = extract_status(query)
    if status_match:
        combined['conditions'].update(status_match)
    
    return combined if (combined['columns'] or combined['conditions']) else None

def process_dynamic_query(query_type, conditions, all_data):
    if query_type == 'columns':
        return process_column_selection(conditions, all_data)
    elif query_type == 'id':
        return process_id_query(conditions['id'], all_data)
    elif query_type == 'status':
        return process_status_query(conditions, all_data)
    elif query_type == 'combined':
        return process_combined_query(conditions, all_data)
    return None

def process_column_selection(columns, all_data):
    valid_columns = {
        'id': lambda s, e: s.id,
        'company_name': lambda s, e: s.company_name,
        'erb': lambda s, e: s.ERB,
        'erb_description': lambda s, e: s.ERB_description,
        'date': lambda s, e: s.date,
        'time': lambda s, e: s.time,
        'recipe_front': lambda s, e: s.recipe_front,
        'recipe_back': lambda s, e: s.recipe_back,
        'glass_type': lambda s, e: s.glass_type,
        'dimensions': lambda s, e: f"{s.length}x{s.thickness}x{s.height}",
        'cleaning': lambda s, e: s.cleaning,
        'coating': lambda s, e: s.coating,
        'annealing': lambda s, e: s.annealing,
        'done': lambda s, e: s.done,
        'transmittance': lambda s, e: e.transmittance if e else None,
        'reflectance': lambda s, e: e.reflectance if e else None,
        'absorbance': lambda s, e: e.absorbance if e else None,
        'plqy': lambda s, e: e.plqy if e else None,
        'sem': lambda s, e: e.sem if e else None,
        'edx': lambda s, e: e.edx if e else None,
        'xrd': lambda s, e: e.xrd if e else None
    }
    
    # Convert all column names to lowercase for case-insensitive comparison
    # and replace spaces with underscores
    columns = [col.lower().replace(' ', '_') for col in columns]
    
    # Validate and collect the requested columns
    valid_cols = []
    for col in columns:
        if col in valid_columns:
            valid_cols.append((col, valid_columns[col]))
        else:
            print(f"Warning: Invalid column name '{col}'")
    
    # If no valid columns were found, return all data
    if not valid_cols:
        return all_data, None
    
    # Always include 'id' column for reference if not already included
    if 'id' not in [col[0] for col in valid_cols]:
        valid_cols.insert(0, ('id', valid_columns['id']))
    
    # Transform the data to include only requested columns
    transformed_data = []
    for sample, exp in all_data:
        row = {}
        for col_name, col_func in valid_cols:
            row[col_name] = col_func(sample, exp)
        transformed_data.append(row)
    
    return transformed_data, [col[0] for col in valid_cols]

def process_id_query(id_value, all_data):
    return [(sample, exp) for sample, exp in all_data if sample.id.lower() == id_value.lower()]

def process_status_query(conditions, all_data):
    print(f"Processing status query with conditions: {conditions}")  # Debug print
    filtered_data = all_data
    
    if not conditions.get('_conditions'):
        return filtered_data
    
    operator = conditions.get('_operator', 'AND')
    status_conditions = conditions.get('_conditions', {})
    
    print(f"Operator: {operator}, Status conditions: {status_conditions}")  # Debug print
    
    # Handle combined company name and status conditions
    matching_data = []
    company_name = None
    
    # First extract company name if present
    if 'company_name' in status_conditions:
        company_name = status_conditions['company_name']
        del status_conditions['company_name']  # Remove it so we can process other conditions separately
    
    # Filter by company name first if present
    if company_name:
        filtered_data = [
            (sample, exp) for sample, exp in filtered_data
            if sample.company_name.lower() == company_name.lower()
        ]
        print(f"Filtering by company name '{company_name}', found {len(filtered_data)} matches")
    
    # Then process status conditions
    if operator == 'AND':
        # All conditions must match
        for field, value in status_conditions.items():
            if field == 'erb':
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if sample.ERB == str(value)
                ]
            else:
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if str(getattr(sample, field, '')).upper() == str(value).upper()
                ]
    else:  # OR
        # Any condition can match
        for field, value in status_conditions.items():
            if field == 'erb':
                matches = [
                    (sample, exp) for sample, exp in filtered_data
                    if sample.ERB == str(value)
                ]
            else:
                matches = [
                    (sample, exp) for sample, exp in filtered_data
                    if str(getattr(sample, field, '')).upper() == str(value).upper()
                ]
            matching_data.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_data = [x for x in matching_data if not (x in seen or seen.add(x))]
    
    print(f"Returning {len(filtered_data)} filtered records")  # Debug print
    return filtered_data

def process_combined_query(conditions, all_data):
    filtered_data = all_data
    
    # Apply conditions first with AND logic
    if conditions['conditions']:
        for field, value in conditions['conditions'].items():
            if field == 'id':
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if sample.id.lower() == value.lower()
                ]
            else:
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if getattr(sample, field, '').upper() == value.upper()
                ]
    
    # Return with selected columns if specified
    if conditions['columns']:
        return process_column_selection(conditions['columns'], filtered_data)
    
    return filtered_data

def get_help_message():
    return (
        "I can help you find information about:\n\n"
        "1. View specific columns:\n"
        "   - Example: 'Show id, company name, glass type columns'\n"
        "   - Example: 'Show glass type column'\n"
        "   - Example: 'Show cleaning, coating, annealing columns'\n\n"
        "2. Search by ID:\n"
        "   - Example: 'Show me the records of ID=\"AWI001\"'\n"
        "   - Example: 'Show records with ID=\"AWI001\"'\n\n"
        "3. Search by Company Name:\n"
        "   - Example: 'Show records where company name=\"Sun Density\"'\n"
        "   - Example: 'Show transmittance data where company name=\"Sun Density\"'\n\n"
        "4. Search by ERB:\n"
        "   - Example: 'Show records where ERB=\"1\"'\n"
        "   - Example: 'Show id, erb columns where ERB=\"1\"'\n\n"
        "5. Search by process status:\n"
        "   Using AND:\n"
        "   - Example: 'Show records with cleaning=\"Y\" and coating=\"Y\"'\n"
        "   - Example: 'Show records where cleaning=\"Y\" and coating=\"Y\"'\n\n"
        "   Using OR:\n"
        "   - Example: 'Show records with cleaning=\"Y\" or coating=\"Y\"'\n"
        "   - Example: 'Show records where cleaning=\"Y\" or coating=\"Y\"'\n\n"
        "   Combined with columns:\n"
        "   - Example: 'Show id, date columns where cleaning=\"Y\" and coating=\"Y\"'\n"
        "   - Example: 'Show id, date columns where cleaning=\"Y\" or coating=\"Y\"'\n\n"
        "Available columns:\n"
        "- Basic Info: id, company name, erb\n"
        "- Document Links: erb description (URL)\n"
        "- Date/Time: date, time\n"
        "- Recipe Info: recipe front, recipe back\n"
        "- Material Info: glass type, dimensions\n"
        "- Process Info: cleaning, coating, annealing, done\n"
        "- Experimental Data: transmittance, reflectance, absorbance, plqy, sem, edx, xrd\n\n"
        "You can combine these in any way, for example:\n"
        "- 'Show id, company name, cleaning, coating columns'\n"
        "- 'Show recipe front, glass type columns'\n"
        "- 'Show transmittance, reflectance, absorbance columns'\n"
        "- 'Show transmittance data where company name=\"Sun Density\"'\n"
        "- 'Show id, date columns where cleaning=\"Y\" and coating=\"Y\"'\n"
        "- 'Show id, date columns where cleaning=\"Y\" or coating=\"Y\"'\n"
        "- 'Show id, erb columns where ERB=\"1\"'"
    )

@app.route('/prefix_table', methods=['GET', 'POST'])
@login_required
def prefix_table():
    try:
        if request.method == 'POST':
            # Handle adding new prefix
            prefix = request.form.get('prefix')
            full_form = request.form.get('full_form')
            
            if prefix and full_form:
                # Check if prefix already exists
                existing_prefix = Prefix.query.get(prefix)
                if existing_prefix:
                    flash('Prefix already exists!', 'error')
                else:
                    new_prefix = Prefix(prefix=prefix, full_form=full_form)
                    db.session.add(new_prefix)
                    db.session.commit()
                    flash('Prefix added successfully!', 'success')
                
        # Get all prefixes
        prefixes = Prefix.query.order_by(Prefix.prefix).all()
        return render_template('prefix_table.html', prefixes=prefixes)
    
    except Exception as e:
        # Log the error (you might want to use proper logging)
        print(f"Error in prefix_table: {str(e)}")
        flash('An error occurred while loading the prefix table.', 'error')
        return render_template('prefix_table.html', prefixes=[])

@app.route('/delete_prefix/<string:prefix>')
@login_required
def delete_prefix(prefix):
    try:
        prefix_entry = Prefix.query.get_or_404(prefix)
        db.session.delete(prefix_entry)
        db.session.commit()
        flash('Prefix deleted successfully!', 'success')
    except Exception as e:
        flash('Error deleting prefix!', 'error')
    return redirect(url_for('prefix_table'))

# Add this route after the login route
@app.route('/register', methods=['POST'])
@login_required
@admin_required
def register():
    name = request.form.get('name')
    prefix = request.form.get('prefix')
    erb = request.form.get('ERB')  # Get ERB from form
    erb_description = request.form.get('ERB_description')  # Get ERB description from form
    
    if name and prefix:
        new_sample = Sample(
            name=name, 
            prefix=prefix,
            ERB=erb,
            ERB_description=erb_description
        )
        db.session.add(new_sample)
        db.session.commit()
        flash('Sample registered successfully!', 'success')
    else:
        flash('Name and prefix are required!', 'error')
    
    return redirect(url_for('index'))

# Add route to view trash
@app.route('/trash')
@login_required
def view_trash():
    # Get all trash records with their deletion info
    trash_records = db.session.query(SampleTrash, ExperimentTrash)\
        .outerjoin(ExperimentTrash, SampleTrash.id == ExperimentTrash.id)\
        .order_by(SampleTrash.deleted_at.desc())\
        .all()
    return render_template('trash.html', trash_records=trash_records)

# Add route to restore from trash
@app.route('/restore/<string:id>')
@login_required
def restore_from_trash(id):
    try:
        # Get trash records
        sample_trash = SampleTrash.query.get_or_404(id)
        experiment_trash = ExperimentTrash.query.get(id)
        
        # Check if a sample with this ID already exists
        if Sample.query.get(sample_trash.id):
            flash(f'A sample with ID {sample_trash.id} already exists!', 'error')
            return redirect(url_for('view_trash'))
        
        # Restore sample
        sample = Sample(
            id=sample_trash.id,
            company_name=sample_trash.company_name,
            ERB=sample_trash.ERB,
            ERB_description=sample_trash.ERB_description,
            date=sample_trash.date,
            time=sample_trash.time,
            am_pm=sample_trash.am_pm,
            recipe_front=sample_trash.recipe_front,
            recipe_back=sample_trash.recipe_back,
            glass_type=sample_trash.glass_type,
            length=sample_trash.length,
            thickness=sample_trash.thickness,
            height=sample_trash.height,
            cleaning=sample_trash.cleaning,
            coating=sample_trash.coating,
            annealing=sample_trash.annealing,
            done=sample_trash.done
        )
        db.session.add(sample)
        
        # Restore experiment if it exists
        if experiment_trash:
            experiment = Experiment(
                id=experiment_trash.id,
                transmittance=experiment_trash.transmittance,
                reflectance=experiment_trash.reflectance,
                absorbance=experiment_trash.absorbance,
                plqy=experiment_trash.plqy,
                sem=experiment_trash.sem,
                edx=experiment_trash.edx,
                xrd=experiment_trash.xrd
            )
            db.session.add(experiment)
            db.session.delete(experiment_trash)
            
        # Delete trash records
        db.session.delete(sample_trash)
        db.session.commit()
        flash('Record restored successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error restoring record: {str(e)}', 'error')
        print(f"Error: {str(e)}")
        
    return redirect(url_for('view_trash'))

# Add these new routes for password reset
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(username=email).first()
        
        if user:
            # Generate reset token
            token = secrets.token_urlsafe(32)
            user.reset_token = token
            user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
            db.session.commit()
            
            # Send reset email
            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message('Password Reset Request',
                        sender=os.getenv('EMAIL_ADDRESS'),
                        recipients=[email])
            msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request, please ignore this email.
'''
            mail.send(msg)
            flash('Reset instructions sent to your email.', 'info')
            return redirect(url_for('login'))
        
        flash('Email address not found.', 'error')
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if user is None or user.reset_token_expiry < datetime.utcnow():
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
        else:
            user.password = generate_password_hash(password, method='pbkdf2:sha256')
            user.reset_token = None
            user.reset_token_expiry = None
            db.session.commit()
            flash('Your password has been updated.', 'success')
            return redirect(url_for('login'))
            
    return render_template('reset_password.html')

@app.route('/plots')
@login_required
def plots():
    # Query all experiments
    experiments = db.session.query(Sample, Experiment).join(Experiment).all()
    
    # Initialize empty lists for each measurement type
    plot_data = {
        'transmittance': [],
        'reflectance': [],
        'absorbance': [],
        'plqy': [],
        'sem': [],
        'edx': [],
        'xrd': []
    }
    
    # Process each experiment
    for sample, experiment in experiments:
        for measurement_type in plot_data.keys():
            data = getattr(experiment, measurement_type)
            if data:  # If data exists for this measurement
                try:
                    # Try to parse the data (assuming it's stored as a string representation of data)
                    data_dict = {
                        'id': sample.id,
                        'data': data,
                        'recipe_front': sample.recipe_front,
                        'recipe_back': sample.recipe_back,
                        'glass_type': sample.glass_type
                    }
                    plot_data[measurement_type].append(data_dict)
                except:
                    continue
    
    # Convert plot data to JSON for JavaScript
    plot_json = json.dumps(plot_data)
    
    return render_template('plots.html', plot_data=plot_json)

def update_password_hashes():
    users = User.query.all()
    for user in users:
        # Only update if the hash doesn't start with 'pbkdf2:sha256'
        if not user.password.startswith('pbkdf2:sha256'):
            # Assuming you have a default password or way to reset
            new_hash = generate_password_hash('temporary_password', method='pbkdf2:sha256')
            user.password = new_hash
    db.session.commit()

# Add a route to reset admin password
@app.route('/reset_admin', methods=['GET'])
def reset_admin():
    try:
        with app.app_context():
            # Find admin user
            admin = User.query.filter_by(username='admin').first()
            if admin:
                # Update admin password
                admin.password = generate_password_hash('admin123', method='pbkdf2:sha256')
                db.session.commit()
                return 'Admin password reset successfully to "admin123"'
            else:
                # Create new admin user
                admin_user = User(
                    username='admin',
                    password=generate_password_hash('admin123', method='pbkdf2:sha256'),
                    is_admin=True
                )
                db.session.add(admin_user)
                db.session.commit()
                return 'New admin user created with password "admin123"'
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    # Get port from Replit environment if available
    port = int(os.environ.get('PORT', 5111))  # Using port 5000 as default
    
    with app.app_context():
        # Create tables
        db.create_all()
        print("Database tables created successfully!")
        
        # Clear existing users and create fresh admin user
        try:
            User.query.delete()
            db.session.commit()
            print("Cleared existing users")
            
            # Create fresh admin user
            admin_user = User(
                username='admin',
                password=generate_password_hash('admin123', method='pbkdf2:sha256'),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Created fresh admin user with password: admin123")
            
        except Exception as e:
            print(f"Error during user setup: {str(e)}")
            db.session.rollback()
    
    # Use Replit's host and port
    app.run(
        host='0.0.0.0',
        port=port,
        debug='development' in os.environ.get('PYTHON_ENV', '').lower()
    )