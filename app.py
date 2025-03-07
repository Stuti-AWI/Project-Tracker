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

app = Flask(__name__)
# Database configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use Replit PostgreSQL connection if available
if 'DATABASE_URL' in os.environ:
    # Use Replit PostgreSQL
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
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
try:
    from supabase_config import supabase
except ImportError:
    supabase = None

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
    password = db.Column(db.String(120), nullable=False)
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
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    am_pm = db.Column(db.String(2), nullable=False)
    ERB = db.Column(db.Text, nullable=True)
    ERB_description = db.Column(db.Text, nullable=True)
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
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    am_pm = db.Column(db.String(2), nullable=False)
    ERB = db.Column(db.Text, nullable=True)
    ERB_description = db.Column(db.Text, nullable=True)
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
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Modify your existing routes to require login
@app.route('/')
@login_required
def index():
    samples = Sample.query.all()
    return render_template('index.html', samples=samples)

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_sample():
    if request.method == 'POST':
        # Get form data for sample
        cleaning = 'Y' if request.form.get('cleaning') else 'N'
        coating = 'Y' if request.form.get('coating') else 'N'
        annealing = 'Y' if request.form.get('annealing') else 'N'
        done = 'Y' if all([cleaning == 'Y', coating == 'Y', annealing == 'Y']) else 'N'
        
        # Create new sample
        new_sample = Sample(
            id=request.form['id'],
            date=request.form['date'],
            time=request.form['time'],
            am_pm=request.form['am_pm'],
            ERB=request.form.get('ERB'),  # Add ERB field
            ERB_description=request.form.get('ERB_description'),  # Add ERB description field
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
                id=request.form['id'],
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
    return render_template('add.html')

@app.route('/edit/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_sample(id):
    sample = Sample.query.get_or_404(id)
    if request.method == 'POST':
        sample.id = request.form['id']
        sample.date = request.form['date']
        sample.time = request.form['time']
        sample.am_pm = request.form['am_pm']
        sample.ERB = request.form.get('ERB')  # Add ERB field
        sample.ERB_description = request.form.get('ERB_description')  # Add ERB description field
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
    return render_template('edit.html', sample=sample)

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
            date=sample.date,
            time=sample.time,
            am_pm=sample.am_pm,
            ERB=sample.ERB,
            ERB_description=sample.ERB_description,
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
    # Join Sample and Experiment tables
    results = db.session.query(Sample, Experiment)\
        .outerjoin(Experiment, Sample.id == Experiment.id)\
        .all()
    return render_template('combined_view.html', results=results)

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    results = None
    query = None
    error = None
    response = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip().lower()
        try:
            # Get all data first
            all_data = db.session.query(Sample, Experiment)\
                .outerjoin(Experiment, Sample.id == Experiment.id)\
                .all()
            
            # Split query into conditions using AND/OR
            conditions = parse_query_conditions(query)
            if conditions:
                results = process_conditions(conditions, all_data)
            else:
                response = "I can help you find information about:\n" + \
                          "- Use AND/OR for multiple conditions\n" + \
                          "Examples:\n" + \
                          "- 'show samples with cleaning done AND coating done'\n" + \
                          "- 'show samples with length > 100 OR thickness > 50'\n" + \
                          "- 'show samples with glass type=\"X\" AND recipe front contains \"Y\"'\n" + \
                          "- 'show samples with transmittance data AND done=Y'"

            if results is not None and len(results) == 0 and not response:
                error = "No matching records found"
                
        except Exception as e:
            error = "I didn't understand that query. Please try rephrasing or check the example formats."
            results = None

    return render_template('chatbot.html', 
                         results=results if not error else None,
                         error=error,
                         response=response,
                         query=query)

def parse_query_conditions(query):
    """Parse query into list of conditions with their logical operators"""
    # Normalize the query to handle different formats
    query = query.replace("'", '"').lower()  # Convert all quotes to double quotes
    
    if ' and ' not in query and ' or ' not in query:
        return [{'condition': query, 'operator': None}]
    
    conditions = []
    # Split by OR first
    or_parts = query.split(' or ')
    
    for or_part in or_parts:
        # Split by AND
        and_parts = or_part.split(' and ')
        for i, part in enumerate(and_parts):
            part = part.strip()
            # Handle special case where ID is mentioned without id=
            if "id" in part and "=" not in part:
                # Convert to standard format
                if '"' in part:
                    id_val = part.split('"')[1]
                    part = f'id="{id_val}"'
                else:
                    id_val = part.split('id')[1].strip()
                    part = f'id="{id_val}"'
            
            conditions.append({
                'condition': part,
                'operator': 'AND' if i < len(and_parts)-1 else 'OR' if or_part != or_parts[-1] else None
            })
    
    return conditions

def process_conditions(conditions, all_data):
    """Process multiple conditions with AND/OR logic"""
    current_results = set()
    first_condition = True
    
    for i, condition_data in enumerate(conditions):
        condition = condition_data['condition']
        operator = condition_data['operator']
        
        # Get results for current condition
        condition_results = set(evaluate_single_condition(condition, all_data))
        
        # Apply logical operators
        if first_condition:
            current_results = condition_results
            first_condition = False
        else:
            prev_operator = conditions[i-1]['operator']
            if prev_operator == 'AND':
                current_results = current_results.intersection(condition_results)
            elif prev_operator == 'OR':
                current_results = current_results.union(condition_results)
    
    return list(current_results)

def evaluate_single_condition(condition, all_data):
    """Evaluate a single condition and return matching results"""
    condition = condition.lower().strip()
    
    # Handle ID conditions
    if 'id' in condition:
        try:
            # Extract ID value from various formats
            if '"' in condition:
                id_value = condition.split('"')[1]
            elif "'" in condition:
                id_value = condition.split("'")[1]
            elif "=" in condition:
                id_value = condition.split('=')[1].strip()
            else:
                id_value = condition.split('id')[1].strip()
            
            # Case insensitive ID search
            return [item for item in all_data if item[0].id.lower() == id_value.lower()]
        except:
            return []
    
    # Handle process conditions
    elif any(process in condition for process in ['cleaning', 'coating', 'annealing']):
        for process in ['cleaning', 'coating', 'annealing']:
            if process in condition:
                status = 'Y' if any(word in condition for word in ['completed', 'yes', 'done']) else 'N'
                return [item for item in all_data if getattr(item[0], process) == status]
    
    elif 'recipe' in condition:
        if 'front' in condition:
            recipe = extract_value_from_query(condition, 'recipe front')
            return [item for item in all_data if recipe.lower() in item[0].recipe_front.lower()]
        elif 'back' in condition:
            recipe = extract_value_from_query(condition, 'recipe back')
            return [item for item in all_data if recipe.lower() in item[0].recipe_back.lower()]
            
    elif 'glass type' in condition:
        glass_type = extract_value_from_query(condition, 'glass type')
        return [item for item in all_data if glass_type.lower() in item[0].glass_type.lower()]
        
    elif 'date' in condition:
        date = extract_value_from_query(condition, 'date')
        return [item for item in all_data if date in item[0].date]
        
    elif 'done' in condition:
        status = 'Y' if any(word in condition for word in ['completed', 'yes', 'done']) else 'N'
        return [item for item in all_data if item[0].done == status]
            
    elif 'dimensions' in condition or any(dim in condition for dim in ['length', 'thickness', 'height']):
        return handle_dimension_query(condition, all_data)
        
    elif any(exp_type in condition for exp_type in ['transmittance', 'reflectance', 'absorbance', 'plqy', 'sem', 'edx', 'xrd']):
        return handle_experiment_query(condition, all_data)
    
    return []

def extract_id_from_query(query):
    if '"' in query:
        return query.split('"')[1]
    elif "'" in query:
        return query.split("'")[1]
    else:
        return query.split('=')[1].strip()

def extract_value_from_query(query, field):
    if '"' in query:
        return query.split('"')[1]
    elif "'" in query:
        return query.split("'")[1]
    else:
        parts = query.split(field)
        if len(parts) > 1:
            return parts[1].strip()
    return ""

def handle_process_query(query, all_data):
    process_map = {
        'cleaning': 'cleaning',
        'coating': 'coating',
        'annealing': 'annealing'
    }
    
    for process_name, field in process_map.items():
        if process_name in query:
            status = 'Y' if any(word in query for word in ['completed', 'yes', 'done']) else 'N'
            return [item for item in all_data if getattr(item[0], field) == status]
            
    return "Please specify which process (cleaning, coating, or annealing) you're interested in"

def handle_dimension_query(query, all_data):
    dimensions = {
        'length': 'length',
        'thickness': 'thickness',
        'height': 'height'
    }
    
    for dim_name, field in dimensions.items():
        if dim_name in query:
            try:
                value = int(''.join(c for c in query.split(dim_name)[1] if c.isdigit()))
                if '>' in query:
                    return [item for item in all_data if getattr(item[0], field) > value]
                elif '<' in query:
                    return [item for item in all_data if getattr(item[0], field) < value]
                else:
                    return [item for item in all_data if getattr(item[0], field) == value]
            except:
                return []
    return []

def handle_experiment_query(query, all_data):
    exp_types = ['transmittance', 'reflectance', 'absorbance', 'plqy', 'sem', 'edx', 'xrd']
    
    for exp_type in exp_types:
        if exp_type in query:
            if 'has' in query or 'with' in query:
                return [item for item in all_data if item[1] and getattr(item[1], exp_type)]
            else:
                return [item for item in all_data if item[1] and not getattr(item[1], exp_type)]
    return []

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
@app.route('/restore/<string:id>')  # Changed from int:trash_id to string:id
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
            date=sample_trash.date,
            time=sample_trash.time,
            am_pm=sample_trash.am_pm,
            ERB=sample_trash.ERB,
            ERB_description=sample_trash.ERB_description,
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
            user.password = generate_password_hash(password, method='sha256')
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

if __name__ == '__main__':
    with app.app_context():
        # Create tables only if they don't exist
        db.create_all()
        
        # Create admin user if it doesn't exist
        if not User.query.filter_by(username='admin').first():
            admin_user = User(
                username='admin',
                password=generate_password_hash('admin123', method='sha256'),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created successfully!")
    
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)  # Set debug=True during development