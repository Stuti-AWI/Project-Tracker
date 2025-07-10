#!/bin/bash

# AWI Project Tracker - Deployment Script
# This script helps prepare the application for deployment

echo "🚀 AWI Project Tracker - Deployment Preparation"
echo "=============================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ No git repository found. Initializing git..."
    git init
    git add .
    git commit -m "Initial commit"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository found"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating static directories..."
mkdir -p static/sample_images
mkdir -p static/uploads
echo "✅ Static directories created"

# Set up database (if running locally)
if [ "$1" = "local" ]; then
    echo "🗄️ Setting up local database..."
    export FLASK_APP=app.py
    flask db init 2>/dev/null || echo "Database already initialized"
    flask db migrate -m "Initial migration" 2>/dev/null || echo "Migration files exist"
    flask db upgrade
    echo "✅ Local database setup complete"
fi

# Check for required files
echo "🔍 Checking required files..."
required_files=("app.py" "requirements.txt" "Procfile" "app.yaml")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
    fi
done

# Display next steps
echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. 📝 Update app.yaml with your GitHub repository details"
echo "2. 🔑 Set up MongoDB Atlas account and get connection string"
echo "3. 🤖 Get OpenAI API key from OpenAI Platform"
echo "4. 📧 Set up SendGrid account and get API key"
echo "5. 🌊 Push to GitHub: git add . && git commit -m 'Deploy setup' && git push"
echo "6. 🚀 Deploy on Digital Ocean App Platform"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions"
echo ""
echo "✅ Deployment preparation complete!" 