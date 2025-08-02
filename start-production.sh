#!/bin/bash

# Production startup script for PDF RAG System

echo "🚀 Starting PDF RAG System in Production Mode..."
echo "================================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file and add your configuration"
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if ! grep -q "^GOOGLE_API_KEY=.*[^_here]" .env; then
    echo "❌ GOOGLE_API_KEY not set in .env file"
    echo "   Please edit .env and add your Google API key"
    exit 1
fi

echo "✅ Environment configuration found"

# Export environment variables
export $(cat .env | xargs)

# Set production defaults
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export ENVIRONMENT=${ENVIRONMENT:-production}

echo "🖥️  Starting FastAPI server in production mode..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Environment: $ENVIRONMENT"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Start with gunicorn for production (install with: pip install gunicorn)
if command -v gunicorn &> /dev/null; then
    echo "Using Gunicorn for production deployment..."
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind $HOST:$PORT
else
    echo "Gunicorn not found, using uvicorn..."
    python main.py
fi
