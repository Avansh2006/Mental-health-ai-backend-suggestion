#!/bin/bash

# PDF RAG System Startup Script

echo "🚀 Starting PDF RAG System..."
echo "================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.template .env
    echo "📝 Please edit .env file and add your GOOGLE_API_KEY"
    echo "   Get your key from: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Press Enter after you've added your API key to .env file..."
fi

# Check if GOOGLE_API_KEY is set
if ! grep -q "^GOOGLE_API_KEY=.*[^_here]" .env; then
    echo "❌ GOOGLE_API_KEY not set in .env file"
    echo "   Please edit .env and add your Google API key"
    exit 1
fi

echo "✅ Environment configuration found"

# Start the server
echo "🖥️  Starting FastAPI server..."
echo "   Web interface: http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

python main.py
