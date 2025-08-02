import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI(
    title="PDF RAG System",
    description="A lightweight Retrieval-Augmented Generation system for PDF documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve a simple web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF RAG System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            input, textarea { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
            .result { background: white; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <h1>🏥 Medical PDF RAG System</h1>
        <div class="container">
            <h2>Welcome to the PDF RAG System</h2>
            <p>This is a lightweight version optimized for Render's free tier.</p>
            <p><strong>Status:</strong> ✅ System is running successfully!</p>
            
            <h3>Available Endpoints:</h3>
            <ul>
                <li><strong>GET /</strong> - This page</li>
                <li><strong>GET /docs</strong> - API documentation</li>
                <li><strong>GET /health</strong> - Health check</li>
                <li><strong>POST /query</strong> - Ask questions (coming soon)</li>
            </ul>
            
            <div style="margin-top: 20px;">
                <a href="/docs" target="_blank">
                    <button>📚 View API Documentation</button>
                </a>
                <a href="/health" target="_blank">
                    <button style="background: #28a745; margin-left: 10px;">💚 Health Check</button>
                </a>
            </div>
        </div>
        
        <div class="container">
            <h3>🚀 Deployment Success!</h3>
            <p>Your PDF RAG system has been successfully deployed to Render.</p>
            <p>This lightweight version uses minimal memory to work within the 512MB limit.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "PDF RAG System is running",
        "version": "1.0.0",
        "memory_optimized": True
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents endpoint - simplified version."""
    return QueryResponse(
        answer=f"This is a demo response to: '{request.question}'. Full RAG functionality will be available in the next update.",
        sources=["demo_source.pdf"],
        question=request.question
    )

@app.get("/system/info")
async def get_system_info():
    """Get system information."""
    return {
        "status": "running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "memory_optimized": True,
        "google_api_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }

def main():
    print("Starting Lightweight PDF RAG System...")
    
    # Get host and port from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Server will start on {host}:{port}")
    print("🚀 Optimized for Render free tier (512MB RAM)")
    
    # Use minimal workers for memory efficiency
    uvicorn.run(
        "main_lite:app", 
        host=host, 
        port=port, 
        reload=False,
        workers=1,
        access_log=False  # Reduce memory usage
    )

if __name__ == "__main__":
    main()
