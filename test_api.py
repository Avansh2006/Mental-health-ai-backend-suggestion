"""
Test script for the PDF RAG system.
This script demonstrates how to interact with the API endpoints.
"""
import requests
import json
import os

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_system_info():
    """Test the system info endpoint."""
    response = requests.get(f"{BASE_URL}/system-info")
    print("System Info:", response.json())

def upload_pdf(file_path):
    """Upload a PDF file."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return None
    
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Upload successful: {result}")
        return result['document_id']
    else:
        print(f"Upload failed: {response.text}")
        return None

def query_system(question, max_results=5):
    """Query the RAG system."""
    data = {
        "question": question,
        "max_results": max_results
    }
    
    response = requests.post(f"{BASE_URL}/query", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['context_used']}")
        print("\nSource documents:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['text'][:100]}...")
    else:
        print(f"Query failed: {response.text}")

def main():
    """Main test function."""
    print("PDF RAG System Test Script")
    print("=" * 40)
    
    # Test basic endpoints
    print("\n1. Testing health check...")
    test_health()
    
    print("\n2. Testing system info...")
    test_system_info()
    
    # Example queries (these will work after you upload some PDFs)
    print("\n3. Example queries (upload PDFs first):")
    
    example_questions = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the conclusions mentioned?",
        "Are there any recommendations provided?"
    ]
    
    for question in example_questions:
        print(f"\nTesting query: {question}")
        query_system(question)
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nTo upload a PDF and test queries:")
    print("1. Start the server: python main.py")
    print("2. Upload a PDF: upload_pdf('path/to/your/file.pdf')")
    print("3. Query: query_system('Your question here')")

if __name__ == "__main__":
    main()
