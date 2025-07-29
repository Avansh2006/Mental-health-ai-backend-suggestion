#!/usr/bin/env python3
"""
Test script for the Medical PDF RAG System
"""
import os
import sys
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_medical_system():
    """Test the medical system endpoints"""
    base_url = "http://localhost:8000"
    
    print("🏥 Testing Medical PDF RAG System")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("   Run: python main.py")
        return False
    
    # Test 2: Add medical document
    print("\n2. Testing medical document upload...")
    medical_doc = {
        "title": "Type 2 Diabetes Overview",
        "content": """Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). 
        In type 2 diabetes, the body either resists the effects of insulin or doesn't produce enough insulin. 
        Common symptoms include increased thirst, frequent urination, increased hunger, fatigue, and blurred vision.
        Treatment often includes lifestyle changes, blood sugar monitoring, diabetes medications, and sometimes insulin therapy.
        Metformin is commonly prescribed as a first-line medication.""",
        "disease": "Type 2 Diabetes",
        "medicine": "Metformin",
        "tags": ["endocrinology", "chronic", "metabolic"]
    }
    
    try:
        response = requests.post(f"{base_url}/upload-medical-document", json=medical_doc)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Medical document uploaded: {result['document_id']}")
            doc_id = result['document_id']
        else:
            print(f"❌ Medical document upload failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error uploading medical document: {e}")
        return False
    
    # Test 3: Medical query
    print("\n3. Testing medical query...")
    query = {
        "question": "What are the symptoms of type 2 diabetes and what is the first-line treatment?",
        "max_results": 3
    }
    
    try:
        response = requests.post(f"{base_url}/medical-query", json=query)
        if response.status_code == 200:
            result = response.json()
            print("✅ Medical query successful")
            print(f"   Answer: {result['answer'][:200]}...")
            print(f"   Sources found: {len(result['sources'])}")
        else:
            print(f"❌ Medical query failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error with medical query: {e}")
        return False
    
    print("\n🎉 All tests passed! Medical system is working correctly.")
    print("\n📋 Next steps:")
    print("1. Set up your Google API key in .env file")
    print("2. Optionally configure ChromaDB cloud credentials")
    print("3. Access the web interface at: http://localhost:8000")
    print("4. Start uploading medical documents and patient reports!")
    
    return True

if __name__ == "__main__":
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables")
        print("   The system will still work but AI responses may not function correctly")
        print("   Please add your Google AI API key to the .env file")
        print()
    
    test_medical_system()
