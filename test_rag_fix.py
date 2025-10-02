"""
Test the Patient Report Query Fix for RAGSystem
"""
import requests

def test_patient_query_fix():
    url = "http://localhost:8000/patient-report/query"
    
    # Test query
    test_data = {
        "question": "What medical information is available in the patient reports?",
        "max_results": 3
    }
    
    print("🧪 Testing Patient Report Query Fix...")
    print("=" * 50)
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"📝 Question: {result['question']}")
            print(f"🔍 Context sections used: {result['context_used']}")
            print(f"📊 Reports available: {result['reports_available']}")
            print(f"📄 Sources found: {len(result['sources'])}")
            print(f"💬 Answer preview: {result['answer'][:150]}...")
            
            # Check if the answer contains an error
            if "Error" in result['answer']:
                print(f"⚠️ Answer contains error: {result['answer']}")
            else:
                print("✅ Answer generated successfully!")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_patient_query_fix()