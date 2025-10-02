"""
Quick test to verify the patient report query fix
"""
import requests

def test_query_fix():
    url = "http://localhost:8000/patient-report/query"
    
    # Test with a simple question
    test_question = {
        "question": "What information is available in the patient reports?",
        "max_results": 3
    }
    
    try:
        response = requests.post(url, json=test_question)
        result = response.json()
        
        print("🧪 Testing Patient Report Query Fix...")
        print("=" * 50)
        
        if response.status_code == 200:
            print("✅ Query successful!")
            print(f"📋 Question: {result['question']}")
            print(f"🔍 Context used: {result['context_used']} sections")
            print(f"📊 Reports available: {result['reports_available']}")
            print(f"📝 Answer preview: {result['answer'][:200]}...")
            
            if result['sources']:
                print(f"📄 Sources found: {len(result['sources'])}")
            else:
                print("⚠️ No sources found")
                
        else:
            print(f"❌ Query failed: {result}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_query_fix()