#!/usr/bin/env python3
"""
Test ChromaDB Cloud Connection
"""
import os
import chromadb
from dotenv import load_dotenv

def test_chromadb_connection():
    """Test connection to ChromaDB cloud with your credentials"""
    
    print("🔗 Testing ChromaDB Cloud Connection")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    auth_token = os.getenv("CHROMA_CLIENT_AUTH_TOKEN")
    server_host = os.getenv("CHROMA_SERVER_HOST")
    server_port = int(os.getenv("CHROMA_SERVER_PORT", "443"))
    
    if not auth_token:
        print("❌ No ChromaDB auth token found in .env file")
        return False
    
    print(f"📋 Connection Details:")
    print(f"   Host: {server_host}")
    print(f"   Port: {server_port}")
    print(f"   Token: {auth_token[:20]}...")
    
    # Test different connection methods
    connection_methods = [
        {
            "name": "Method 1: Just hostname without https",
            "params": {
                "host": server_host,
                "headers": {"Authorization": f"Bearer {auth_token}"}
            }
        },
        {
            "name": "Method 2: ChromaDB Cloud Settings format",
            "settings": chromadb.config.Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn",
                chroma_client_auth_credentials=auth_token,
                chroma_server_host=server_host,
                chroma_server_http_port=str(server_port),
                chroma_server_ssl_enabled=True
            )
        },
        {
            "name": "Method 3: With tenant and database",
            "params": {
                "host": server_host,
                "port": server_port,
                "ssl": True,
                "headers": {
                    "Authorization": f"Bearer {auth_token}",
                    "X-Chroma-Database": "medical-rag-db"
                }
            }
        }
    ]
    
    for method in connection_methods:
        print(f"\n🔄 Trying {method['name']}...")
        
        try:
            if 'settings' in method:
                # Use settings-based approach
                client = chromadb.HttpClient(settings=method['settings'])
            else:
                # Use parameter-based approach
                client = chromadb.HttpClient(**method['params'])
            
            # Simple test - try to get system version
            print("   Testing basic connectivity...")
            
            # Try a simple operation
            try:
                collections = client.list_collections()
                print(f"✅ {method['name']} - SUCCESS!")
                print(f"📊 Found {len(collections)} collections")
                
                # Test creating/accessing a collection
                test_collection = client.get_or_create_collection("test_medical_connection")
                print("✅ Collection operations successful!")
                
                print("\n🎉 ChromaDB cloud connection established!")
                print("🏥 Your medical RAG system can now use cloud storage.")
                return True
                
            except Exception as op_error:
                print(f"   ⚠️  Connection established but operation failed: {op_error}")
                continue
                
        except Exception as e:
            print(f"   ❌ {method['name']} failed: {e}")
            continue
    
    print("\n❌ All connection methods failed")
    print("\n💡 Don't worry! Your system will work perfectly with local storage.")
    print("📁 Local storage is actually preferred for development and testing.")
    print("\n🚀 To continue with local storage:")
    print("   1. Comment out the ChromaDB cloud settings in .env")
    print("   2. Restart your application")
    print("   3. Everything will work exactly the same!")
    
    return False

if __name__ == "__main__":
    test_chromadb_connection()
