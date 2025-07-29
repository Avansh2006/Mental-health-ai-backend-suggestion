#!/usr/bin/env python3
"""
ChromaDB Configuration Helper Script
"""
import os
from dotenv import load_dotenv

def check_chromadb_config():
    """Check ChromaDB configuration and provide guidance"""
    
    print("🔍 ChromaDB Configuration Checker")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check for ChromaDB cloud credentials
    auth_provider = os.getenv("CHROMA_CLIENT_AUTH_PROVIDER")
    auth_token = os.getenv("CHROMA_CLIENT_AUTH_TOKEN")
    server_host = os.getenv("CHROMA_SERVER_HOST")
    server_port = os.getenv("CHROMA_SERVER_PORT")
    
    print("\n📋 Current Configuration:")
    print("-" * 30)
    
    if auth_provider and auth_token and auth_token != "your_actual_token_from_chroma_dashboard":
        print("☁️  ChromaDB Cloud Configuration:")
        print(f"   Provider: {auth_provider}")
        print(f"   Token: {auth_token[:20]}..." if len(auth_token) > 20 else f"   Token: {auth_token}")
        print(f"   Host: {server_host}")
        print(f"   Port: {server_port}")
        print("\n✅ Cloud configuration detected!")
        print("   Your data will be stored in ChromaDB cloud.")
        
    else:
        print("💾 Local Storage Configuration:")
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        print(f"   Storage Directory: {persist_dir}")
        print("\n✅ Local configuration active!")
        print("   Your data will be stored locally.")
        
        if auth_token == "your_actual_token_from_chroma_dashboard":
            print("\n💡 To use ChromaDB Cloud:")
            print("   1. Sign up at https://www.trychroma.com/")
            print("   2. Create a database/tenant")
            print("   3. Copy your authentication token")
            print("   4. Update CHROMA_CLIENT_AUTH_TOKEN in .env file")
            print("   5. Restart the application")
    
    print("\n🎯 Recommendation:")
    print("   - For development/personal use: Local storage is perfect!")
    print("   - For production/team use: Consider ChromaDB cloud")
    
    print("\n🚀 Your medical RAG system works great with either option!")

if __name__ == "__main__":
    check_chromadb_config()
