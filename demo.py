"""
Demo script showing how to use the PDF RAG system programmatically.
This script demonstrates the core functionality without using the API.
"""

import os
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_system import RAGSystem

def demo_pdf_rag():
    """Demonstrate PDF RAG functionality."""
    
    print("🤖 PDF RAG System Demo")
    print("=" * 40)
    
    # Initialize components
    print("Initializing components...")
    pdf_processor = PDFProcessor()
    vector_store = VectorStore(collection_name="demo_documents")
    
    # Note: This will fail without a valid GOOGLE_API_KEY
    try:
        rag_system = RAGSystem(vector_store)
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("Make sure GOOGLE_API_KEY is set in your .env file")
        return
    
    # Example: Process text directly (simulating PDF content)
    sample_text = """
    Artificial Intelligence (AI) is transforming the way we work and live. 
    Machine learning algorithms can analyze vast amounts of data to identify 
    patterns and make predictions. Natural language processing enables computers 
    to understand and generate human language. Computer vision allows machines 
    to interpret and understand visual information from the world around them.
    
    The applications of AI are numerous and diverse, including:
    - Healthcare: Diagnosis assistance and drug discovery
    - Finance: Fraud detection and algorithmic trading
    - Transportation: Autonomous vehicles and traffic optimization
    - Education: Personalized learning and intelligent tutoring systems
    
    As AI continues to evolve, it's important to consider ethical implications
    and ensure that these technologies are developed responsibly.
    """
    
    print("\n📄 Adding sample document to knowledge base...")
    doc_id = rag_system.add_document(
        sample_text, 
        metadata={"title": "AI Overview", "source": "demo"}
    )
    print(f"✅ Document added with ID: {doc_id}")
    
    # Example queries
    questions = [
        "What is artificial intelligence?",
        "What are some applications of AI in healthcare?",
        "What should we consider as AI evolves?",
        "How does machine learning work?",
    ]
    
    print("\n🔍 Asking questions about the document...")
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        result = rag_system.query(question, k=3)
        print(f"   Answer: {result['answer']}")
        print(f"   Sources used: {result['context_used']}")
    
    # System info
    print("\n📊 System Information:")
    info = rag_system.get_system_info()
    print(f"   Status: {info.get('status', 'unknown')}")
    print(f"   Documents in database: {info.get('vector_store', {}).get('document_count', 0)}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the web interface:")
    print("1. Set your GOOGLE_API_KEY in .env file")
    print("2. Run: python main.py")
    print("3. Open: http://localhost:8000")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    demo_pdf_rag()
