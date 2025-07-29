"""
RAG (Retrieval-Augmented Generation) system for querying documents.
"""
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import VectorStore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGSystem:
    """Main RAG system that combines retrieval and generation."""
    
    def __init__(self, vector_store: VectorStore, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Vector store instance for document retrieval
            model_name: Name of the language model to use
        """
        self.vector_store = vector_store
        
        # Initialize the language model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7
        )
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dict containing answer and source documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "question": question,
                "context_used": 0
            }
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Generate prompt
        prompt = self._create_prompt(question, context)
        
        # Generate answer
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        # Prepare sources
        sources = self._prepare_sources(retrieved_docs)
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "context_used": len(retrieved_docs)
        }
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc['document']}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the language model."""
        return f"""You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}

Question: {question}

Instructions:
1. Answer the question based solely on the information provided in the context documents
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your response
4. If you reference specific information, try to indicate which document it came from
5. Maintain a helpful and professional tone

Answer:"""
    
    def _prepare_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information from retrieved documents."""
        sources = []
        for doc in documents:
            source = {
                "text": doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document'],
                "metadata": doc['metadata'],
                "relevance_score": 1 - doc['distance'] if doc['distance'] is not None else None
            }
            sources.append(source)
        
        return sources
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the system.
        
        Args:
            text: Document text
            metadata: Optional metadata
            
        Returns:
            str: Document ID
        """
        return self.vector_store.add_document(text, metadata)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the system.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if successful
        """
        return self.vector_store.delete_document(document_id)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        vector_info = self.vector_store.get_collection_info()
        return {
            "vector_store": vector_info,
            "model": "Google Gemini 2.0 Flash",
            "status": "ready"
        }
