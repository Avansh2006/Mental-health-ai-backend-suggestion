"""
Vector database for storing and retrieving document embeddings.
"""
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use OpenAI embeddings instead of HuggingFace for memory efficiency
from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Dict, Any, Optional
import uuid
import os
import re


class VectorStore:
    """Handles document chunking, embedding, and similarity search."""
    
    def __init__(self, collection_name: str = "pdf_documents", use_cloud: bool = True):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            use_cloud: Whether to use ChromaDB cloud or local instance
        """
        self.collection_name = collection_name
        self.use_cloud = use_cloud
        
        # Initialize ChromaDB client
        if use_cloud and os.getenv("CHROMA_CLIENT_AUTH_TOKEN"):
            # Cloud configuration
            auth_token = os.getenv("CHROMA_CLIENT_AUTH_TOKEN")
            server_host = os.getenv("CHROMA_SERVER_HOST", "api.trychroma.com")
            server_port = int(os.getenv("CHROMA_SERVER_PORT", "443"))
            
            try:
                # Try different connection methods for ChromaDB cloud
                try:
                    # Method 1: Direct URL with HTTPS
                    self.client = chromadb.HttpClient(
                        host=f"https://{server_host}",
                        headers={"Authorization": f"Bearer {auth_token}"}
                    )
                    # Test connection
                    self.client.heartbeat()
                    print(f"✅ Connected to ChromaDB cloud at https://{server_host}")
                except:
                    # Method 2: With port and SSL
                    self.client = chromadb.HttpClient(
                        host=server_host,
                        port=server_port,
                        ssl=True,
                        headers={"Authorization": f"Bearer {auth_token}"}
                    )
                    # Test connection
                    self.client.heartbeat()
                    print(f"✅ Connected to ChromaDB cloud at {server_host}:{server_port}")
                    
            except Exception as e:
                print(f"⚠️  ChromaDB cloud connection failed: {e}")
                print("📁 Falling back to local storage...")
                self.client = chromadb.PersistentClient(
                    path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
                    settings=Settings(anonymized_telemetry=False)
                )
        else:
            # Local configuration (fallback)
            print("📁 Using local ChromaDB storage...")
            self.client = chromadb.PersistentClient(
                path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Initialize embeddings model (use ChromaDB default for memory efficiency)
        # For memory optimization on free tier, we'll use ChromaDB's default embedding function
        # instead of loading heavy models like HuggingFace transformers
        self.embeddings = None  # Will use ChromaDB's default
        self.embedding_function = None  # ChromaDB will use default sentence transformers
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
        
        # Medical keywords for automatic categorization
        self.medical_keywords = {
            'diseases': [
                'diabetes', 'hypertension', 'cancer', 'pneumonia', 'tuberculosis',
                'asthma', 'bronchitis', 'arthritis', 'osteoporosis', 'migraine',
                'depression', 'anxiety', 'schizophrenia', 'alzheimer', 'parkinson',
                'stroke', 'heart attack', 'myocardial infarction', 'angina',
                'hepatitis', 'cirrhosis', 'kidney disease', 'renal failure'
            ],
            'medications': [
                'aspirin', 'ibuprofen', 'acetaminophen', 'insulin', 'metformin',
                'lisinopril', 'amlodipine', 'atorvastatin', 'omeprazole',
                'levothyroxine', 'amoxicillin', 'azithromycin', 'prednisone',
                'warfarin', 'metoprolol', 'hydrochlorothiazide', 'simvastatin'
            ],
            'symptoms': [
                'fever', 'cough', 'headache', 'nausea', 'vomiting', 'diarrhea',
                'fatigue', 'weakness', 'dizziness', 'chest pain', 'shortness of breath',
                'abdominal pain', 'back pain', 'joint pain', 'muscle pain',
                'rash', 'itching', 'swelling', 'bleeding', 'weight loss', 'weight gain'
            ],
            'medical_terms': [
                'diagnosis', 'treatment', 'prognosis', 'therapy', 'surgery',
                'medication', 'prescription', 'dosage', 'side effects',
                'contraindications', 'patient', 'clinical', 'medical history',
                'vital signs', 'blood pressure', 'heart rate', 'temperature'
            ]
        }
    
    def create_medical_collections(self):
        """Create separate collections for medical documents and patient reports."""
        try:
            print("🏥 Creating medical collections...")
            self.medical_docs_collection = self.client.get_or_create_collection("medical_documents")
            self.patient_reports_collection = self.client.get_or_create_collection("patient_reports")
            
            # Debug: Check counts
            med_count = self.medical_docs_collection.count()
            patient_count = self.patient_reports_collection.count()
            print(f"📊 Medical documents: {med_count}, Patient reports: {patient_count}")
            
        except Exception as e:
            print(f"❌ Error creating medical collections: {e}")
            # Fallback to main collection
            self.medical_docs_collection = self.collection
            self.patient_reports_collection = self.collection
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the vector store.
        
        Args:
            text: Document text to add
            metadata: Optional metadata for the document
            
        Returns:
            str: Document ID
        """
        if metadata is None:
            metadata = {}
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ', '.join(str(v) for v in value)
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Process each chunk
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Prepare metadata (no custom embeddings needed)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "document_id": doc_id,
                "chunk_index": i,
                "chunk_id": chunk_id
            })
            
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        # Add to collection (let ChromaDB handle embeddings automatically)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return doc_id
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search using ChromaDB's built-in functionality.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Use ChromaDB's built-in query method without custom embeddings
            results = self.collection.query(
                query_texts=[query],  # Use query_texts instead of query_embeddings
                n_results=k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def add_medical_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a medical document to the medical documents collection.
        
        Args:
            text: Medical document text
            metadata: Medical document metadata (disease, medicine, etc.)
            
        Returns:
            str: Document ID
        """
        if not hasattr(self, 'medical_docs_collection'):
            self.create_medical_collections()
            
        if metadata is None:
            metadata = {}
        
        # Convert lists to strings for ChromaDB compatibility
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            metadata['tags'] = ', '.join(metadata['tags'])
        
        metadata['document_type'] = 'medical_reference'
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Process each chunk
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Prepare metadata (no custom embeddings needed)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "document_id": doc_id,
                "chunk_index": i,
                "chunk_id": chunk_id
            })
            
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        # Add to medical documents collection (let ChromaDB handle embeddings automatically)
        self.medical_docs_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return doc_id
    
    def add_patient_report(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a patient medical report to the patient reports collection.
        
        Args:
            text: Patient report text
            metadata: Patient report metadata
            
        Returns:
            str: Document ID
        """
        if not hasattr(self, 'patient_reports_collection'):
            self.create_medical_collections()
            
        if metadata is None:
            metadata = {}
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ', '.join(str(v) for v in value)
        
        metadata['document_type'] = 'patient_report'
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Process each chunk
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Prepare metadata (no custom embeddings needed)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "document_id": doc_id,
                "chunk_index": i,
                "chunk_id": chunk_id
            })
            
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        # Add to patient reports collection (let ChromaDB handle embeddings automatically)
        self.patient_reports_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return doc_id
    
    def search_medical_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search in medical documents collection.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar medical documents with metadata
        """
        if not hasattr(self, 'medical_docs_collection'):
            self.create_medical_collections()
            
        # Search in medical documents collection (use query_texts instead of query_embeddings)
        results = self.medical_docs_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def search_patient_reports(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search in patient reports collection.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar patient reports with metadata
        """
        if not hasattr(self, 'patient_reports_collection'):
            self.create_medical_collections()
            
        # Search in patient reports collection (use query_texts instead of query_embeddings)
        results = self.patient_reports_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if successful
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                # Delete all chunks
                self.collection.delete(ids=results['ids'])
                return True
            
            return False
        except Exception as e:
            print(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count
        }
    
    def detect_medical_content_type(self, text: str) -> Dict[str, Any]:
        """
        Automatically detect the type of medical content in the text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dict containing detected medical information
        """
        text_lower = text.lower()
        detected_info = {
            'is_medical': False,
            'content_types': [],
            'detected_diseases': [],
            'detected_medications': [],
            'detected_symptoms': [],
            'confidence_score': 0.0
        }
        
        total_matches = 0
        
        # Check for diseases
        for disease in self.medical_keywords['diseases']:
            if disease in text_lower:
                detected_info['detected_diseases'].append(disease)
                total_matches += 1
        
        # Check for medications
        for medication in self.medical_keywords['medications']:
            if medication in text_lower:
                detected_info['detected_medications'].append(medication)
                total_matches += 1
        
        # Check for symptoms
        for symptom in self.medical_keywords['symptoms']:
            if symptom in text_lower:
                detected_info['detected_symptoms'].append(symptom)
                total_matches += 1
        
        # Check for general medical terms
        medical_term_matches = 0
        for term in self.medical_keywords['medical_terms']:
            if term in text_lower:
                medical_term_matches += 1
        
        total_matches += medical_term_matches
        
        # Determine content types
        if detected_info['detected_diseases']:
            detected_info['content_types'].append('disease_information')
        if detected_info['detected_medications']:
            detected_info['content_types'].append('medication_information')
        if detected_info['detected_symptoms']:
            detected_info['content_types'].append('symptom_information')
        
        # Check for specific document types
        if any(term in text_lower for term in ['patient report', 'medical record', 'case study']):
            detected_info['content_types'].append('patient_report')
        if any(term in text_lower for term in ['treatment protocol', 'clinical guideline', 'medical procedure']):
            detected_info['content_types'].append('clinical_guideline')
        if any(term in text_lower for term in ['drug information', 'pharmaceutical', 'medication guide']):
            detected_info['content_types'].append('drug_information')
        
        # Calculate confidence score
        word_count = len(text_lower.split())
        if word_count > 0:
            detected_info['confidence_score'] = min(total_matches / (word_count * 0.1), 1.0)
        
        # Determine if content is medical
        detected_info['is_medical'] = (
            total_matches >= 3 or 
            medical_term_matches >= 5 or 
            detected_info['confidence_score'] > 0.1
        )
        
        return detected_info
    
    def extract_medical_metadata(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Extract medical-specific metadata from text content.
        
        Args:
            text: Document text
            filename: Original filename
            
        Returns:
            Dict containing medical metadata
        """
        medical_info = self.detect_medical_content_type(text)
        
        metadata = {
            'filename': filename,
            'document_type': 'medical_document' if medical_info['is_medical'] else 'general_document',
            'is_medical': medical_info['is_medical'],
            'confidence_score': medical_info['confidence_score'],
            'content_types': ','.join(medical_info['content_types']),  # Convert list to string
            'detected_diseases': ','.join(medical_info['detected_diseases'][:10]),  # Convert list to string
            'detected_medications': ','.join(medical_info['detected_medications'][:10]),  # Convert list to string
            'detected_symptoms': ','.join(medical_info['detected_symptoms'][:10]),  # Convert list to string
            'total_diseases': len(medical_info['detected_diseases']),
            'total_medications': len(medical_info['detected_medications']),
            'total_symptoms': len(medical_info['detected_symptoms'])
        }
        
        # Extract additional patterns
        text_lower = text.lower()
        
        # Extract dosage information
        dosage_patterns = re.findall(r'\d+\s*(?:mg|g|ml|mcg|units?)\b', text_lower)
        if dosage_patterns:
            metadata['contains_dosage_info'] = True
            metadata['dosage_examples'] = ','.join(dosage_patterns[:5])  # Convert list to string
        
        # Extract age/demographic information
        age_patterns = re.findall(r'(?:age|aged)\s+(\d+)', text_lower)
        if age_patterns:
            metadata['contains_age_info'] = True
            metadata['age_references'] = ','.join(age_patterns[:3])  # Convert list to string
        
        # Check for severity indicators
        severity_terms = ['mild', 'moderate', 'severe', 'critical', 'acute', 'chronic']
        found_severity = [term for term in severity_terms if term in text_lower]
        if found_severity:
            metadata['severity_indicators'] = ','.join(found_severity)  # Convert list to string
        
        return metadata
    
    def add_medical_pdf_document(self, text: str, filename: str = None) -> str:
        """
        Add a PDF document with automatic medical content detection and categorization.
        
        Args:
            text: Extracted PDF text
            filename: Original PDF filename
            
        Returns:
            str: Document ID
        """
        # Extract medical metadata
        metadata = self.extract_medical_metadata(text, filename)
        
        # Determine which collection to use
        if metadata['is_medical']:
            print(f"📋 Detected medical document: {filename}")
            print(f"🔍 Content types: {', '.join(metadata['content_types'])}")
            print(f"🏥 Found: {metadata['total_diseases']} diseases, {metadata['total_medications']} medications, {metadata['total_symptoms']} symptoms")
            
            # Route to appropriate medical collection
            if 'patient_report' in metadata['content_types']:
                return self.add_patient_report(text, metadata)
            else:
                return self.add_medical_document(text, metadata)
        else:
            print(f"📄 Adding as general document: {filename}")
            return self.add_document(text, metadata)
    
    def batch_add_medical_pdfs(self, pdf_texts_and_filenames: List[tuple]) -> List[str]:
        """
        Batch process multiple medical PDF documents.
        
        Args:
            pdf_texts_and_filenames: List of (text, filename) tuples
            
        Returns:
            List of document IDs
        """
        document_ids = []
        medical_count = 0
        general_count = 0
        
        print(f"📚 Processing {len(pdf_texts_and_filenames)} PDF documents...")
        
        for i, (text, filename) in enumerate(pdf_texts_and_filenames, 1):
            print(f"\n📄 Processing document {i}/{len(pdf_texts_and_filenames)}: {filename}")
            
            try:
                doc_id = self.add_medical_pdf_document(text, filename)
                document_ids.append(doc_id)
                
                # Count document types
                metadata = self.extract_medical_metadata(text, filename)
                if metadata['is_medical']:
                    medical_count += 1
                else:
                    general_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                continue
        
        print(f"\n✅ Batch processing complete!")
        print(f"🏥 Medical documents: {medical_count}")
        print(f"📄 General documents: {general_count}")
        print(f"📊 Total processed: {len(document_ids)}")
        
        return document_ids
    
    def search_medical_knowledge(self, query: str, k: int = 5, content_type: str = None) -> List[Dict[str, Any]]:
        """
        Search medical knowledge base with optional content type filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            content_type: Optional filter by content type
            
        Returns:
            List of relevant medical documents
        """
        # Prepare search filters
        where_filter = {"is_medical": True}
        if content_type:
            where_filter["content_types"] = {"$contains": content_type}
        
        # Search in main collection with medical filter (use query_texts instead of query_embeddings)
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter
            )
        except:
            # Fallback without where filter if ChromaDB version doesn't support it
            results = self.collection.query(
                query_texts=[query],
                n_results=k * 2  # Get more results to filter manually
            )
            
            # Manual filtering
            if results['metadatas'] and results['metadatas'][0]:
                filtered_indices = []
                for i, metadata in enumerate(results['metadatas'][0]):
                    if metadata.get('is_medical', False):
                        if not content_type or content_type in str(metadata.get('content_types', '')):
                            filtered_indices.append(i)
                            if len(filtered_indices) >= k:
                                break
                
                # Filter results
                for key in results:
                    if results[key] and results[key][0]:
                        results[key][0] = [results[key][0][i] for i in filtered_indices]
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "relevance_score": 1 - results['distances'][0][i] if results['distances'] else 1.0
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_medical_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the medical knowledge base.
        
        Returns:
            Dict containing medical database statistics
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            stats = {
                'total_documents': len(all_docs['ids']) if all_docs['ids'] else 0,
                'medical_documents': 0,
                'general_documents': 0,
                'content_type_breakdown': {},
                'top_diseases': {},
                'top_medications': {},
                'top_symptoms': {}
            }
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    if metadata.get('is_medical', False):
                        stats['medical_documents'] += 1
                        
                        # Count content types
                        content_types = metadata.get('content_types', [])
                        if isinstance(content_types, str):
                            content_types = [content_types]
                        
                        for content_type in content_types:
                            stats['content_type_breakdown'][content_type] = stats['content_type_breakdown'].get(content_type, 0) + 1
                        
                        # Count diseases, medications, symptoms
                        for category in ['detected_diseases', 'detected_medications', 'detected_symptoms']:
                            items = metadata.get(category, [])
                            if isinstance(items, str):
                                items = items.split(', ')
                            
                            target_dict = {
                                'detected_diseases': stats['top_diseases'],
                                'detected_medications': stats['top_medications'],
                                'detected_symptoms': stats['top_symptoms']
                            }[category]
                            
                            for item in items:
                                if item:
                                    target_dict[item] = target_dict.get(item, 0) + 1
                    else:
                        stats['general_documents'] += 1
            
            # Sort top items
            for key in ['top_diseases', 'top_medications', 'top_symptoms']:
                stats[key] = dict(sorted(stats[key].items(), key=lambda x: x[1], reverse=True)[:10])
            
            return stats
            
        except Exception as e:
            print(f"Error getting medical statistics: {e}")
            return {
                'total_documents': 0,
                'medical_documents': 0,
                'general_documents': 0,
                'error': str(e)
            }
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict containing collection statistics
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            return {
                "collection_name": collection_name,
                "document_count": count
            }
        except Exception as e:
            print(f"Error getting stats for collection {collection_name}: {e}")
            return {
                "collection_name": collection_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def upload_medical_pdf(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Upload a medical PDF with detailed analysis and feedback.
        Similar to patient report analysis but for general medical content.
        
        Args:
            text: Extracted PDF text
            filename: Original PDF filename
            
        Returns:
            Dict containing upload results and medical analysis
        """
        # Extract comprehensive medical metadata
        metadata = self.extract_medical_metadata(text, filename)
        
        # Perform detailed medical content analysis
        analysis_result = {
            "upload_successful": False,
            "document_id": None,
            "filename": filename,
            "file_size_chars": len(text),
            "is_medical_content": metadata['is_medical'],
            "confidence_score": metadata['confidence_score'],
            "content_analysis": {
                "content_types": metadata['content_types'],
                "diseases_found": metadata['detected_diseases'][:10],
                "medications_found": metadata['detected_medications'][:10],
                "symptoms_found": metadata['detected_symptoms'][:10],
                "total_medical_entities": {
                    "diseases": metadata['total_diseases'],
                    "medications": metadata['total_medications'],
                    "symptoms": metadata['total_symptoms']
                },
                "contains_dosage_info": metadata.get('contains_dosage_info', False),
                "contains_age_info": metadata.get('contains_age_info', False),
                "severity_indicators": metadata.get('severity_indicators', [])
            },
            "processing_details": {
                "chunks_created": 0,
                "collection_used": "",
                "processing_time": 0
            },
            "recommendations": []
        }
        
        try:
            import time
            start_time = time.time()
            
            # Count chunks that will be created
            chunks = self.text_splitter.split_text(text)
            analysis_result["processing_details"]["chunks_created"] = len(chunks)
            
            # Determine collection and upload
            if metadata['is_medical']:
                print(f"🏥 Processing medical document: {filename}")
                print(f"📊 Confidence score: {metadata['confidence_score']:.2f}")
                print(f"🔍 Content types detected: {', '.join(metadata['content_types'])}")
                
                # Route to appropriate medical collection
                if 'patient_report' in metadata['content_types']:
                    doc_id = self.add_patient_report(text, metadata)
                    analysis_result["processing_details"]["collection_used"] = "patient_reports"
                else:
                    doc_id = self.add_medical_document(text, metadata)
                    analysis_result["processing_details"]["collection_used"] = "medical_documents"
                
                # Generate recommendations for medical content
                analysis_result["recommendations"] = self._generate_medical_recommendations(metadata)
                
            else:
                print(f"📄 Processing as general document: {filename}")
                doc_id = self.add_document(text, metadata)
                analysis_result["processing_details"]["collection_used"] = "general_documents"
                
                # Recommendations for non-medical content
                analysis_result["recommendations"] = [
                    "Document does not appear to contain medical content",
                    "Consider uploading medical textbooks, journals, or clinical guidelines for better medical knowledge base",
                    "Current content will be stored in general document collection"
                ]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            analysis_result["processing_details"]["processing_time"] = round(processing_time, 2)
            
            # Mark as successful
            analysis_result["upload_successful"] = True
            analysis_result["document_id"] = doc_id
            
            print(f"✅ Upload successful! Document ID: {doc_id}")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error uploading medical PDF: {str(e)}")
            analysis_result["error"] = str(e)
            analysis_result["recommendations"].append(f"Upload failed: {str(e)}")
        
        return analysis_result
    
    def _generate_medical_recommendations(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on medical content analysis.
        
        Args:
            metadata: Medical metadata from content analysis
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Content type recommendations
        content_types = metadata.get('content_types', [])
        if 'disease_information' in content_types:
            recommendations.append("📋 Disease information detected - great for diagnostic queries")
        if 'medication_information' in content_types:
            recommendations.append("💊 Medication information found - useful for drug interaction queries")
        if 'symptom_information' in content_types:
            recommendations.append("🩺 Symptom information available - helpful for symptom analysis")
        if 'clinical_guideline' in content_types:
            recommendations.append("📖 Clinical guidelines detected - excellent for treatment protocols")
        
        # Quantity recommendations
        total_diseases = metadata.get('total_diseases', 0)
        total_medications = metadata.get('total_medications', 0)
        total_symptoms = metadata.get('total_symptoms', 0)
        
        if total_diseases > 10:
            recommendations.append(f"🏥 Rich disease content ({total_diseases} diseases) - excellent for diagnostic support")
        if total_medications > 10:
            recommendations.append(f"💉 Comprehensive medication data ({total_medications} drugs) - great for prescription guidance")
        if total_symptoms > 15:
            recommendations.append(f"🔍 Extensive symptom information ({total_symptoms} symptoms) - valuable for symptom checking")
        
        # Quality recommendations
        confidence_score = metadata.get('confidence_score', 0)
        if confidence_score > 0.8:
            recommendations.append("⭐ High-quality medical content detected")
        elif confidence_score > 0.5:
            recommendations.append("✅ Good medical content quality")
        else:
            recommendations.append("⚠️ Lower medical content density - consider uploading more specialized medical documents")
        
        # Special features
        if metadata.get('contains_dosage_info'):
            recommendations.append("📏 Dosage information found - useful for medication queries")
        if metadata.get('contains_age_info'):
            recommendations.append("👥 Age-specific information detected - helpful for demographic-based queries")
        
        severity_indicators = metadata.get('severity_indicators', [])
        if severity_indicators:
            recommendations.append(f"⚡ Severity classifications found: {', '.join(severity_indicators[:3])}")
        
        # Usage recommendations
        recommendations.append("💡 Try asking specific questions about diseases, treatments, or symptoms mentioned in this document")
        recommendations.append("🔍 Use medical search to find content specifically from this document")
        
        return recommendations
    
    def get_medical_upload_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for medical uploads.
        
        Returns:
            Dict containing detailed medical database statistics
        """
        try:
            # Get basic medical statistics
            basic_stats = self.get_medical_statistics()
            
            # Add collection-specific statistics
            collections_stats = {}
            
            # Main collection stats
            collections_stats['main_collection'] = self.get_collection_stats(self.collection_name)
            
            # Medical collections stats if they exist
            if hasattr(self, 'medical_docs_collection'):
                collections_stats['medical_documents'] = self.get_collection_stats("medical_documents")
            if hasattr(self, 'patient_reports_collection'):
                collections_stats['patient_reports'] = self.get_collection_stats("patient_reports")
            
            # Enhanced statistics
            enhanced_stats = {
                **basic_stats,
                "collections": collections_stats,
                "upload_recommendations": [
                    "Upload medical textbooks for comprehensive disease information",
                    "Add pharmaceutical guides for medication details",
                    "Include clinical guidelines for treatment protocols",
                    "Upload case studies for real-world examples"
                ],
                "search_tips": [
                    "Use specific medical terms for better results",
                    "Include symptoms and diseases in your queries",
                    "Try medication names for drug information",
                    "Ask about treatment protocols and procedures"
                ]
            }
            
            return enhanced_stats
            
        except Exception as e:
            print(f"Error getting medical upload statistics: {e}")
            return {
                'error': str(e),
                'total_documents': 0,
                'medical_documents': 0,
                'recommendations': ["Try uploading medical PDFs to build your knowledge base"]
            }
