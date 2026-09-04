import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2
from vector_store import VectorStore

class PatientReportManager:
    def __init__(self, storage_dir: str = "patient_reports_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.pdf_dir = self.storage_dir / "pdfs"
        self.metadata_dir = self.storage_dir / "metadata"
        
        self.pdf_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Initialize vector store for patient reports
        self.vector_store = VectorStore()
        
        # Load existing reports
        self.reports_index = self._load_reports_index()
    
    def _load_reports_index(self) -> Dict:
        """Load the reports index from disk"""
        index_file = self.storage_dir / "reports_index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_reports_index(self):
        """Save the reports index to disk"""
        index_file = self.storage_dir / "reports_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self.reports_index, f, indent=2, ensure_ascii=False)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def save_patient_report(self, file_content: bytes, filename: str, 
                          patient_name: Optional[str] = None,
                          patient_id: Optional[str] = None) -> Dict:
        """
        Save a patient report PDF and extract its content
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            patient_name: Patient name (optional)
            patient_id: Patient ID (optional)
            
        Returns:
            Dict with report information
        """
        # Generate unique report ID
        report_id = str(uuid.uuid4())
        
        # Save PDF file
        pdf_filename = f"{report_id}_{filename}"
        pdf_path = self.pdf_dir / pdf_filename
        
        with open(pdf_path, 'wb') as f:
            f.write(file_content)
        
        # Extract text content
        try:
            text_content = self._extract_text_from_pdf(str(pdf_path))
        except Exception as e:
            # Clean up the saved file if text extraction fails
            os.remove(pdf_path)
            raise Exception(f"Failed to process PDF: {str(e)}")
        
        # Create metadata
        metadata = {
            "report_id": report_id,
            "filename": filename,
            "pdf_path": str(pdf_path),
            "patient_name": patient_name,
            "patient_id": patient_id,
            "upload_date": datetime.now().isoformat(),
            "text_length": len(text_content),
            "status": "processed"
        }
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{report_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Add to vector store
        try:
            self.vector_store.add_patient_report_document(
                content=text_content,
                metadata={
                    "report_id": report_id,
                    "filename": filename,
                    "patient_name": patient_name or "Unknown",
                    "patient_id": patient_id or "Unknown",
                    "upload_date": metadata["upload_date"],
                    "document_type": "patient_report"
                }
            )
        except Exception as e:
            # Clean up files if vector store operation fails
            os.remove(pdf_path)
            os.remove(metadata_file)
            raise Exception(f"Failed to add to vector store: {str(e)}")
        
        # Update reports index
        self.reports_index[report_id] = {
            "filename": filename,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "upload_date": metadata["upload_date"],
            "status": "active"
        }
        self._save_reports_index()
        
        return {
            "report_id": report_id,
            "filename": filename,
            "status": "successfully_saved",
            "text_length": len(text_content),
            "message": f"Patient report saved and processed. {len(text_content)} characters extracted."
        }
    
    def query_patient_reports(self, question: str, max_results: int = 5) -> Dict:
        """
        Query all patient reports with a question
        
        Args:
            question: The question to ask
            max_results: Maximum number of results to return
            
        Returns:
            Dict with answer and sources
        """
        if not self.reports_index:
            return {
                "answer": "No patient reports have been uploaded yet. Please upload a patient report first to get analysis.",
                "sources": [],
                "question": question,
                "context_used": 0,
                "reports_available": 0
            }
        
        try:
            # Search in patient reports vector store
            results = self.vector_store.search_patient_reports(
                query=question,
                k=max_results
            )
            
            if not results:
                return {
                    "answer": f"I found {len(self.reports_index)} patient report(s) but couldn't find relevant information to answer your question. Please try rephrasing your question or ensure it relates to the uploaded patient reports.",
                    "sources": [],
                    "question": question,
                    "context_used": 0,
                    "reports_available": len(self.reports_index)
                }
            
            # Build context from results
            context_parts = []
            sources = []
            
            for i, result in enumerate(results):
                context_parts.append(f"Source {i+1}: {result['content']}")
                sources.append({
                    "content": result['content'],
                    "metadata": result['metadata'],
                    "similarity": result['similarity']
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using Google Generative AI directly
            from langchain_google_genai import ChatGoogleGenerativeAI
            import os
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return {
                    "answer": "Error: Google API key not configured. Please set the GOOGLE_API_KEY environment variable.",
                    "sources": sources,
                    "question": question,
                    "context_used": len(results),
                    "reports_available": len(self.reports_index)
                }
            
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7
            )
            
            prompt = f"""Based ONLY on the following patient report information, please answer the question. 
Do not use any general medical knowledge - only analyze what is specifically mentioned in these patient reports.

Patient Report Context:
{context}

Question: {question}

Please provide a detailed analysis based solely on the information from the uploaded patient reports. If the reports don't contain enough information to answer the question, please state that clearly."""

            try:
                response = llm.invoke(prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error generating AI response: {str(e)}"
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "context_used": len(results),
                "reports_available": len(self.reports_index)
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "question": question,
                "context_used": 0,
                "reports_available": len(self.reports_index)
            }
    
    def list_patient_reports(self) -> List[Dict]:
        """List all saved patient reports"""
        reports = []
        for report_id, info in self.reports_index.items():
            # Get detailed metadata
            metadata_file = self.metadata_dir / f"{report_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    detailed_metadata = json.load(f)
                
                reports.append({
                    "report_id": report_id,
                    "filename": info["filename"],
                    "patient_name": info.get("patient_name"),
                    "patient_id": info.get("patient_id"),
                    "upload_date": info["upload_date"],
                    "status": info["status"],
                    "text_length": detailed_metadata.get("text_length", 0)
                })
        
        return sorted(reports, key=lambda x: x["upload_date"], reverse=True)
    
    def delete_patient_report(self, report_id: str) -> Dict:
        """Delete a patient report"""
        if report_id not in self.reports_index:
            return {"status": "error", "message": "Report not found"}
        
        try:
            # Remove from vector store
            self.vector_store.delete_patient_report(report_id)
            
            # Remove files
            metadata_file = self.metadata_dir / f"{report_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Remove PDF file
                pdf_path = Path(metadata["pdf_path"])
                if pdf_path.exists():
                    os.remove(pdf_path)
                
                # Remove metadata file
                os.remove(metadata_file)
            
            # Remove from index
            del self.reports_index[report_id]
            self._save_reports_index()
            
            return {"status": "success", "message": "Report deleted successfully"}
            
        except Exception as e:
            return {"status": "error", "message": f"Error deleting report: {str(e)}"}
    
    def get_report_summary(self) -> Dict:
        """Get summary of all patient reports"""
        return {
            "total_reports": len(self.reports_index),
            "reports": self.list_patient_reports()
        }