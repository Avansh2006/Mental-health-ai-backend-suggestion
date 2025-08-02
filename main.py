import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# Import our custom modules
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_system import RAGSystem
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI(
    title="PDF RAG System",
    description="A Retrieval-Augmented Generation system for PDF documents",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from these origins
    allow_credentials=True,  # Allows cookies, sessions, etc.
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (Authorization, Content-Type, etc.)
)


# Initialize components
pdf_processor = PDFProcessor()
vector_store = VectorStore()
rag_system = RAGSystem(vector_store)

# Initialize medical collections
vector_store.create_medical_collections()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5

class MedicalQueryRequest(BaseModel):
    question: str
    patient_report_id: Optional[str] = None
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str
    context_used: int

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str

class MedicalDocumentRequest(BaseModel):
    title: str
    content: str
    disease: Optional[str] = None
    medicine: Optional[str] = None
    tags: Optional[List[str]] = None

class SymptomAnalysisRequest(BaseModel):
    symptoms: List[str]
    age: Optional[int] = None
    gender: Optional[str] = None
    additional_info: Optional[str] = None

class SuggestedCondition(BaseModel):
    name: str
    probability: int
    severity: str
    recommendations: List[str]
    specialists: List[str]
    urgency: str

class SymptomAnalysisResponse(BaseModel):
    conditions: List[SuggestedCondition]
    general_advice: str
    emergency_warning: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the medical web interface."""
    try:
        with open("medical_interface.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {
            "message": "Medical PDF RAG System API",
            "version": "1.0.0",
            "endpoints": {
                "upload": "/upload-pdf",
                "upload-medical": "/upload-medical-document", 
                "upload-patient-report": "/upload-patient-report",
                "query": "/query",
                "medical-query": "/medical-query",
                "health": "/health",
                "system-info": "/system-info"
            }
        }

@app.post("/upload-pdf", response_model=DocumentResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and add it to the knowledge base.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        
        # Extract text from PDF
        text = pdf_processor.extract_text_from_bytes(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Add to vector store
        metadata = {
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type
        }
        
        document_id = rag_system.add_document(text, metadata)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="Successfully uploaded and processed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base with a question.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Query the RAG system
        result = rag_system.query(request.question, k=request.max_results)
        
        return QueryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload-medical-document", response_model=DocumentResponse)
async def upload_medical_document(request: MedicalDocumentRequest):
    """
    Upload a medical document (disease/medicine information) to the knowledge base.
    """
    try:
        if not request.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Prepare metadata
        metadata = {
            "title": request.title,
            "document_type": "medical_reference",
            "disease": request.disease,
            "medicine": request.medicine,
            "tags": request.tags or []
        }
        
        # Add to medical documents collection
        document_id = vector_store.add_medical_document(request.content, metadata)
        
        return DocumentResponse(
            document_id=document_id,
            filename=request.title,
            status="Medical document successfully added to knowledge base"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing medical document: {str(e)}")

@app.post("/upload-medical-pdf")
async def upload_medical_pdf(file: UploadFile = File(...)):
    """
    Upload a medical PDF file with detailed analysis and feedback.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        
        # Extract text from PDF
        text = pdf_processor.extract_text_from_bytes(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Upload medical PDF with detailed analysis
        result = vector_store.upload_medical_pdf(text, file.filename)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing medical PDF: {str(e)}")

@app.post("/upload-patient-report", response_model=DocumentResponse)
async def upload_patient_report(file: UploadFile = File(...)):
    """
    Upload a patient medical report PDF.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        
        # Extract text from PDF
        text = pdf_processor.extract_text_from_bytes(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Add to patient reports collection
        metadata = {
            "filename": file.filename,
            "file_size": len(content),
            "content_type": file.content_type,
            "document_type": "patient_report"
        }
        
        document_id = vector_store.add_patient_report(text, metadata)
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="Patient report successfully uploaded and processed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing patient report: {str(e)}")

@app.post("/medical-query", response_model=QueryResponse)
async def medical_query(request: MedicalQueryRequest):
    """
    Query medical documents and patient reports with medical context.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Search medical documents for context
        medical_results = vector_store.search_medical_documents(request.question, k=request.max_results)
        
        # If patient report ID is provided, also search that specific report
        patient_results = []
        if request.patient_report_id:
            patient_results = vector_store.search_patient_reports(request.question, k=request.max_results)
        
        # Combine results for context
        all_results = medical_results + patient_results
        
        # Prepare context for RAG
        context_docs = []
        sources = []
        
        for result in all_results:
            context_docs.append(result["document"])
            source_info = {
                "content": result["document"][:200] + "...",
                "metadata": result["metadata"],
                "similarity": 1 - result["distance"] if result["distance"] else 1.0
            }
            sources.append(source_info)
        
        # Generate answer using RAG system with medical context
        if context_docs:
            context = "\n\n".join(context_docs)
            medical_prompt = f"""Based on the following medical information, please answer the question: {request.question}

Medical Context:
{context}

Please provide a comprehensive answer based on the medical information provided. If the question is about a patient report, analyze the symptoms, conditions, and provide relevant medical insights."""
            
            # Use the RAG system's LLM to generate response
            answer = rag_system.llm.invoke(medical_prompt).content
        else:
            answer = "I don't have enough medical information to answer your question. Please try uploading relevant medical documents first."
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            question=request.question,
            context_used=len(context_docs)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing medical query: {str(e)}")

@app.post("/symptom-analysis", response_model=SymptomAnalysisResponse)
async def symptom_analysis(request: SymptomAnalysisRequest):
    """
    Analyze symptoms and provide potential conditions based on uploaded medical knowledge.
    """
    try:
        if not request.symptoms or len(request.symptoms) == 0:
            raise HTTPException(status_code=400, detail="At least one symptom must be provided")
        
        # Create search query from symptoms
        symptoms_text = " ".join(request.symptoms)
        search_query = symptoms_text  # Remove "symptoms:" prefix
        
        # Add additional context if provided
        if request.additional_info:
            search_query += f" {request.additional_info}"
        
        # Search medical documents for relevant conditions
        medical_results = vector_store.search_medical_documents(search_query, k=5) or []
        
        # Also search patient reports collection since medical knowledge is stored there
        patient_results = vector_store.search_patient_reports(search_query, k=5) or []
        
        # Combine results
        all_results = medical_results + patient_results
        
        # Debug logging
        print(f"🔍 Search query: {search_query}")
        print(f"📊 Medical docs results: {len(medical_results)}")
        print(f"📊 Patient reports results: {len(patient_results)}")
        print(f"📊 Total results: {len(all_results)}")
        if all_results:
            print(f"📄 First result preview: {all_results[0]['document'][:100]}...")
        
        # Prepare context for analysis
        context_docs = []
        for result in all_results:
            context_docs.append(result["document"])
        
        if not context_docs:
            # Fallback response when no medical data is available
            return SymptomAnalysisResponse(
                conditions=[],
                general_advice="No medical data available. Please upload medical documents first to get accurate analysis.",
                emergency_warning="If you have severe symptoms, please consult a healthcare professional immediately."
            )
        
        # Create detailed prompt for symptom analysis
        age_info = f", age {request.age}" if request.age else ""
        gender_info = f", {request.gender}" if request.gender else ""
        patient_info = f"Patient{age_info}{gender_info}"
        
        analysis_prompt = f"""Based on the following medical information, analyze the symptoms and provide potential conditions for {patient_info}.

Symptoms reported: {', '.join(request.symptoms)}
{f'Additional information: {request.additional_info}' if request.additional_info else ''}

Medical Knowledge Base:
{chr(10).join(context_docs[:5])}

Please analyze these symptoms and provide a response in the following JSON format:
{{
    "conditions": [
        {{
            "name": "Condition Name",
            "probability": 85,
            "severity": "low|medium|high",
            "recommendations": ["recommendation 1", "recommendation 2"],
            "specialists": ["specialist type 1", "specialist type 2"],
            "urgency": "routine|soon|urgent"
        }}
    ],
    "general_advice": "General medical advice based on symptoms",
    "emergency_warning": "Warning if urgent care needed (optional)"
}}

Provide 3-5 most likely conditions based on the symptoms. Be conservative with probability scores and always recommend professional medical consultation. Include emergency warning if symptoms suggest urgent care is needed."""
        
        # Generate analysis using RAG system
        response = rag_system.llm.invoke(analysis_prompt).content
        
        # Try to parse JSON response
        import json
        try:
            # Clean the response to extract JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            parsed_response = json.loads(json_str)
            
            # Convert to our response model
            conditions = []
            for condition_data in parsed_response.get("conditions", []):
                condition = SuggestedCondition(
                    name=condition_data.get("name", "Unknown Condition"),
                    probability=min(condition_data.get("probability", 50), 100),
                    severity=condition_data.get("severity", "medium"),
                    recommendations=condition_data.get("recommendations", []),
                    specialists=condition_data.get("specialists", ["General Physician"]),
                    urgency=condition_data.get("urgency", "routine")
                )
                conditions.append(condition)
            
            return SymptomAnalysisResponse(
                conditions=conditions,
                general_advice=parsed_response.get("general_advice", "Please consult with a healthcare professional for proper diagnosis."),
                emergency_warning=parsed_response.get("emergency_warning")
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if JSON parsing fails
            return SymptomAnalysisResponse(
                conditions=[
                    SuggestedCondition(
                        name="Medical Consultation Needed",
                        probability=100,
                        severity="medium",
                        recommendations=["Consult with a healthcare professional", "Provide detailed symptom history"],
                        specialists=["General Physician"],
                        urgency="soon"
                    )
                ],
                general_advice="Based on your symptoms, it's recommended to consult with a healthcare professional for proper evaluation and diagnosis.",
                emergency_warning="If symptoms are severe or worsening, seek immediate medical attention."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing symptoms: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the knowledge base.
    """
    try:
        success = rag_system.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "system": "PDF RAG System",
        "timestamp": "2025-07-28"
    }

@app.get("/system-info")
async def get_system_info():
    """
    Get information about the RAG system.
    """
    try:
        info = rag_system.get_system_info()
        return info
    except Exception as e:
        return {"error": f"Error getting system info: {str(e)}"}

def main():
    print("Starting PDF RAG System...")
    print("Make sure to set your GOOGLE_API_KEY environment variable!")
    
    # Get host and port from environment variables for production deployment
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    
    print(f"\nOnce running, you can:")
    print(f"- Access the web interface at: http://{host}:{port}")
    print(f"- View API docs at: http://{host}:{port}/docs")
    
    # Use reload=False in production
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    uvicorn.run("main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    main()