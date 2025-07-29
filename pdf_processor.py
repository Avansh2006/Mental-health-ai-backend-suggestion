"""
PDF processing utilities for extracting text from PDF documents.
"""
import PyPDF2
from typing import List
import io


class PDFProcessor:
    """Handles PDF text extraction and processing."""
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            str: Extracted text from the PDF
        """
        pdf_file = io.BytesIO(pdf_bytes)
        return self.extract_text_from_pdf(pdf_file)
