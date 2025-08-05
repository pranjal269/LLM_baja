import google.generativeai as genai
from typing import List, Dict, Any
from app.models import SearchResult, DocumentChunk
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class QuestionAnswerer:
    """Service to answer multiple questions based on document content"""
    
    def __init__(self):
        # Check if Gemini API key is configured
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key-here":
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.gemini_available = True
            logger.info("✅ Gemini API configured for question answering")
        else:
            self.model = None
            self.gemini_available = False
            logger.warning("⚠️  Gemini API key not configured. Using fallback responses.")
    
    def answer_questions(self, questions: List[str], search_results: List[SearchResult]) -> List[str]:
        """
        Answer multiple questions based on search results
        Returns a list of answers corresponding to each question
        """
        try:
            if not self.gemini_available:
                return self._generate_fallback_answers(questions)
            
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Generate answers for each question
            answers = []
            for question in questions:
                answer = self._answer_single_question(question, context)
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error answering questions: {str(e)}")
            return self._generate_fallback_answers(questions)
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context string from search results"""
        if not search_results:
            return "No relevant document content found."
        
        context_parts = []
        for i, result in enumerate(search_results[:10], 1):  # Use top 10 results
            chunk_text = result.chunk.chunk_text
            document_name = result.chunk.document_name
            similarity = result.similarity_score
            
            context_parts.append(
                f"[Context {i}] (Document: {document_name}, Relevance: {similarity:.2f})\n"
                f"{chunk_text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _answer_single_question(self, question: str, context: str) -> str:
        """Answer a single question using LLM"""
        try:
            prompt = f"""
Based on the following document context, please answer the question accurately and concisely.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a direct, factual answer based only on the information in the context
- If the answer is not found in the context, state "The information is not available in the provided document"
- Include specific details like numbers, percentages, time periods when available
- Keep the answer concise but complete

ANSWER:"""

            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Clean up the answer
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {str(e)}")
            return "Unable to generate answer due to processing error."
    
    def _generate_fallback_answers(self, questions: List[str]) -> List[str]:
        """Generate fallback answers when LLM is not available"""
        fallback_responses = [
            "The information is not available in the provided document.",
            "Unable to process the question without proper document analysis.",
            "Document processing service is currently unavailable.",
            "Cannot determine the answer from the available information.",
            "The query requires document analysis capabilities that are currently disabled."
        ]
        
        answers = []
        for i, question in enumerate(questions):
            # Use different fallback responses for variety
            fallback_index = i % len(fallback_responses)
            answers.append(fallback_responses[fallback_index])
        
        return answers
    
    def answer_questions_with_text(self, questions: List[str], document_text: str) -> List[str]:
        """
        Answer questions directly from document text (when search is not available)
        """
        try:
            if not self.gemini_available:
                return self._generate_fallback_answers(questions)
            
            answers = []
            for question in questions:
                answer = self._answer_with_full_text(question, document_text)
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error answering questions with text: {str(e)}")
            return self._generate_fallback_answers(questions)
    
    def _answer_with_full_text(self, question: str, document_text: str) -> str:
        """Answer a question using the full document text"""
        try:
            # Truncate document if too long
            max_context_length = 8000  # Adjust based on model limits
            if len(document_text) > max_context_length:
                document_text = document_text[:max_context_length] + "..."
            
            prompt = f"""
Based on the following document, please answer the question accurately and concisely.

DOCUMENT:
{document_text}

QUESTION: {question}

INSTRUCTIONS:
- Provide a direct, factual answer based only on the information in the document
- If the answer is not found in the document, state "The information is not available in the provided document"
- Include specific details like numbers, percentages, time periods when available
- Keep the answer concise but complete

ANSWER:"""

            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Clean up the answer
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with full text: {str(e)}")
            return "Unable to generate answer due to processing error."