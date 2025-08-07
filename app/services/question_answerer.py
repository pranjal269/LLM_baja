import google.generativeai as genai
from typing import List, Dict, Any
from app.models import SearchResult, DocumentChunk
from app.config import settings
from app.services.general_analyzer import GeneralDocumentAnalyzer
import logging
import re

logger = logging.getLogger(__name__)

class QuestionAnswerer:
    """Service to answer multiple questions based on document content"""
    
    def __init__(self):
        # Initialize general document analyzer for intelligent fallback
        self.general_analyzer = GeneralDocumentAnalyzer()
        
        try:
            if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key-here":
                genai.configure(api_key=settings.GEMINI_API_KEY)
                
                # Try different model variants
                model_variants = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                
                for model_name in model_variants:
                    try:
                        # Test the model with a simple prompt
                        test_model = genai.GenerativeModel(model_name)
                        test_response = test_model.generate_content("Hello")
                        
                        if test_response and test_response.text:
                            self.model = test_model
                            self.gemini_available = True
                            logger.info(f"✅ Successfully initialized Gemini with {model_name}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed with {model_name}: {str(e)}")
                        continue
                else:
                    # No model worked
                    self.model = None
                    self.gemini_available = False
                    logger.warning("⚠️ All Gemini models failed, using intelligent fallback")
            else:
                self.model = None
                self.gemini_available = False
                logger.warning("⚠️ Gemini API key not configured. Using rule-based responses.")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.model = None
            self.gemini_available = False

    def answer_questions(self, questions: List[str], search_results: List[SearchResult]) -> List[str]:
        """Answer multiple questions based on search results"""
        try:
            logger.info(f"📝 Answering {len(questions)} questions with {len(search_results)} search results")
            
            if not self.gemini_available:
                logger.warning("⚠️  Gemini not available, using intelligent document analysis")
                # Extract text from search results for intelligent processing
                combined_text = ""
                for result in search_results:
                    combined_text += result.chunk.chunk_text + "\n"
                
                # Use general analyzer for intelligent answers
                answers = []
                document_analysis = self.general_analyzer.analyze_document(combined_text)
                for question in questions:
                    answer = self.general_analyzer.answer_question(question, combined_text, document_analysis)
                    answers.append(answer)
                return answers
            
            # Prepare context from search results
            context = self._prepare_context(search_results)
            logger.debug(f"📄 Context prepared: {len(context)} characters")
            
            # Generate answers for each question
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"❓ Processing question {i+1}/{len(questions)}: {question[:50]}...")
                answer = self._answer_single_question(question, context)
                answers.append(answer)
                logger.info(f"✅ Answer {i+1}: {answer[:100]}...")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error answering questions: {str(e)}")
            # Extract text from search results for intelligent processing
            try:
                combined_text = ""
                for result in search_results:
                    combined_text += result.chunk.chunk_text + "\n"
                
                # Use general analyzer as fallback
                answers = []
                document_analysis = self.general_analyzer.analyze_document(combined_text)
                for question in questions:
                    answer = self.general_analyzer.answer_question(question, combined_text, document_analysis)
                    answers.append(answer)
                return answers
            except Exception as fallback_error:
                logger.error(f"Fallback analysis failed: {fallback_error}")
                return ["Unable to process the question due to technical difficulties." for _ in questions]

    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context string from search results"""
        if not search_results:
            logger.warning("⚠️  No search results provided for context")
            return "No relevant document content found."
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
            chunk_text = result.chunk.chunk_text[:1000]  # Limit chunk size
            document_name = result.chunk.document_name
            similarity = result.similarity_score
            
            context_parts.append(
                f"[Context {i}] (Document: {document_name}, Relevance: {similarity:.2f})\n"
                f"{chunk_text}\n"
            )
        
        full_context = "\n---\n".join(context_parts)
        logger.debug(f"📄 Context contains {len(context_parts)} chunks, {len(full_context)} total characters")
        return full_context

    def _answer_single_question(self, question: str, context: str) -> str:
        """Answer a single question using LLM"""
        try:
            if not self.gemini_available:
                logger.warning("⚠️  Gemini not available for question answering")
                return self._extract_relevant_content(question, context)
            
            prompt = f"""
You are an expert document analyst. Answer the question based ONLY on the document context below.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- You MUST provide an answer based on the context provided.
- Extract and synthesize the most relevant information from the context.
- If exact information isn't explicitly stated, provide the closest relevant information.
- NEVER say the information is unavailable or not in the document.
- Be concise but informative.

ANSWER:"""

            logger.debug(f"🤖 Sending prompt to Gemini (length: {len(prompt)})")
            
            try:
                response = self.model.generate_content(prompt)
                answer = response.text.strip() if response and response.text else ""
            except Exception as api_error:
                logger.error(f"API error: {str(api_error)}")
                return self._extract_relevant_content(question, context)
            
            # Clean up the answer
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            # Check for empty or error responses
            if not answer or "not available in" in answer.lower() or "unable to" in answer.lower():
                answer = self._extract_relevant_content(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {str(e)}")
            return self._extract_relevant_content(question, context)

    def _extract_relevant_content(self, question: str, context: str) -> str:
        """Return up to 3 relevant, meaningful sentences from the context based on keyword overlap with the question."""
        try:
            # If context is empty, return a clear message
            if not context or context.strip() == "":
                return "No relevant information found in the document."
            
            # Clean context
            cleaned_context = re.sub(r'---\s*Page\s+\d+\s*---', '', context)
            # Remove lines that look like contact info, addresses, or policy numbers
            cleaned_context = re.sub(r'(www\.[^\s]+|E[- ]?mail:|Call at:|Toll Free|Policy Wordings|UIN-|Issuing Office:|For more details, log on to:|Sales - \d+|Service - \d+)', '', cleaned_context, flags=re.IGNORECASE)
            
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_context)
            question_words = set(re.findall(r'\w+', question.lower()))
            
            # Score each sentence by number of overlapping keywords
            scored_sentences = []
            for sentence in sentences:
                sentence_clean = sentence.strip()
                # Filter out short, junk, or header sentences
                if len(sentence_clean) < 30 or re.match(r'^[A-Z\s\-:]+$', sentence_clean):
                    continue
                sentence_words = set(re.findall(r'\w+', sentence_clean.lower()))
                score = len(question_words & sentence_words)
                if score > 0:
                    scored_sentences.append((score, sentence_clean))
            
            # Sort by score (descending) and pick top 3
            scored_sentences.sort(reverse=True)
            top_sentences = [s for _, s in scored_sentences[:3]]
            
            if top_sentences:
                return " ".join(top_sentences)
            
            # If no good match, fallback to first 2 meaningful sentences
            fallback = []
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if len(sentence_clean) > 40 and not re.match(r'^[A-Z\s\-:]+$', sentence_clean):
                    fallback.append(sentence_clean)
                if len(fallback) == 2:
                    break
            if fallback:
                return " ".join(fallback)
            
            return "No relevant information found in the document."
        except Exception as e:
            logger.error(f"Error extracting relevant content: {str(e)}")
            return "No relevant information found in the document."

    def _generate_fallback_answers(self, questions: List[str]) -> List[str]:
        """Generate answers using general analyzer when LLM is not available"""
        logger.info("Using general analyzer fallback answers")
        # Return generic answers when no document context is available
        return ["Unable to answer this question without proper document context." for question in questions]
    
    def answer_questions_with_text(self, questions: List[str], document_text: str) -> List[str]:
        """Answer questions directly from document text"""
        try:
            logger.info(f"📝 Answering {len(questions)} questions with full text ({len(document_text)} chars)")
            
            if not self.gemini_available:
                logger.info("Using intelligent document analysis (Gemini not available)")
                # Use general analyzer for intelligent processing
                answers = []
                document_analysis = self.general_analyzer.analyze_document(document_text)
                for question in questions:
                    answer = self.general_analyzer.answer_question(question, document_text, document_analysis)
                    answers.append(answer)
                return answers
            
            answers = []
            for i, question in enumerate(questions):
                try:
                    logger.info(f"❓ Processing question {i+1}/{len(questions)} with full text")
                    answer = self._answer_with_full_text(question, document_text)
                    
                    # Check if answer indicates API failure (quota exceeded, etc.)
                    if "No relevant information found" in answer or len(answer) < 10:
                        logger.warning(f"Gemini failed for question, using intelligent analysis fallback")
                        document_analysis = self.general_analyzer.analyze_document(document_text)
                        answer = self.general_analyzer.answer_question(question, document_text, document_analysis)
                    
                    answers.append(answer)
                except Exception as e:
                    # If individual question fails (quota, etc.), use intelligent analysis
                    if "429" in str(e) or "quota" in str(e).lower():
                        logger.warning(f"Gemini quota exceeded, using intelligent analysis fallback")
                        document_analysis = self.general_analyzer.analyze_document(document_text)
                        answer = self.general_analyzer.answer_question(question, document_text, document_analysis)
                        answers.append(answer)
                    else:
                        logger.error(f"Error processing question: {e}")
                        document_analysis = self.general_analyzer.analyze_document(document_text)
                        answer = self.general_analyzer.answer_question(question, document_text, document_analysis)
                        answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error answering questions with text: {str(e)}")
            # Final fallback: use general analyzer
            logger.info("Using general analyzer as final fallback")
            try:
                answers = []
                document_analysis = self.general_analyzer.analyze_document(document_text)
                for question in questions:
                    answer = self.general_analyzer.answer_question(question, document_text, document_analysis)
                    answers.append(answer)
                return answers
            except Exception as final_error:
                logger.error(f"Final fallback failed: {final_error}")
                return ["Unable to process the question due to technical difficulties." for _ in questions]
    
    def _answer_with_full_text(self, question: str, document_text: str) -> str:
        """Answer a question using the full document text"""
        try:
            if not self.gemini_available:
                document_analysis = self.general_analyzer.analyze_document(document_text)
                return self.general_analyzer.answer_question(question, document_text, document_analysis)
            
            # Truncate document if too long
            max_context_length = 6000
            if len(document_text) > max_context_length:
                # Instead of just truncating, take beginning and end portions
                start_text = document_text[:4000]
                end_text = document_text[-2000:]
                document_text = start_text + "\n[...document continues...]\n" + end_text
                logger.info(f"📄 Truncated document to {len(document_text)} chars with beginning and end portions")
            
            prompt = f"""
You are an expert document analyst. Answer the question based ONLY on the document below.

DOCUMENT:
{document_text}

QUESTION: {question}

INSTRUCTIONS:
- You MUST provide an answer based on the document provided.
- Extract and synthesize the most relevant information from the document.
- If exact information isn't explicitly stated, provide the closest relevant information.
- NEVER say the information is unavailable or not in the document.
- Be concise but informative.

ANSWER:"""

            logger.debug(f"🤖 Sending full text prompt to Gemini")
            try:
                response = self.model.generate_content(prompt)
                answer = response.text.strip() if response and response.text else ""
            except Exception as api_error:
                logger.error(f"API error with full text: {str(api_error)}")
                document_analysis = self.general_analyzer.analyze_document(document_text)
                return self.general_analyzer.answer_question(question, document_text, document_analysis)
            
            # Clean up the answer
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            # Check for empty or error responses
            if not answer or "not available in" in answer.lower() or "unable to" in answer.lower():
                # Use general analyzer as fallback
                document_analysis = self.general_analyzer.analyze_document(document_text)
                return self.general_analyzer.answer_question(question, document_text, document_analysis)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with full text: {str(e)}")
            try:
                document_analysis = self.general_analyzer.analyze_document(document_text)
                return self.general_analyzer.answer_question(question, document_text, document_analysis)
            except Exception as final_error:
                logger.error(f"Final fallback in _answer_with_full_text failed: {final_error}")
                return "Unable to process the question due to technical difficulties."