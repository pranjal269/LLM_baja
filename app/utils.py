import re
import unicodedata
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-"\'%$₹]', ' ', text)
    
    return text.strip()

def extract_amount_from_text(text: str) -> Optional[float]:
    """Extract monetary amounts from text"""
    # Patterns for Indian currency
    patterns = [
        r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|inr|₹|rupees?)',
        r'(\d+(?:,\d{3})*)\s*lakhs?',
        r'(\d+(?:,\d{3})*)\s*crores?'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            amount_str = matches[0].replace(',', '')
            try:
                amount = float(amount_str)
                # Convert lakhs and crores
                if 'lakh' in text.lower():
                    amount *= 100000
                elif 'crore' in text.lower():
                    amount *= 10000000
                return amount
            except ValueError:
                continue
    
    return None

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(clean_text(text1.lower()).split())
    words2 = set(clean_text(text2.lower()).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def generate_chunk_id(document_name: str, chunk_index: int, content: str) -> str:
    """Generate a unique chunk ID"""
    # Create a hash of the content for uniqueness
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{document_name}_{chunk_index}_{content_hash}"

def format_indian_currency(amount: float) -> str:
    """Format amount in Indian currency format"""
    if amount >= 10000000:  # 1 crore
        crores = amount / 10000000
        return f"₹{crores:.2f} crores"
    elif amount >= 100000:  # 1 lakh
        lakhs = amount / 100000
        return f"₹{lakhs:.2f} lakhs"
    else:
        return f"₹{amount:,.2f}"

def validate_age(age: Any) -> Optional[int]:
    """Validate and convert age to integer"""
    try:
        age_int = int(age)
        if 0 <= age_int <= 120:
            return age_int
    except (ValueError, TypeError):
        pass
    return None

def extract_dates_from_text(text: str) -> List[str]:
    """Extract dates from text"""
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}\b',
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{2,4}\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return dates

def normalize_medical_terms(text: str) -> str:
    """Normalize medical terminology"""
    # Common medical term mappings
    medical_mappings = {
        'cardiac': 'heart',
        'orthopedic': 'bone',
        'ophthalmology': 'eye',
        'neurology': 'brain',
        'oncology': 'cancer',
        'gynaecology': 'gynecology',
        'paediatrics': 'pediatrics'
    }
    
    normalized_text = text.lower()
    for original, normalized in medical_mappings.items():
        normalized_text = normalized_text.replace(original, normalized)
    
    return normalized_text

def get_file_type_from_filename(filename: str) -> Optional[str]:
    """Determine file type from filename"""
    if not filename:
        return None
    
    extension = filename.lower().split('.')[-1]
    type_mapping = {
        'pdf': 'pdf',
        'docx': 'docx',
        'doc': 'docx',
        'eml': 'email',
        'txt': 'text'
    }
    
    return type_mapping.get(extension)

def chunk_overlap_score(chunk1_text: str, chunk2_text: str) -> float:
    """Calculate overlap score between two chunks"""
    words1 = set(chunk1_text.lower().split())
    words2 = set(chunk2_text.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1.intersection(words2))
    min_length = min(len(words1), len(words2))
    
    return overlap / min_length if min_length > 0 else 0.0

def create_search_filters(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Create search filters from extracted entities"""
    filters = {}
    
    # Add location filter if available
    if entities.get('location'):
        # This would need to be customized based on your metadata structure
        filters['location'] = entities['location']
    
    # Add procedure type filter
    if entities.get('procedure'):
        filters['procedure_type'] = entities['procedure']
    
    return filters

def log_processing_metrics(
    query: str,
    processing_time_ms: int,
    num_chunks_found: int,
    decision: str,
    confidence: float
):
    """Log processing metrics for monitoring"""
    logger.info(f"METRICS - Query: '{query[:50]}...', "
               f"Time: {processing_time_ms}ms, "
               f"Chunks: {num_chunks_found}, "
               f"Decision: {decision}, "
               f"Confidence: {confidence:.3f}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 100:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:95] + '.' + ext if ext else name[:100]
    
    return filename

def parse_policy_duration(duration_str: str) -> Optional[Dict[str, Any]]:
    """Parse policy duration string into structured format"""
    if not duration_str:
        return None
    
    duration_lower = duration_str.lower()
    
    # Extract number and unit
    match = re.search(r'(\d+)\s*(month|mon|year|yr)s?', duration_lower)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        
        # Normalize unit
        if unit in ['month', 'mon']:
            unit = 'months'
        elif unit in ['year', 'yr']:
            unit = 'years'
        
        return {
            'number': number,
            'unit': unit,
            'total_months': number if unit == 'months' else number * 12
        }
    
    return None