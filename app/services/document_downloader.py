import httpx
import tempfile
import os
from typing import Tuple, Optional
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DocumentDownloader:
    """Downloads documents from URLs for processing"""
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
    
    def download_document(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Download document from URL
        Returns: (document_content, file_extension)
        """
        try:
            logger.info(f"Downloading document from URL: {url}")
            
            # Parse URL to get file extension
            parsed_url = urlparse(url)
            file_path = parsed_url.path
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Determine file type from extension or URL
            if not file_extension:
                if '.pdf' in url.lower():
                    file_extension = '.pdf'
                elif '.docx' in url.lower():
                    file_extension = '.docx'
                else:
                    file_extension = '.pdf'  # Default assumption
            
            # Download the document
            response = self.client.get(url)
            response.raise_for_status()
            
            logger.info(f"Successfully downloaded document ({len(response.content)} bytes)")
            return response.content, file_extension
            
        except httpx.RequestError as e:
            logger.error(f"Network error downloading document: {str(e)}")
            return None, None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading document: {e.response.status_code}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error downloading document: {str(e)}")
            return None, None
    
    def get_file_type_from_extension(self, extension: str) -> str:
        """Convert file extension to document type"""
        extension = extension.lower()
        if extension == '.pdf':
            return 'pdf'
        elif extension in ['.docx', '.doc']:
            return 'docx'
        elif extension in ['.eml', '.msg']:
            return 'email'
        else:
            return 'pdf'  # Default
    
    def __del__(self):
        """Close HTTP client on cleanup"""
        try:
            self.client.close()
        except:
            pass