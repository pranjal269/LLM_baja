import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-processing")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Application Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "52428800"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # LLM Settings
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    # Query Processing Settings
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    MAX_RETRIEVED_CHUNKS: int = int(os.getenv("MAX_RETRIEVED_CHUNKS", "5"))
    
    # Debug Settings
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Fallback Settings
    FALLBACK_RESPONSE: bool = os.getenv("FALLBACK_RESPONSE", "true").lower() == "true"
    MIN_CONFIDENCE_SCORE: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.3"))
    
    # Extended Settings
    RESPONSE_FORMAT: str = os.getenv("RESPONSE_FORMAT", "json")
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    ENABLE_QUERY_LOGGING: bool = os.getenv("ENABLE_QUERY_LOGGING", "false").lower() == "true"
    ENABLE_RESPONSE_LOGGING: bool = os.getenv("ENABLE_RESPONSE_LOGGING", "false").lower() == "true"
    ENABLE_ERROR_DETAILS: bool = os.getenv("ENABLE_ERROR_DETAILS", "false").lower() == "true"
    ENABLE_PARTIAL_ANSWERS: bool = os.getenv("ENABLE_PARTIAL_ANSWERS", "false").lower() == "true"
    ENABLE_PREPROCESSING: bool = os.getenv("ENABLE_PREPROCESSING", "false").lower() == "true"
    ENABLE_POSTPROCESSING: bool = os.getenv("ENABLE_POSTPROCESSING", "false").lower() == "true"
    SAFE_MODE: bool = os.getenv("SAFE_MODE", "false").lower() == "true"
    
    class Config:
        env_file = ".env"

settings = Settings()