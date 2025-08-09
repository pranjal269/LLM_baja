import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-processing")
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension
    
    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Application Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 500  # tokens per chunk
    CHUNK_OVERLAP: int = 50  # overlap between chunks
    
    # LLM Settings
    GEMINI_MODEL: str = "gemini-pro"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.1
    
    class Config:
        env_file = ".env"

settings = Settings()