# 🎯 LLM Document Processing System

A production-ready AI-powered document processing system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from unstructured documents such as policy documents, contracts, and emails.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, Email processing
- **Natural Language Queries**: Handles vague, incomplete, or plain English queries
- **Semantic Understanding**: Beyond simple keyword matching using vector embeddings
- **Structured Decision Generation**: Approval/rejection with justification and amounts
- **Clause References**: Maps decisions to specific document clauses
- **RESTful API**: Easy integration with external systems
- **HackRX Compatible**: Matches exact API specification for competitions
- **Authentication**: JWT-based security
- **Health Monitoring**: System status and diagnostics

## 🏗️ Architecture

```
bajaj/
├── app/
│   ├── main.py              # FastAPI application with REST endpoints
│   ├── config.py           # Configuration management
│   ├── models.py           # Pydantic data models
│   └── services/           # Core business logic
│       ├── document_loader.py      # Parse PDFs, DOCX, emails
│       ├── document_downloader.py  # Download documents from URLs
│       ├── embeddings.py          # Vector embeddings & Pinecone storage
│       ├── semantic_search.py     # Similarity search with reranking
│       ├── query_parser.py        # Extract entities from queries
│       ├── answer_generator.py    # Generate decisions using Gemini
│       ├── question_answerer.py   # Multi-question answering service
│       └── auth.py               # Authentication service
├── requirements.txt        # Python dependencies
└── Procfile               # Deployment configuration
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+ (recommended)
- pip package manager

### 1. Clone Repository
```bash
git clone https://github.com/pranjal269/LLM_baja.git
cd LLM_baja
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:

```bash
# API Keys - Get from respective services
GEMINI_API_KEY=your-gemini-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here

# Pinecone Configuration
PINECONE_INDEX_NAME=document-processing
EMBEDDING_DIMENSION=384

# Authentication
SECRET_KEY=your-secret-key-here-change-this-in-production

# Application Settings
MAX_FILE_SIZE=52428800
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# LLM Settings
GEMINI_MODEL=gemini-pro
MAX_TOKENS=2048
TEMPERATURE=0.1
```

### 5. Get API Keys
- **Gemini API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Pinecone API Key**: [Pinecone Console](https://app.pinecone.io/) (free tier available)

### 6. Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at: **http://localhost:8000**

## 📊 API Endpoints

| Endpoint | Method | Description | Format |
|----------|--------|-------------|---------|
| `/` | GET | System info and API format | Standard |
| `/health` | GET | Health check | Standard |
| `/demo-token` | GET | Get authentication token | Standard |
| **`/hackrx/run`** | POST | **Main HackRX endpoint** | Competition Format |
| `/query` | POST | Legacy single query processing | Legacy |
| `/upload-document` | POST | Upload documents | Standard |
| `/search` | POST | Search without decisions | Standard |

## 🎯 HackRX API Format

### Request Format
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

### Response Format
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six (36) months...",
        "Yes, the policy covers maternity expenses with specific conditions..."
    ]
}
```

## 🧪 Testing

### Using cURL
```bash
# Get authentication token
TOKEN=$(curl -s http://localhost:8000/demo-token | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

# Test HackRX endpoint
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "DOCUMENT_URL_HERE",
    "questions": ["Question 1?", "Question 2?"]
  }'
```

### Using Postman
1. Get demo token from `/demo-token` endpoint
2. Set Authorization header: `Bearer <token>`
3. Use `/hackrx/run` endpoint with JSON body

## 🎨 Use Cases

- **Insurance Claims Processing**: Automated claim approval/rejection
- **Legal Document Analysis**: Contract clause extraction and analysis
- **Policy Compliance**: Regulatory compliance checking
- **HR Document Processing**: Employee handbook queries
- **Research Paper Analysis**: Academic document Q&A

## 🔧 Development

### Running Tests
```bash
python -m pytest test/
```

### Code Structure
- **Services**: Modular business logic components
- **Models**: Pydantic data validation
- **Config**: Environment-based configuration
- **Error Handling**: Graceful fallbacks and meaningful errors

## 🚀 Deployment

### Production Deployment
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📝 License

This project is licensed under the MIT License.

## 🔗 Links

- **Repository**: [https://github.com/pranjal269/LLM_baja.git](https://github.com/pranjal269/LLM_baja.git)
- **API Documentation**: Visit `/docs` endpoint when server is running
- **Health Check**: Visit `/health` endpoint

## 🆘 Support

For issues and questions:
1. Check the `/health` endpoint for system status
2. Review logs in terminal output
3. Verify API keys in `.env` file
4. Open GitHub issue for bugs

---

**Built with ❤️ for intelligent document processing**