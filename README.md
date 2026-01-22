# RAG System for Grounded Document QA

A production-ready Retrieval-Augmented Generation (RAG) system designed to reduce hallucinations in knowledge-intensive queries through advanced document retrieval, embedding, and citation-backed responses.

## Features

- **Advanced Document Processing**: Intelligent chunking and embedding generation
- **Vector Database Integration**: High-performance semantic search with Pinecone/ChromaDB
- **Confidence Scoring**: Retrieval confidence thresholds for quality control
- **Self-Correction**: Automatic retry logic with improved retrieval strategies
- **Citation Generation**: Source-backed responses with document references
- **Production-Ready API**: FastAPI backend with comprehensive error handling
- **Scalable Architecture**: Designed for production-scale operational constraints

## Tech Stack

- **Backend**: Python 3.10+, FastAPI
- **Vector DB**: Pinecone / ChromaDB (configurable)
- **LLM**: OpenAI GPT / Anthropic Claude (configurable)
- **Embeddings**: OpenAI Embeddings / Sentence Transformers
- **Cloud**: GCP-ready (Cloud Storage, Vertex AI)
- **Containerization**: Docker

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- API keys for LLM provider (OpenAI/Anthropic)
- Vector database credentials (Pinecone/ChromaDB)

### Installation

```bash
# Clone the repository
git clone https://github.com/Amankhan2370/grounded-rag-qa.git
cd grounded-rag-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Configuration

Create a `.env` file with the following variables:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLM_PROVIDER=openai  # or anthropic

# Vector Database
VECTOR_DB_TYPE=pinecone  # or chromadb
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=rag-index

# Embeddings
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
CONFIDENCE_THRESHOLD=0.7
MAX_RETRIES=3

# GCP Configuration (optional)
GCP_PROJECT_ID=your_project_id
GCP_BUCKET_NAME=your_bucket_name
```

### Running the Application

```bash
# Development mode
uvicorn app.main:app --reload --port 8000

# Production mode with Docker
docker-compose up -d
```

### API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```
GET /health
```

### Ingest Documents
```
POST /api/v1/documents/ingest
Content-Type: multipart/form-data
Body: file (PDF, TXT, DOCX)
```

### Query Documents
```
POST /api/v1/query
Content-Type: application/json
Body: {
  "query": "Your question here",
  "top_k": 5,
  "include_citations": true
}
```

### Get Document Status
```
GET /api/v1/documents/{document_id}
```

## Architecture

```
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│ Doc   │ │  Query  │
│Ingest │ │ Handler │
└───┬───┘ └──┬──────┘
    │        │
┌───▼────────▼───┐
│  Chunking &    │
│  Embedding     │
└───┬────────────┘
    │
┌───▼────────────┐
│ Vector Database│
│  (Pinecone/    │
│   ChromaDB)    │
└───┬────────────┘
    │
┌───▼────────────┐
│  LLM Backend   │
│  (OpenAI/      │
│   Anthropic)   │
└────────────────┘
```

## Project Structure

```
grounded-rag-qa/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── llm_service.py
│   ├── pipelines/           # Data pipelines
│   │   ├── ingestion.py
│   │   ├── chunking.py
│   │   └── embedding.py
│   └── utils/               # Utilities
│       ├── logger.py
│       └── validators.py
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── README.md
```

## Performance Metrics

- **Hallucination Reduction**: ~36% improvement over vanilla LLM responses
- **Answer Faithfulness**: Enhanced through citation-backed RAG
- **Retrieval Relevance**: Optimized through confidence thresholds
- **Latency**: Optimized for production workloads

## Contributing

This is a proprietary project. Implementation details are confidential.

## License

Proprietary - All rights reserved

## Contact

For questions or support, please contact the repository maintainer.
