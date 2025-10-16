# Graph + Vector Retrieval-Augmented Generation (RAG) System

A fully local RAG system that combines graph-based knowledge representation with vector similarity search for enhanced retrieval and reasoning.

## Features

- **Document Ingestion**: Supports PDF, Markdown, Word, Excel, and PowerPoint files
- **Knowledge Graph Construction**: Automatically extracts entities and relationships using local LLMs
- **Vector Embeddings**: Generates embeddings locally using sentence-transformers for semantic similarity search
- **ChromaDB Integration**: Uses ChromaDB as the vector database for efficient similarity search
- **Hybrid Retrieval**: Combines graph traversal with vector similarity for optimal results
- **Local LLM Inference**: Uses local language models via Ollama for answer generation
- **REST API**: FastAPI endpoint for querying the system
- **Health Checks**: Validates Ollama connectivity and model availability before processing
- **Error Handling**: Comprehensive error handling and validation throughout the pipeline

## Architecture

The system is organized into modular components:

- **ingestion/**: Document parsing and preprocessing
- **db/**: ChromaDB integration and SQLite for graph data management
- **rag/**: Retrieval and generation logic with Ollama health checks
- **api/**: FastAPI endpoints and server
- **config.yaml**: Configuration management
- **requirements.txt**: Python dependencies (Python 3.13 compatible)

## Quick Start

### Prerequisites
- Python 3.13+
- Virtual environment (already set up in `.venv/`)
- Local LLM server (Ollama recommended)

### Setup Steps

1. **Install dependencies:**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install and start local LLM (Ollama):**
   ```bash
   # Install Ollama from https://ollama.ai/
   # Pull a lightweight model:
   ollama pull granite4:micro-h
   # or
   ollama pull llama3.2:1b

   # Start the server in background
   ollama serve
   ```

3. **Place documents in `documents/` directory**
   - Sample documents are already provided
   - Supported formats: PDF, Markdown, Word, Excel, PowerPoint, plain text

4. **Test the system:**
   ```bash
   source .venv/bin/activate
   python3 test_system.py
   ```

5. **Ingest documents:**
   ```bash
   source .venv/bin/activate
   python3 -c "from db.ingest_data import DataIngester; ingester = DataIngester(); ingester.ingest_file('documents/sample_ml.md')"
   ```

6. **Start the API server:**
   ```bash
   source .venv/bin/activate
   python3 -m api.main
   ```

7. **Query the system:**
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'
   ```

## API Endpoints

### Query the System
```bash
POST /query
Content-Type: application/json

{
  "query": "What is machine learning?",
  "top_k": 5,
  "include_entities": true
}
```

### Ingest Documents
```bash
POST /ingest
Content-Type: application/x-www-form-urlencoded

file_path=documents/sample.md
```

### Upload and Ingest
```bash
POST /upload
Content-Type: multipart/form-data

file: @document.pdf
```

### System Statistics
```bash
GET /stats
```

### Health Check
```bash
GET /health
```

## Configuration

Edit `config.yaml` to customize:
- Model settings (embedding and LLM models)
- Database paths
- Chunking parameters
- API settings

### Key Configuration Options

```yaml
# Embedding model (local only)
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  device: "cpu"

# Local LLM settings
llm:
  base_url: "http://localhost:11434"
  model_name: "llama2:7b"
  temperature: 0.1

# Hybrid retrieval settings
retrieval:
  vector_top_k: 5
  graph_depth: 2
  hybrid_alpha: 0.7  # Balance between vector and graph search
```

## Local-Only Design

This system is designed to run entirely locally:
- No external API calls
- All models and embeddings are local
- Data stays on your machine
- Full privacy and control

## Project Structure

```
graph_vector_rag/
├── ingestion/           # Document parsing and preprocessing
│   ├── parse_docs.py   # Multi-format document parser
│   ├── extract_relations.py  # Entity and relationship extraction
│   └── embeddings.py   # Local embedding generation
├── db/                 # Database and storage
│   ├── helix_interface.py   # ChromaDB + SQLite operations
│   └── ingest_data.py  # Data ingestion pipeline
├── rag/                # Retrieval and generation
│   ├── retrieve.py     # Hybrid retrieval system
│   └── generate_answer.py   # Local LLM answer generation with health checks
├── api/                # REST API
│   └── main.py         # FastAPI application
├── config.yaml         # Configuration settings
├── requirements.txt    # Python dependencies (Python 3.13 compatible)
└── README.md          # This file
```

## Development

See `ledger.md` for detailed development progress and implementation notes.

## Requirements

- Python 3.8+
- Local LLM server (Ollama recommended)
- Sufficient RAM for embedding models
- Storage space for documents and database

## License

This project is designed for local, private use with full data control.
