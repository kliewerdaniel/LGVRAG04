#!/usr/bin/env python3
"""
Test script to demonstrate the Graph + Vector RAG system functionality.
This script tests all core components without relying on FastAPI.
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing Configuration System...")
    try:
        from config import get_config
        config = get_config()
        print(f"✅ Config loaded successfully!")
        print(f"   Database path: {config.database.path}")
        print(f"   Embedding model: {config.embeddings.model_name}")
        print(f"   LLM model: {config.llm.model_name}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_document_parsing():
    """Test document parsing functionality."""
    print("\n📄 Testing Document Parsing...")
    try:
        from ingestion.parse_docs import DocumentParser
        parser = DocumentParser()

        # Test parsing a sample document
        chunks = parser.parse_file("documents/sample_ml.md")
        print(f"✅ Document parsing successful!")
        print(f"   Parsed {len(chunks)} chunks")
        if chunks:
            print(f"   First chunk preview: {chunks[0]['text'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Document parsing test failed: {e}")
        return False

def test_database_operations():
    """Test database operations."""
    print("\n💾 Testing Database Operations...")
    try:
        from db.helix_interface import ChromaDBInterface
        db = ChromaDBInterface()
        stats = db.get_stats()
        print(f"✅ Database operations successful!")
        print(f"   Current database stats: {stats}")
        return True
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation."""
    print("\n🧠 Testing Embedding Generation...")
    try:
        from ingestion.embeddings import EmbeddingGenerator
        generator = EmbeddingGenerator()
        print(f"✅ Embedding generator initialized!")
        print(f"   Model: {generator.model_name}")
        print(f"   Embedding dimension: {generator.get_embedding_dimension()}")

        # Test generating embeddings for sample text
        sample_texts = ["Machine learning is a subset of artificial intelligence."]
        embeddings = generator._generate_embeddings(sample_texts)
        print(f"   Generated embeddings with shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"❌ Embedding generation test failed: {e}")
        return False

def test_data_ingestion():
    """Test the complete data ingestion pipeline."""
    print("\n🔄 Testing Data Ingestion Pipeline...")
    try:
        from db.ingest_data import DataIngester
        ingester = DataIngester()

        # Ingest the sample document
        stats = ingester.ingest_file("documents/sample_ml.md")
        print(f"✅ Data ingestion successful!")
        print(f"   Ingestion stats: {stats}")
        return True
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return False

def test_retrieval():
    """Test the hybrid retrieval system."""
    print("\n🔍 Testing Hybrid Retrieval...")
    try:
        from rag.retrieve import HybridRetriever
        retriever = HybridRetriever()

        # Test retrieval
        query = "What is machine learning?"
        results = retriever.retrieve(query, top_k=3)
        print(f"✅ Retrieval successful!")
        print(f"   Query: {query}")
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('text', '')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False

def test_answer_generation():
    """Test answer generation."""
    print("\n🤖 Testing Answer Generation...")
    try:
        from rag.generate_answer import AnswerGenerator
        from rag.retrieve import HybridRetriever

        # Get some context first
        retriever = HybridRetriever()
        results = retriever.retrieve("What is machine learning?", top_k=2)

        # Generate answer
        generator = AnswerGenerator()
        answer = generator.generate_answer_sync("What is machine learning?", results)

        print(f"✅ Answer generation successful!")
        print(f"   Answer: {answer.answer[:200]}...")
        print(f"   Generation time: {answer.generation_time:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Answer generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Graph + Vector RAG System - Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        test_configuration,
        test_document_parsing,
        test_database_operations,
        test_embedding_generation,
        test_data_ingestion,
        test_retrieval,
        test_answer_generation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The RAG system is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Start a local LLM server (e.g., Ollama)")
        print("2. Run the API server: python3 -m api.main")
        print("3. Test the API endpoints with curl or a REST client")
        print("4. Upload documents and query the system")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
