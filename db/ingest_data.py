"""
Data ingestion pipeline for the Graph + Vector RAG system.

This module orchestrates the complete data ingestion process from document
parsing through entity extraction, embedding generation, and storage in HelixDB.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from ingestion.parse_docs import DocumentParser
from ingestion.extract_relations import EntityExtractor
from ingestion.embeddings import EmbeddingGenerator
from .helix_interface import ChromaDBInterface

logger = logging.getLogger(__name__)


class DataIngester:
    """
    Orchestrates the complete data ingestion pipeline.

    This class coordinates document parsing, entity extraction, embedding
    generation, and storage in HelixDB for the complete RAG system setup.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data ingester.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)

        # Initialize components
        self.doc_parser = DocumentParser(config_path)
        self.entity_extractor = EntityExtractor(config_path)
        self.embedding_generator = EmbeddingGenerator(config_path)
        self.db_interface = ChromaDBInterface(config_path)

    def ingest_directory(self, directory_path: Optional[str] = None) -> Dict[str, int]:
        """
        Ingest all documents from a directory.

        Args:
            directory_path: Path to directory (uses config default if None)

        Returns:
            Dictionary with ingestion statistics
        """
        if directory_path is None:
            directory_path = self.config.documents.input_dir

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        logger.info(f"Starting ingestion from directory: {directory_path}")

        # Parse documents
        chunks = self.doc_parser.parse_directory(directory_path)

        if not chunks:
            logger.warning("No chunks found to ingest")
            return {"chunks": 0, "entities": 0, "relationships": 0, "edges": 0}

        logger.info(f"Parsed {len(chunks)} chunks from documents")

        # Extract entities and relationships (if enabled)
        entities = []
        relationships = []

        if self.config.entities.enabled:
            try:
                logger.info("Starting entity and relationship extraction...")
                entities = self.entity_extractor.extract_entities_sync(chunks)
                relationships = self.entity_extractor.extract_relationships_sync(chunks)

                logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")

            except Exception as e:
                logger.error(f"Error in entity/relationship extraction: {e}")
                logger.warning("Continuing without entity extraction - documents will still be ingested")
                logger.info("To fix this issue, consider: 1) Using a smaller/faster model, 2) Increasing timeout in config.yaml, 3) Disabling entity extraction in config.yaml")
        else:
            logger.info("Entity extraction disabled in configuration")

        # Generate embeddings
        try:
            chunks_with_embeddings = self.embedding_generator.generate_chunk_embeddings(chunks)

            if entities:
                entities_with_embeddings = self.embedding_generator.generate_entity_embeddings(entities)
            else:
                entities_with_embeddings = []

            if relationships:
                relationships_with_embeddings = self.embedding_generator.generate_relationship_embeddings(relationships)
            else:
                relationships_with_embeddings = []

            logger.info("Generated embeddings for all data")

        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            chunks_with_embeddings = chunks
            entities_with_embeddings = entities
            relationships_with_embeddings = relationships

        # Store in database
        try:
            chunks_stored = self.db_interface.store_chunks(chunks_with_embeddings)
            entities_stored = self.db_interface.store_entities(entities_with_embeddings)
            relationships_stored = self.db_interface.store_relationships(relationships_with_embeddings)

            # Create graph edges from relationships
            edges_created = 0
            if entities_with_embeddings and relationships_with_embeddings:
                edges_created = self._create_graph_edges(entities_with_embeddings, relationships_with_embeddings)

            logger.info("Data stored in HelixDB successfully")

        except Exception as e:
            logger.error(f"Error storing data in database: {e}")
            chunks_stored = 0
            entities_stored = 0
            relationships_stored = 0
            edges_created = 0

        # Return statistics
        stats = {
            "chunks": chunks_stored,
            "entities": entities_stored,
            "relationships": relationships_stored,
            "edges": edges_created
        }

        logger.info(f"Ingestion completed. Stats: {stats}")
        return stats

    def ingest_file(self, file_path: str) -> Dict[str, int]:
        """
        Ingest a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion of file: {file_path}")

        try:
            # Parse document
            chunks = self.doc_parser.parse_file(file_path)

            if not chunks:
                logger.warning(f"No chunks found in file: {file_path}")
                return {"chunks": 0, "entities": 0, "relationships": 0, "edges": 0}

            logger.info(f"Parsed {len(chunks)} chunks from {file_path}")

            # Extract entities and relationships (if enabled)
            entities = []
            relationships = []

            if self.config.entities.enabled:
                try:
                    entities = self.entity_extractor.extract_entities_sync(chunks)
                    relationships = self.entity_extractor.extract_relationships_sync(chunks)

                    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")

                except Exception as e:
                    logger.error(f"Error in entity/relationship extraction: {e}")
                    logger.info("Continuing without entity extraction")
            else:
                logger.info("Entity extraction disabled in configuration")

            # Generate embeddings
            chunks_with_embeddings = self.embedding_generator.generate_chunk_embeddings(chunks)

            # Convert dataclasses to dictionaries for embedding generation
            entities_dict = [entity.to_dict() if hasattr(entity, 'to_dict') else entity.__dict__ for entity in entities]
            relationships_dict = [rel.to_dict() if hasattr(rel, 'to_dict') else rel.__dict__ for rel in relationships]

            entities_with_embeddings = self.embedding_generator.generate_entity_embeddings(entities_dict)
            relationships_with_embeddings = self.embedding_generator.generate_relationship_embeddings(relationships_dict)

            # Store in database
            chunks_stored = self.db_interface.store_chunks(chunks_with_embeddings)
            entities_stored = self.db_interface.store_entities(entities_with_embeddings)
            relationships_stored = self.db_interface.store_relationships(relationships_with_embeddings)

            # Create graph edges
            edges_created = 0
            if entities_with_embeddings and relationships_with_embeddings:
                edges_created = self._create_graph_edges(entities_with_embeddings, relationships_with_embeddings)

            # Return statistics
            stats = {
                "chunks": chunks_stored,
                "entities": entities_stored,
                "relationships": relationships_stored,
                "edges": edges_created
            }

            logger.info(f"File ingestion completed. Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return {"chunks": 0, "entities": 0, "relationships": 0, "edges": 0}

    def _create_graph_edges(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> int:
        """
        Create graph edges from extracted relationships.

        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries

        Returns:
            Number of edges created
        """
        # Create entity name to ID mapping
        entity_name_to_id = {}
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_id = entity.get("id")
            if entity_name and entity_id:
                entity_name_to_id[entity_name] = entity_id

        edges_created = 0

        # Create edges for each relationship
        for relationship in relationships:
            subject_name = relationship.get("subject", "")
            object_name = relationship.get("object", "")
            predicate = relationship.get("predicate", "")

            if subject_name in entity_name_to_id and object_name in entity_name_to_id:
                subject_id = entity_name_to_id[subject_name]
                object_id = entity_name_to_id[object_name]

                try:
                    self.db_interface.store_graph_edges(
                        subject_id,
                        object_id,
                        predicate,
                        weight=relationship.get("confidence", 1.0),
                        metadata={
                            "chunk_id": relationship.get("chunk_id", ""),
                            "source_text": relationship.get("source_text", "")
                        }
                    )
                    edges_created += 1

                except Exception as e:
                    logger.error(f"Error creating graph edge for {subject_name}-{predicate}-{object_name}: {e}")

        logger.info(f"Created {edges_created} graph edges from {len(relationships)} relationships")
        return edges_created

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get overall ingestion statistics."""
        db_stats = self.db_interface.get_stats()

        return {
            "database_stats": db_stats,
            "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            "embedding_model": self.config.embeddings.model_name,
            "llm_model": self.config.llm.model_name
        }

    def clear_all_data(self) -> None:
        """Clear all ingested data."""
        logger.warning("Clearing all ingested data...")
        self.db_interface.clear_database()
        self.embedding_generator.clear_cache()
        logger.info("All data cleared")


def main():
    """Example usage of the DataIngester."""
    logging.basicConfig(level=logging.INFO)

    try:
        ingester = DataIngester()

        # Create sample document for testing
        os.makedirs("documents", exist_ok=True)

        sample_content = """# Machine Learning Overview

Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions without being explicitly programmed.

## Key Concepts

Neural networks are computational models inspired by biological neural networks. They consist of layers of interconnected nodes that process and transform data.

Deep learning is a subset of machine learning that uses deep neural networks with multiple layers to model complex patterns in data.

## Applications

Machine learning is used in various fields including:
- Computer vision for image recognition
- Natural language processing for text analysis
- Recommendation systems for personalized content
- Autonomous vehicles for self-driving cars

## Companies

OpenAI is a leading research organization in AI development. They created GPT models for natural language understanding.

Google uses machine learning extensively in their search algorithms and various products like Google Photos and Google Translate.
"""

        sample_file = "documents/sample_ml.md"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)

        print("Created sample document for testing")

        # Ingest the sample file
        stats = ingester.ingest_file(sample_file)
        print(f"Ingestion stats: {stats}")

        # Get overall stats
        overall_stats = ingester.get_ingestion_stats()
        print(f"Overall system stats: {overall_stats}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
