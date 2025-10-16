"""
ChromaDB interface for the Graph + Vector RAG system.

This module provides a high-level interface for interacting with ChromaDB,
handling both vector and graph data storage and retrieval.
"""

import os
import json
import logging
import sqlite3
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

logger = logging.getLogger(__name__)


class ChromaDBInterface:
    """
    Interface for ChromaDB operations with SQLite for graph data.

    This class provides methods for storing and retrieving both vector
    embeddings using ChromaDB and graph data using SQLite.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ChromaDB interface.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.db_path = self.config.database.path
        self.vector_dim = self.config.database.vector_dim
        self.max_connections = self.config.database.max_connections

        # Initialize databases
        self._init_databases()

    def _init_databases(self) -> None:
        """Initialize ChromaDB and SQLite databases."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collections for chunks and entities
            self.chunks_collection = self.chroma_client.get_or_create_collection(
                name="chunks",
                metadata={"dimension": self.vector_dim}
            )
            self.entities_collection = self.chroma_client.get_or_create_collection(
                name="entities",
                metadata={"dimension": self.vector_dim}
            )

            # Initialize SQLite for graph data
            graph_db_path = os.path.join(self.db_path, "graph.db")
            os.makedirs(self.db_path, exist_ok=True)
            self.sqlite_conn = sqlite3.connect(graph_db_path)
            self._create_graph_tables()

            logger.info(f"ChromaDB initialized at: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _create_graph_tables(self) -> None:
        """Create SQLite tables for graph data."""
        cursor = self.sqlite_conn.cursor()

        # Table for storing entities (metadata only, embeddings in ChromaDB)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                confidence REAL,
                chunk_id TEXT,
                source_text TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                metadata TEXT,  -- JSON metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for storing relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL,
                chunk_id TEXT,
                source_text TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                metadata TEXT,  -- JSON metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for storing graph edges (for efficient traversal)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,  -- JSON metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES entities (id),
                FOREIGN KEY (object_id) REFERENCES entities (id)
            )
        """)

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(predicate)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(object)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_subject ON graph_edges(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_object ON graph_edges(object_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_predicate ON graph_edges(predicate)")

        self.sqlite_conn.commit()

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store text chunks in ChromaDB and SQLite.

        Args:
            chunks: List of chunk dictionaries with embeddings

        Returns:
            Number of chunks stored
        """
        if not chunks:
            return 0

        stored_count = 0
        chunk_ids = []
        chunk_texts = []
        chunk_embeddings = []
        chunk_metadatas = []

        for chunk in chunks:
            try:
                # Generate unique ID
                chunk_id = chunk.get("chunk_id", "")
                chunk_text = chunk.get("text", "")
                unique_id = self._generate_id(f"chunk_{chunk_id}_{hash(chunk_text) % 10000}")

                # Prepare data for ChromaDB
                embedding = chunk.get("embedding")
                if embedding is not None:
                    chunk_ids.append(unique_id)
                    chunk_texts.append(chunk.get("text", ""))
                    chunk_embeddings.append(embedding)

                    # Prepare metadata for ChromaDB
                    metadata = {
                        "source": chunk.get("source", ""),
                        "file_type": chunk.get("file_type", ""),
                        "chunk_id": chunk_id,
                        "embedding_model": chunk.get("embedding_model", ""),
                        "embedding_dim": chunk.get("embedding_dim", 0)
                    }
                    # Add any additional metadata
                    for k, v in chunk.items():
                        if k not in ["text", "embedding", "chunk_id", "source", "file_type", "embedding_model", "embedding_dim"]:
                            metadata[k] = v

                    chunk_metadatas.append(metadata)

                stored_count += 1

            except Exception as e:
                logger.error(f"Error preparing chunk {chunk.get('chunk_id', 'unknown')}: {e}")

        # Store in ChromaDB if we have embeddings
        if chunk_embeddings:
            try:
                self.chunks_collection.add(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas
                )
                logger.info(f"Stored {len(chunk_embeddings)} chunks in ChromaDB")
            except Exception as e:
                logger.error(f"Error storing chunks in ChromaDB: {e}")
                stored_count = 0

        logger.info(f"Successfully prepared {stored_count} chunks for storage")
        return stored_count

    def store_entities(self, entities: List[Dict[str, Any]]) -> int:
        """
        Store entities in the database.

        Args:
            entities: List of entity dictionaries with embeddings

        Returns:
            Number of entities stored
        """
        if not entities:
            return 0

        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        stored_count = 0

        for entity in entities:
            try:
                # Generate unique ID
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "")
                unique_id = self._generate_id(f"entity_{entity_name}_{entity_type}")

                # Prepare data
                embedding = entity.get("embedding")
                embedding_json = json.dumps(embedding) if embedding else None

                metadata = {k: v for k, v in entity.items()
                           if k not in ["name", "type", "confidence", "embedding", "chunk_id", "source_text"]}

                cursor.execute("""
                    INSERT OR REPLACE INTO entities
                    (id, name, type, confidence, chunk_id, source_text, embedding,
                     embedding_model, embedding_dim, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_id,
                    entity.get("name", ""),
                    entity.get("type", ""),
                    entity.get("confidence", 0.0),
                    entity.get("chunk_id", ""),
                    entity.get("source_text", ""),
                    embedding_json,
                    entity.get("embedding_model", ""),
                    entity.get("embedding_dim", 0),
                    json.dumps(metadata)
                ))

                stored_count += 1

            except Exception as e:
                logger.error(f"Error storing entity {entity.get('name', 'unknown')}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Stored {stored_count} entities in database")
        return stored_count

    def store_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        Store relationships in the database.

        Args:
            relationships: List of relationship dictionaries with embeddings

        Returns:
            Number of relationships stored
        """
        if not relationships:
            return 0

        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        stored_count = 0

        for rel in relationships:
            try:
                # Generate unique ID
                subject = rel.get("subject", "")
                obj = rel.get("object", "")
                predicate = rel.get("predicate", "")
                unique_id = self._generate_id(f"rel_{subject}_{predicate}_{obj}")

                # Prepare data
                embedding = rel.get("embedding")
                embedding_json = json.dumps(embedding) if embedding else None

                metadata = {k: v for k, v in rel.items()
                           if k not in ["subject", "predicate", "object", "confidence", "embedding", "chunk_id", "source_text"]}

                cursor.execute("""
                    INSERT OR REPLACE INTO relationships
                    (id, subject, predicate, object, confidence, chunk_id, source_text,
                     embedding, embedding_model, embedding_dim, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_id,
                    subject,
                    predicate,
                    obj,
                    rel.get("confidence", 0.0),
                    rel.get("chunk_id", ""),
                    rel.get("source_text", ""),
                    embedding_json,
                    rel.get("embedding_model", ""),
                    rel.get("embedding_dim", 0),
                    json.dumps(metadata)
                ))

                stored_count += 1

            except Exception as e:
                logger.error(f"Error storing relationship {rel.get('subject', 'unknown')}-{rel.get('predicate', 'unknown')}-{rel.get('object', 'unknown')}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Stored {stored_count} relationships in database")
        return stored_count

    def store_graph_edges(self, subject_entity_id: str, object_entity_id: str,
                         predicate: str, weight: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a graph edge between two entities.

        Args:
            subject_entity_id: ID of subject entity
            object_entity_id: ID of object entity
            predicate: Relationship predicate
            weight: Edge weight
            metadata: Additional metadata

        Returns:
            ID of the created edge
        """
        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        try:
            edge_id = self._generate_id(f"edge_{subject_entity_id}_{object_entity_id}_{predicate}")

            cursor.execute("""
                INSERT OR REPLACE INTO graph_edges
                (id, subject_id, object_id, predicate, weight, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                subject_entity_id,
                object_entity_id,
                predicate,
                weight,
                json.dumps(metadata) if metadata else None
            ))

            conn.commit()
            logger.debug(f"Stored graph edge: {subject_entity_id} -{predicate}-> {object_entity_id}")

            return edge_id

        except Exception as e:
            logger.error(f"Error storing graph edge: {e}")
            raise
        finally:
            conn.close()

    def search_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to query embedding using ChromaDB.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of similar chunks with similarity scores
        """
        try:
            # Use ChromaDB's built-in similarity search
            results = self.chunks_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i] if results['documents'] and results['documents'][0] else "",
                        "source": results['metadatas'][0][i].get('source', '') if results['metadatas'] and results['metadatas'][0] else '',
                        "chunk_id": results['metadatas'][0][i].get('chunk_id', '') if results['metadatas'] and results['metadatas'][0] else '',
                        "similarity": 1.0 - distance,  # Convert distance to similarity
                        "embedding_model": results['metadatas'][0][i].get('embedding_model', '') if results['metadatas'] and results['metadatas'][0] else ''
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching similar chunks in ChromaDB: {e}")
            return []

    def search_similar_entities(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for entities similar to query embedding.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of similar entities with similarity scores
        """
        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        try:
            # Get all entities with embeddings
            cursor.execute("""
                SELECT id, name, type, confidence, embedding, embedding_model
                FROM entities
                WHERE embedding IS NOT NULL
            """)

            results = []

            for row in cursor.fetchall():
                entity_id, name, entity_type, confidence, embedding_json, embedding_model = row

                if embedding_json:
                    entity_embedding = np.array(json.loads(embedding_json))

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, entity_embedding)

                    results.append({
                        "id": entity_id,
                        "name": name,
                        "type": entity_type,
                        "confidence": confidence,
                        "similarity": similarity,
                        "embedding_model": embedding_model
                    })

            # Sort by similarity and return top k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error searching similar entities: {e}")
            return []
        finally:
            conn.close()

    def get_entity_relationships(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get relationships for a given entity using graph traversal.

        Args:
            entity_name: Name of the entity
            max_depth: Maximum traversal depth

        Returns:
            List of related entities and relationships
        """
        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        try:
            # Find entity by name
            cursor.execute("""
                SELECT id, name, type FROM entities WHERE name = ?
            """, (entity_name,))

            entity_row = cursor.fetchone()
            if not entity_row:
                return []

            entity_id, _, entity_type = entity_row

            # Get direct relationships
            cursor.execute("""
                SELECT ge.predicate, e2.name, e2.type, ge.weight
                FROM graph_edges ge
                JOIN entities e1 ON ge.subject_id = e1.id
                JOIN entities e2 ON ge.object_id = e2.id
                WHERE ge.subject_id = ? OR ge.object_id = ?
            """, (entity_id, entity_id))

            relationships = []
            for row in cursor.fetchall():
                predicate, related_name, related_type, weight = row

                relationships.append({
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "related_entity": related_name,
                    "related_type": related_type,
                    "relationship": predicate,
                    "weight": weight,
                    "depth": 1
                })

            return relationships

        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return []
        finally:
            conn.close()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with timestamp."""
        import time
        import hashlib

        timestamp = str(int(time.time() * 1000000))  # Microseconds
        unique_string = f"{prefix}_{timestamp}_{id(self)}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        try:
            stats = {}

            # Count records in each table
            tables = ["chunks", "entities", "relationships", "graph_edges"]

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            conn.close()

    def clear_database(self) -> None:
        """Clear all data from the database."""
        graph_db_path = os.path.join(self.db_path, "graph.db")
        conn = sqlite3.connect(graph_db_path)
        cursor = conn.cursor()

        try:
            tables = ["chunks", "entities", "relationships", "graph_edges"]
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")

            conn.commit()
            logger.info("Database cleared")

        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
        finally:
            conn.close()


def main():
    """Example usage of ChromaDBInterface."""
    logging.basicConfig(level=logging.INFO)

    try:
        db = ChromaDBInterface()

        # Example data
        sample_chunks = [
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "chunk_id": "chunk_1",
                "source": "test.txt",
                "file_type": "text",
                "embedding": [0.1] * 384,  # Mock embedding
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384
            }
        ]

        # Store chunks
        stored = db.store_chunks(sample_chunks)
        print(f"Stored {stored} chunks")

        # Get stats
        stats = db.get_stats()
        print(f"Database stats: {stats}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
