"""
Embedding generation module using local models.

This module provides functionality to generate vector embeddings for text chunks
and entities using SentenceTransformers or similar local embedding models.
"""

import os
import json
import logging
import hashlib
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import get_config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks and entities using local models.

    This class handles loading embedding models and generating vector
    representations for text content in a local environment.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.model_name = self.config.embeddings.model_name
        self.batch_size = self.config.embeddings.batch_size
        self.device = self.config.embeddings.device
        self.normalize = self.config.embeddings.normalize

        self.model = None
        self.embedding_dim = None
        self._cache = {}

        # Initialize model
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install it with: pip install sentence-transformers"
            )

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.

        Args:
            chunks: List of text chunks with metadata

        Returns:
            List of chunks with added embedding vectors
        """
        if not chunks:
            return []

        texts = [chunk.get("text", "") for chunk in chunks]
        chunk_ids = [chunk.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(chunks)]

        logger.info(f"Generating embeddings for {len(texts)} text chunks")

        # Generate embeddings in batches
        embeddings = self._generate_embeddings(texts)

        # Add embeddings to chunks
        for i, (chunk, embedding, chunk_id) in enumerate(zip(chunks, embeddings, chunk_ids)):
            chunks[i]["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            chunks[i]["embedding_model"] = self.model_name
            chunks[i]["embedding_dim"] = len(embedding)

        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks

    def generate_entity_embeddings(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for entities.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of entities with added embedding vectors
        """
        if not entities:
            return []

        # Create entity representations for embedding
        entity_texts = []
        for entity in entities:
            # Create a rich representation including entity type and context
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            source_text = entity.get("source_text", "")

            # Create contextual representation
            context = f"{entity_type}: {entity_name}"
            if source_text:
                # Add some context from source text (first 100 chars)
                context += f" in {source_text[:100]}..."

            entity_texts.append(context)

        logger.info(f"Generating embeddings for {len(entity_texts)} entities")

        # Generate embeddings in batches
        embeddings = self._generate_embeddings(entity_texts)

        # Add embeddings to entities
        for i, (entity, embedding) in enumerate(zip(entities, embeddings)):
            entities[i]["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            entities[i]["embedding_model"] = self.model_name
            entities[i]["embedding_dim"] = len(embedding)

        logger.info(f"Generated embeddings for {len(entities)} entities")
        return entities

    def generate_relationship_embeddings(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for relationships.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            List of relationships with added embedding vectors
        """
        if not relationships:
            return []

        # Create relationship representations for embedding
        relationship_texts = []
        for rel in relationships:
            subject = rel.get("subject", "")
            predicate = rel.get("predicate", "")
            obj = rel.get("object", "")

            # Create a structured representation
            rel_text = f"{subject} {predicate} {obj}"
            relationship_texts.append(rel_text)

        logger.info(f"Generating embeddings for {len(relationship_texts)} relationships")

        # Generate embeddings in batches
        embeddings = self._generate_embeddings(relationship_texts)

        # Add embeddings to relationships
        for i, (relationship, embedding) in enumerate(zip(relationships, embeddings)):
            relationships[i]["embedding"] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            relationships[i]["embedding_model"] = self.model_name
            relationships[i]["embedding_dim"] = len(embedding)

        logger.info(f"Generated embeddings for {len(relationships)} relationships")
        return relationships

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        # Check cache first if enabled
        if self.config.performance.cache_embeddings:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    cached_embeddings.append((i, self._cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    # Process in batches
                    all_embeddings = []

                    for i in range(0, len(uncached_texts), self.batch_size):
                        batch = uncached_texts[i:i + self.batch_size]
                        batch_embeddings = self.model.encode(
                            batch,
                            convert_to_numpy=True,
                            normalize_embeddings=self.normalize,
                            batch_size=len(batch)
                        )
                        all_embeddings.append(batch_embeddings)

                    if all_embeddings:
                        new_embeddings = np.vstack(all_embeddings)

                        # Cache new embeddings
                        for text, embedding in zip(uncached_texts, new_embeddings):
                            cache_key = self._get_cache_key(text)
                            self._cache[cache_key] = embedding

                            # Limit cache size
                            if len(self._cache) > self.config.performance.cache_size:
                                # Remove oldest entries (simple FIFO)
                                oldest_keys = list(self._cache.keys())[:len(self._cache) - self.config.performance.cache_size]
                                for key in oldest_keys:
                                    del self._cache[key]

                        # Combine cached and new embeddings
                        result_embeddings = [None] * len(texts)
                        for idx, embedding in cached_embeddings:
                            result_embeddings[idx] = embedding

                        for j, (original_idx, embedding) in enumerate(zip(uncached_indices, new_embeddings)):
                            result_embeddings[original_idx] = embedding

                        return np.array(result_embeddings)
                    else:
                        return np.array([None] * len(texts))

                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    # Fallback: return zero vectors
                    return np.zeros((len(texts), self.embedding_dim))

            else:
                # All embeddings were cached
                result_embeddings = [None] * len(texts)
                for idx, embedding in cached_embeddings:
                    result_embeddings[idx] = embedding
                return np.array(result_embeddings)

        else:
            # No caching - generate all embeddings
            try:
                # Process in batches
                all_embeddings = []

                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        batch_size=len(batch)
                    )
                    all_embeddings.append(batch_embeddings)

                if all_embeddings:
                    return np.vstack(all_embeddings)
                else:
                    return np.array([])

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Fallback: return zero vectors
                return np.zeros((len(texts), self.embedding_dim))

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use hash of text for cache key
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        # Convert to numpy arrays if needed
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_similar_chunks(self,
                          query_embedding: np.ndarray,
                          chunks: List[Dict[str, Any]],
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks most similar to query embedding.

        Args:
            query_embedding: Query embedding vector
            chunks: List of chunks with embeddings
            top_k: Number of top results to return

        Returns:
            List of chunks sorted by similarity score
        """
        similarities = []

        for chunk in chunks:
            chunk_embedding = chunk.get("embedding")
            if chunk_embedding is None:
                continue

            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for chunk, similarity in similarities[:top_k]:
            chunk_copy = chunk.copy()
            chunk_copy["similarity_score"] = similarity
            results.append(chunk_copy)

        return results

    def get_embedding_dimension(self) -> int:
        """Get the dimension of generated embeddings."""
        return self.embedding_dim or 0

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")


def main():
    """Example usage of the EmbeddingGenerator."""
    logging.basicConfig(level=logging.INFO)

    try:
        generator = EmbeddingGenerator()

        # Example text chunks
        sample_chunks = [
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "chunk_id": "chunk_1"
            },
            {
                "text": "Neural networks are computational models inspired by biological neural networks.",
                "chunk_id": "chunk_2"
            }
        ]

        # Generate embeddings
        chunks_with_embeddings = generator.generate_chunk_embeddings(sample_chunks)

        print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
        print(f"Embedding dimension: {generator.get_embedding_dimension()}")

        for chunk in chunks_with_embeddings:
            embedding = chunk.get("embedding", [])
            print(f"Chunk {chunk['chunk_id']}: {len(embedding)} dimensions")

        # Test similarity
        if len(chunks_with_embeddings) >= 2:
            emb1 = np.array(chunks_with_embeddings[0]["embedding"])
            emb2 = np.array(chunks_with_embeddings[1]["embedding"])
            similarity = generator.compute_similarity(emb1, emb2)
            print(f"Similarity between chunks: {similarity:.4f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
