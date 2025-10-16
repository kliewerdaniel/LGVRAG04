"""
Hybrid retrieval module combining vector similarity and graph traversal.

This module implements sophisticated retrieval strategies that combine
semantic similarity search with knowledge graph traversal for enhanced
contextual understanding and retrieval accuracy.
"""

import os
import json
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from db.helix_interface import ChromaDBInterface
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Implements hybrid retrieval combining vector similarity and graph traversal.

    This class provides sophisticated retrieval strategies that leverage both
    semantic similarity of text chunks and relationships in the knowledge graph
    to provide more accurate and contextually relevant results.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the hybrid retriever.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.db_interface = ChromaDBInterface(config_path)
        self.embedding_generator = EmbeddingGenerator(config_path)

        # Retrieval parameters
        self.vector_top_k = self.config.retrieval.vector_top_k
        self.graph_depth = self.config.retrieval.graph_depth
        self.hybrid_alpha = self.config.retrieval.hybrid_alpha
        self.rerank_top_k = self.config.retrieval.rerank_top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for a query.

        Args:
            query: Search query string
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of retrieved chunks with relevance scores
        """
        if top_k is None:
            top_k = self.rerank_top_k

        logger.info(f"Performing hybrid retrieval for query: {query}")

        # Generate embedding for query
        query_embedding = self.embedding_generator._generate_embeddings([query])
        if len(query_embedding) == 0:
            logger.error("Failed to generate query embedding")
            return []

        query_embedding = query_embedding[0]

        # Step 1: Vector similarity search
        vector_results = self._vector_search(query_embedding, self.vector_top_k * 2)

        # Step 2: Graph-based retrieval
        graph_results = self._graph_search(query, self.vector_top_k * 2)

        # Step 3: Combine and rerank results
        combined_results = self._combine_results(query, query_embedding, vector_results, graph_results)

        # Step 4: Rerank and return top results
        final_results = self._rerank_results(combined_results, top_k)

        logger.info(f"Retrieved {len(final_results)} results for query")
        return final_results

    def retrieve_with_entities(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve results with entity information.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            Dictionary containing chunks, entities, and relationships
        """
        # Get basic retrieval results
        chunks = self.retrieve(query, top_k)

        # Extract entities from query and results
        query_entities = self._extract_query_entities(query)

        # Find related entities in results
        result_entities = []
        relationships = []

        for chunk in chunks:
            # Find entities mentioned in this chunk
            chunk_entities = self._find_entities_in_chunk(chunk)
            result_entities.extend(chunk_entities)

            # Find relationships involving these entities
            chunk_relationships = self._find_relationships_in_chunk(chunk)
            relationships.extend(chunk_relationships)

        return {
            "query": query,
            "query_entities": query_entities,
            "chunks": chunks,
            "entities": result_entities,
            "relationships": relationships
        }

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            results = self.db_interface.search_similar_chunks(query_embedding, top_k)

            # Add retrieval method metadata
            for result in results:
                result["retrieval_method"] = "vector_similarity"
                result["retrieval_score"] = result.get("similarity", 0.0)

            return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _graph_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform graph-based search using entity relationships."""
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query)

            if not query_entities:
                return []

            graph_results = []

            # For each query entity, find related chunks through graph traversal
            for entity in query_entities:
                entity_name = entity.get("name", "")

                # Get related entities through graph traversal
                related_entities = self.db_interface.get_entity_relationships(
                    entity_name,
                    max_depth=self.graph_depth
                )

                # Find chunks that mention related entities
                for related in related_entities:
                    related_entity_name = related.get("related_entity", "")

                    # Search for chunks containing this entity
                    chunks = self._find_chunks_with_entity(related_entity_name, top_k=3)

                    for chunk in chunks:
                        chunk["retrieval_method"] = "graph_traversal"
                        chunk["retrieval_score"] = related.get("weight", 1.0) * 0.5  # Scale down graph scores
                        chunk["traversal_path"] = [
                            entity_name,
                            related.get("relationship", ""),
                            related_entity_name
                        ]
                        graph_results.append(chunk)

            # Deduplicate results based on chunk_id
            seen_chunks = set()
            unique_results = []

            for result in graph_results:
                chunk_id = result.get("chunk_id", "")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)

            return unique_results[:top_k]

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return []

    def _combine_results(self, query: str, query_embedding: np.ndarray,
                        vector_results: List[Dict[str, Any]],
                        graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine vector and graph search results."""

        # Create a map of chunk_id to result for easy lookup
        combined_map = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                combined_map[chunk_id] = {
                    "chunk": result,
                    "vector_score": result.get("retrieval_score", 0.0),
                    "graph_score": 0.0,
                    "methods": ["vector"]
                }

        # Add/merge graph results
        for result in graph_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                if chunk_id in combined_map:
                    # Merge with existing result
                    existing = combined_map[chunk_id]
                    existing["graph_score"] = result.get("retrieval_score", 0.0)
                    existing["methods"].append("graph")
                    if "traversal_path" in result:
                        existing["traversal_path"] = result["traversal_path"]
                else:
                    # Add new result
                    combined_map[chunk_id] = {
                        "chunk": result,
                        "vector_score": 0.0,
                        "graph_score": result.get("retrieval_score", 0.0),
                        "methods": ["graph"],
                        "traversal_path": result.get("traversal_path", [])
                    }

        # Calculate hybrid scores
        combined_results = []
        for item in combined_map.values():
            chunk = item["chunk"]
            vector_score = item["vector_score"]
            graph_score = item["graph_score"]

            # Calculate hybrid score
            hybrid_score = (self.hybrid_alpha * vector_score +
                          (1 - self.hybrid_alpha) * graph_score)

            chunk["hybrid_score"] = hybrid_score
            chunk["retrieval_methods"] = item["methods"]
            chunk["vector_score"] = vector_score
            chunk["graph_score"] = graph_score

            if "traversal_path" in item:
                chunk["traversal_path"] = item["traversal_path"]

            combined_results.append(chunk)

        return combined_results

    def _rerank_results(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank results using hybrid scoring."""

        # Sort by hybrid score (descending)
        results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

        # Add final ranking
        for i, result in enumerate(results[:top_k]):
            result["final_rank"] = i + 1

        return results[:top_k]

    def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query text."""
        # This is a simplified entity extraction for queries
        # In a full implementation, you might use the same entity extractor

        # For now, we'll do simple pattern matching for common entity types
        entities = []

        # Simple patterns for entity extraction
        patterns = {
            "ORGANIZATION": r'\b[A-Z][a-zA-Z0-9\s]*(Inc|Corp|LLC|Ltd|Company|Corp|Inc)\b',
            "TECHNOLOGY": r'\b(Python|Java|JavaScript|C\+\+|AI|ML|Machine Learning|Deep Learning)\b',
            "CONCEPT": r'\b(algorithm|model|network|system|framework|platform)\b'
        }

        for entity_type, pattern in patterns.items():
            import re
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "name": match.strip(),
                    "type": entity_type,
                    "confidence": 0.8  # Default confidence for pattern matching
                })

        return entities

    def _find_chunks_with_entity(self, entity_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find chunks that contain a specific entity."""
        # This is a simplified implementation
        # In practice, you might want to search the database for chunks containing the entity

        # For now, return empty list - this would need to be implemented
        # based on your specific entity storage strategy
        return []

    def _find_entities_in_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find entities mentioned in a chunk."""
        # Simplified implementation
        return []

    def _find_relationships_in_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relationships mentioned in a chunk."""
        # Simplified implementation
        return []

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""
        return {
            "vector_top_k": self.vector_top_k,
            "graph_depth": self.graph_depth,
            "hybrid_alpha": self.hybrid_alpha,
            "rerank_top_k": self.rerank_top_k,
            "embedding_dimension": self.embedding_generator.get_embedding_dimension()
        }


def main():
    """Example usage of the HybridRetriever."""
    logging.basicConfig(level=logging.INFO)

    try:
        retriever = HybridRetriever()

        # Example query
        query = "What is machine learning and how does it relate to artificial intelligence?"

        # Perform retrieval
        results = retriever.retrieve(query, top_k=5)

        print(f"Query: {query}")
        print(f"Retrieved {len(results)} results:")

        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.get('retrieval_methods', ['unknown'])}]")
            print(f"   Score: {result.get('hybrid_score', 0):.4f}")
            print(f"   Text: {result.get('text', '')[:200]}...")
            print(f"   Source: {result.get('source', 'unknown')}")

        # Get retrieval with entities
        detailed_results = retriever.retrieve_with_entities(query, top_k=3)
        print(f"\nDetailed results include {len(detailed_results.get('entities', []))} entities")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
