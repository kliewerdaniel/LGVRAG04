"""
Entity and relationship extraction module using local LLMs.

This module provides functionality to extract entities and their relationships
from text chunks using local language models via Ollama.
"""

import os
import json
import logging
import asyncio
import aiohttp
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    confidence: float
    chunk_id: str
    source_text: str
    position: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "confidence": self.confidence,
            "chunk_id": self.chunk_id,
            "source_text": self.source_text,
            "position": self.position
        }


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    subject: str
    predicate: str
    object: str
    confidence: float
    chunk_id: str
    source_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "chunk_id": self.chunk_id,
            "source_text": self.source_text
        }


class EntityExtractor:
    """
    Extract entities and relationships from text using local LLMs.

    This class uses Ollama to perform entity recognition and relationship
    extraction from text chunks in a local environment.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the entity extractor.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.base_url = self.config.llm.base_url.rstrip('/')
        self.entity_model = self.config.llm.entity_model
        self.timeout = self.config.llm.timeout

        # Entity type patterns for validation
        self.entity_types = {
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
            "MONEY", "PERCENT", "PRODUCT", "EVENT", "CONCEPT",
            "TECHNOLOGY", "LANGUAGE", "FRAMEWORK", "TOOL"
        }

        self.relationship_types = set(self.config.entities.relationship_types)

    async def extract_entities(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """
        Extract entities from text chunks.

        Args:
            chunks: List of text chunks with metadata

        Returns:
            List of extracted entities
        """
        entities = []

        # Process chunks in batches for better performance
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Process batch concurrently
            tasks = [self._extract_entities_from_chunk(chunk) for chunk in batch]
            batch_entities = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_entities:
                if isinstance(result, Exception):
                    logger.error(f"Error extracting entities: {result}")
                    continue

                entities.extend(result)

        logger.info(f"Extracted {len(entities)} entities from {len(chunks)} chunks")
        return entities

    async def extract_relationships(self, chunks: List[Dict[str, Any]]) -> List[Relationship]:
        """
        Extract relationships from text chunks.

        Args:
            chunks: List of text chunks with metadata

        Returns:
            List of extracted relationships
        """
        relationships = []

        # Process chunks in batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            tasks = [self._extract_relationships_from_chunk(chunk) for chunk in batch]
            batch_relationships = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_relationships:
                if isinstance(result, Exception):
                    logger.error(f"Error extracting relationships: {result}")
                    continue

                relationships.extend(result)

        logger.info(f"Extracted {len(relationships)} relationships from {len(chunks)} chunks")
        return relationships

    async def _extract_entities_from_chunk(self, chunk: Dict[str, Any]) -> List[Entity]:
        """Extract entities from a single chunk."""
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "unknown")

        if not text.strip():
            return []

        try:
            # Create entity extraction prompt
            prompt = self._create_entity_prompt(text)

            # Call Ollama API
            entities_data = await self._call_ollama(prompt, self.entity_model)

            if not entities_data:
                return []

            # Parse response
            entities = self._parse_entity_response(entities_data, chunk_id, text)

            # Filter by confidence threshold
            entities = [e for e in entities if e.confidence >= self.config.entities.min_confidence]

            # Limit number of entities per chunk
            max_entities = self.config.entities.max_entities_per_chunk
            if len(entities) > max_entities:
                # Sort by confidence and keep top entities
                entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:max_entities]

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities from chunk {chunk_id}: {e}")
            return []

    async def _extract_relationships_from_chunk(self, chunk: Dict[str, Any]) -> List[Relationship]:
        """Extract relationships from a single chunk."""
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "unknown")

        if not text.strip():
            return []

        try:
            # Create relationship extraction prompt
            prompt = self._create_relationship_prompt(text)

            # Call Ollama API
            relationships_data = await self._call_ollama(prompt, self.entity_model)

            if not relationships_data:
                return []

            # Parse response
            relationships = self._parse_relationship_response(relationships_data, chunk_id, text)

            # Filter by confidence threshold
            relationships = [r for r in relationships if r.confidence >= self.config.entities.min_confidence]

            return relationships

        except Exception as e:
            logger.error(f"Error extracting relationships from chunk {chunk_id}: {e}")
            return []

    def _create_entity_prompt(self, text: str) -> str:
        """Create prompt for entity extraction."""
        return f"""Extract all named entities from the following text. For each entity, provide:
- name: the exact entity text
- type: one of the specified entity types
- confidence: a number between 0.0 and 1.0 indicating your confidence

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, PRODUCT, EVENT, CONCEPT, TECHNOLOGY, LANGUAGE, FRAMEWORK, TOOL

Return ONLY a JSON object with this exact format:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "ENTITY_TYPE",
      "confidence": 0.95
    }}
  ]
}}

Text: {text}

IMPORTANT: Return only the JSON object, no other text or explanation."""

    def _create_relationship_prompt(self, text: str) -> str:
        """Create prompt for relationship extraction."""
        return f"""Extract relationships between entities in the following text. For each relationship, provide:
- subject: the entity that is performing the action or is the source
- predicate: the relationship type (one of: {', '.join(self.relationship_types)})
- object: the entity that is being acted upon or is the target
- confidence: a number between 0.0 and 1.0 indicating your confidence

Return ONLY a JSON object with this exact format:
{{
  "relationships": [
    {{
      "subject": "entity name",
      "predicate": "relationship_type",
      "object": "entity name",
      "confidence": 0.95
    }}
  ]
}}

Text: {text}

IMPORTANT: Return only the JSON object, no other text or explanation."""

    def _clean_json_response(self, response_text: str) -> str:
        """Clean and prepare JSON response text for parsing."""
        if not response_text:
            return ""

        # Strip whitespace
        cleaned = response_text.strip()

        # Remove any markdown code block markers
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        # Remove any leading/trailing text that isn't JSON
        # Find the first { and last }
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return cleaned

        return cleaned[start_idx:end_idx + 1]

    async def _call_ollama(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """Call Ollama API for text generation."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config.llm.temperature,
                "num_predict": self.config.llm.max_tokens
            }
        }

        try:
            logger.debug(f"Calling Ollama API with model {model} for prompt length: {len(prompt)}")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "")
                        logger.debug(f"Received response from Ollama: {response_text[:200]}...")

                        # Try to parse JSON response
                        try:
                            # Clean the response text first
                            cleaned_response = self._clean_json_response(response_text)
                            logger.debug(f"Cleaned response: {cleaned_response[:200]}...")
                            return json.loads(cleaned_response)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON response: {response_text}")
                            logger.warning(f"JSON decode error: {e}")
                            logger.warning("Attempting to extract JSON from response...")
                            # Try to extract JSON if it's embedded in other text
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    logger.debug(f"Attempting to parse extracted JSON: {json_match.group()[:200]}...")
                                    return json.loads(json_match.group())
                                except json.JSONDecodeError as je:
                                    logger.error(f"Failed to extract JSON even with regex: {je}")
                                    logger.error(f"Extracted text was: {json_match.group()}")
                            else:
                                logger.error("No JSON-like pattern found in response")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Ollama API for model {model} after {self.timeout}s")
            logger.info("Consider using a smaller/faster model or reducing max_tokens for entity extraction")
            return None
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None

    def _parse_entity_response(self, response: Dict[str, Any], chunk_id: str, source_text: str) -> List[Entity]:
        """Parse entity extraction response."""
        entities = []

        try:
            entities_list = response if isinstance(response, list) else response.get("entities", [])

            for entity_data in entities_list:
                if not isinstance(entity_data, dict):
                    continue

                name = entity_data.get("name", "").strip()
                entity_type = entity_data.get("type", "").strip().upper()
                confidence = float(entity_data.get("confidence", 0.0))

                if name and entity_type in self.entity_types and confidence > 0:
                    entity = Entity(
                        name=name,
                        type=entity_type,
                        confidence=confidence,
                        chunk_id=chunk_id,
                        source_text=source_text
                    )
                    entities.append(entity)

        except Exception as e:
            logger.error(f"Error parsing entity response: {e}")

        return entities

    def _parse_relationship_response(self, response: Dict[str, Any], chunk_id: str, source_text: str) -> List[Relationship]:
        """Parse relationship extraction response."""
        relationships = []

        try:
            relationships_list = response if isinstance(response, list) else response.get("relationships", [])

            for rel_data in relationships_list:
                if not isinstance(rel_data, dict):
                    continue

                # Safely extract and convert to strings
                try:
                    subject_raw = rel_data.get("subject", "")
                    predicate_raw = rel_data.get("predicate", "")
                    obj_raw = rel_data.get("object", "")

                    # Handle different types of values
                    if subject_raw is None:
                        subject = ""
                    elif isinstance(subject_raw, list):
                        subject = str(subject_raw[0]) if subject_raw else ""
                    else:
                        subject = str(subject_raw).strip()

                    if predicate_raw is None:
                        predicate = ""
                    elif isinstance(predicate_raw, list):
                        predicate = str(predicate_raw[0]) if predicate_raw else ""
                    else:
                        predicate = str(predicate_raw).strip()

                    if obj_raw is None:
                        obj = ""
                    elif isinstance(obj_raw, list):
                        obj = str(obj_raw[0]) if obj_raw else ""
                    else:
                        obj = str(obj_raw).strip()

                    confidence = float(rel_data.get("confidence", 0.0))

                    if subject and predicate and obj and confidence > 0:
                        relationship = Relationship(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            confidence=confidence,
                            chunk_id=chunk_id,
                            source_text=source_text
                        )
                        relationships.append(relationship)

                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Skipping invalid relationship data: {rel_data} - Error: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing relationship response: {e}")

        return relationships

    def extract_entities_sync(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """Synchronous version of entity extraction."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, use thread-based execution
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async_entities_in_thread, chunks)
                return future.result()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.extract_entities(chunks))

    def extract_relationships_sync(self, chunks: List[Dict[str, Any]]) -> List[Relationship]:
        """Synchronous version of relationship extraction."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, use thread-based execution
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async_relationships_in_thread, chunks)
                return future.result()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.extract_relationships(chunks))

    def _run_async_entities_in_thread(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """Run async entity extraction in a new event loop (for use in threads)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.extract_entities(chunks))
        finally:
            loop.close()

    def _run_async_relationships_in_thread(self, chunks: List[Dict[str, Any]]) -> List[Relationship]:
        """Run async relationship extraction in a new event loop (for use in threads)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.extract_relationships(chunks))
        finally:
            loop.close()


def main():
    """Example usage of the EntityExtractor."""
    logging.basicConfig(level=logging.INFO)

    extractor = EntityExtractor()

    # Example text for testing
    sample_chunks = [
        {
            "text": "John Smith works at OpenAI and is developing GPT-4. He lives in San Francisco.",
            "chunk_id": "sample_chunk_1"
        }
    ]

    try:
        # Extract entities
        entities = extractor.extract_entities_sync(sample_chunks)
        print(f"Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.name} ({entity.type}): {entity.confidence:.2f}")

        # Extract relationships
        relationships = extractor.extract_relationships_sync(sample_chunks)
        print(f"\nExtracted {len(relationships)} relationships:")
        for rel in relationships:
            print(f"  - {rel.subject} --{rel.predicate}--> {rel.object} ({rel.confidence:.2f})")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
