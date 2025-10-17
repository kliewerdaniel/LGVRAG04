#!/usr/bin/env python3
"""
Test script for entity and relationship extraction with improved JSON parsing.
"""

import os
import sys
import logging
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.extract_relations import EntityExtractor

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_entity_extraction():
    """Test the improved entity extraction with better JSON parsing."""
    logger.info("Testing improved entity extraction...")

    # Create extractor
    extractor = EntityExtractor()

    # Test with sample text that should produce entities and relationships
    test_chunks = [
        {
            "text": "John Smith works at OpenAI in San Francisco and is developing GPT-4, a large language model created by Microsoft. The project started in 2022 and uses Python programming language.",
            "chunk_id": "test_chunk_1"
        },
        {
            "text": "Apple Inc. released the iPhone 14 in September 2022. The device features a new A16 chip and improved camera technology developed by Sony.",
            "chunk_id": "test_chunk_2"
        }
    ]

    try:
        logger.info("Testing entity extraction...")
        entities = extractor.extract_entities_sync(test_chunks)

        print(f"\nExtracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.name} ({entity.type}): {entity.confidence:.2f}")

        logger.info("Testing relationship extraction...")
        relationships = extractor.extract_relationships_sync(test_chunks)

        print(f"\nExtracted {len(relationships)} relationships:")
        for rel in relationships:
            print(f"  - {rel.subject} --{rel.predicate}--> {rel.object} ({rel.confidence:.2f})")

        # Test with a problematic text that might cause JSON parsing issues
        logger.info("Testing with potentially problematic text...")
        problematic_chunks = [
            {
                "text": "This is a simple test without complex entities.",
                "chunk_id": "problematic_chunk_1"
            }
        ]

        entities_simple = extractor.extract_entities_sync(problematic_chunks)
        relationships_simple = extractor.extract_relationships_sync(problematic_chunks)

        print(f"\nSimple text - Entities: {len(entities_simple)}, Relationships: {len(relationships_simple)}")

        return True

    except Exception as e:
        logger.error(f"Error during entity extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_parsing_edge_cases():
    """Test JSON parsing with various edge cases."""
    logger.info("Testing JSON parsing edge cases...")

    extractor = EntityExtractor()

    # Test cases that might cause JSON parsing issues
    edge_cases = [
        # Case 1: Response with markdown code blocks
        '```json\n{"entities": [{"name": "test", "type": "PERSON", "confidence": 0.9}]}\n```',

        # Case 2: Response with extra text before JSON
        'Here are the entities I found: {"entities": [{"name": "test", "type": "PERSON", "confidence": 0.9}]}',

        # Case 3: Response with extra text after JSON
        '{"entities": [{"name": "test", "type": "PERSON", "confidence": 0.9}]} Thank you!',

        # Case 4: Malformed JSON that should be handled gracefully
        '{"entities": [{"name": "test", "type": "PERSON", "confidence": 0.9}',

        # Case 5: Valid JSON
        '{"entities": [{"name": "test", "type": "PERSON", "confidence": 0.9}]}'
    ]

    for i, response_text in enumerate(edge_cases):
        logger.info(f"Testing edge case {i+1}: {response_text[:50]}...")
        try:
            cleaned = extractor._clean_json_response(response_text)
            logger.info(f"Cleaned response: {cleaned}")

            # Try to parse as JSON
            parsed = json.loads(cleaned)
            logger.info(f"Successfully parsed: {parsed}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse edge case {i+1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in edge case {i+1}: {e}")

if __name__ == "__main__":
    logger.info("Starting entity extraction tests...")

    # Test basic functionality
    success1 = test_entity_extraction()

    # Test JSON parsing edge cases
    print("\n" + "="*50)
    test_json_parsing_edge_cases()

    if success1:
        logger.info("All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)
