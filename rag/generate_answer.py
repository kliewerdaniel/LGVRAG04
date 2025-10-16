"""
Answer generation module using local LLMs.

This module handles generating contextual answers to queries using
retrieved information and local language models via Ollama.
"""

import os
import json
import logging
import asyncio
import aiohttp
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with metadata."""
    query: str
    answer: str
    context_chunks: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    model_used: str
    tokens_used: Optional[int] = None
    generation_time: Optional[float] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert answer to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "context_chunks": self.context_chunks,
            "entities": self.entities,
            "relationships": self.relationships,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "generation_time": self.generation_time,
            "confidence_score": self.confidence_score
        }


class AnswerGenerator:
    """
    Generate contextual answers using local LLMs and retrieved information.

    This class handles the process of taking retrieved context and generating
    coherent, accurate answers using local language models via Ollama.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the answer generator.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.base_url = self.config.llm.base_url.rstrip('/')
        self.model_name = self.config.llm.model_name
        self.temperature = self.config.llm.temperature
        self.max_tokens = self.config.llm.max_tokens
        self.timeout = self.config.llm.timeout

        # Health check and validation
        self._validate_ollama_connection()

    def _validate_ollama_connection(self) -> None:
        """Validate Ollama server connectivity and model availability."""
        try:
            # Check if Ollama server is running
            asyncio.run(self._check_ollama_health())

            # Check if the specified model is available
            asyncio.run(self._check_model_availability())

            logger.info(f"Ollama connection validated for model: {self.model_name}")

        except Exception as e:
            logger.error(f"Ollama validation failed: {e}")
            raise RuntimeError(f"Ollama server not available or model '{self.model_name}' not found. Please ensure Ollama is running and the model is installed.")

    async def _check_ollama_health(self) -> None:
        """Check if Ollama server is running."""
        url = f"{self.base_url}/api/tags"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Ollama server not responding (status: {response.status})")

        except asyncio.TimeoutError:
            raise RuntimeError("Ollama server timeout - server may not be running")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Cannot connect to Ollama server: {e}")

    async def _check_model_availability(self) -> None:
        """Check if the specified model is available."""
        url = f"{self.base_url}/api/tags"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])

                        # Check if our model is in the list
                        model_names = [model.get("name", "") for model in models]
                        model_found = any(self.model_name in name for name in model_names)

                        if not model_found:
                            available_models = ", ".join(model_names)
                            raise RuntimeError(f"Model '{self.model_name}' not found. Available models: {available_models}")

        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
            # Don't raise here as the model check might fail but generation could still work

    async def generate_answer(self,
                            query: str,
                            context_chunks: List[Dict[str, Any]],
                            entities: Optional[List[Dict[str, Any]]] = None,
                            relationships: Optional[List[Dict[str, Any]]] = None) -> GeneratedAnswer:
        """
        Generate an answer for a query using retrieved context.

        Args:
            query: The user's query
            context_chunks: Retrieved context chunks
            entities: Optional entity information
            relationships: Optional relationship information

        Returns:
            GeneratedAnswer object with the answer and metadata
        """
        import time
        start_time = time.time()

        try:
            # Prepare context
            context = self._prepare_context(query, context_chunks, entities, relationships)

            # Create prompt
            prompt = self._create_prompt(query, context)

            # Generate answer
            answer_data = await self._call_ollama(prompt)

            if not answer_data:
                return GeneratedAnswer(
                    query=query,
                    answer="Sorry, I couldn't generate an answer at this time.",
                    context_chunks=context_chunks,
                    entities=entities or [],
                    relationships=relationships or [],
                    model_used=self.model_name
                )

            # Extract answer
            answer_text = self._extract_answer(answer_data)

            # Calculate generation time
            generation_time = time.time() - start_time

            # Create result
            result = GeneratedAnswer(
                query=query,
                answer=answer_text,
                context_chunks=context_chunks,
                entities=entities or [],
                relationships=relationships or [],
                model_used=self.model_name,
                generation_time=generation_time,
                tokens_used=answer_data.get("tokens_used")
            )

            logger.info(f"Generated answer in {generation_time:.2f}s using {self.model_name}")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            generation_time = time.time() - start_time

            return GeneratedAnswer(
                query=query,
                answer=f"I encountered an error while generating the answer: {str(e)}",
                context_chunks=context_chunks,
                entities=entities or [],
                relationships=relationships or [],
                model_used=self.model_name,
                generation_time=generation_time
            )

    def _prepare_context(self, query: str, chunks: List[Dict[str, Any]],
                        entities: Optional[List[Dict[str, Any]]] = None,
                        relationships: Optional[List[Dict[str, Any]]] = None) -> str:
        """Prepare context from retrieved information."""

        context_parts = []

        # Add text chunks as primary context
        if chunks:
            context_parts.append("Relevant Information:")
            for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
                text = chunk.get("text", "")
                source = chunk.get("source", "Unknown")
                score = chunk.get("hybrid_score", 0)

                context_parts.append(f"[{i+1}] From {source} (relevance: {score:.3f}):")
                context_parts.append(text)
                context_parts.append("")

        # Add entity information
        if entities:
            context_parts.append("Key Entities:")
            for entity in entities[:10]:  # Limit to top 10 entities
                name = entity.get("name", "")
                entity_type = entity.get("type", "")
                confidence = entity.get("confidence", 0)

                context_parts.append(f"- {name} ({entity_type}, confidence: {confidence:.2f})")
            context_parts.append("")

        # Add relationship information
        if relationships:
            context_parts.append("Key Relationships:")
            for rel in relationships[:10]:  # Limit to top 10 relationships
                subject = rel.get("subject", "")
                predicate = rel.get("predicate", "")
                obj = rel.get("object", "")
                confidence = rel.get("confidence", 0)

                context_parts.append(f"- {subject} {predicate} {obj} (confidence: {confidence:.2f})")
            context_parts.append("")

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for answer generation."""

        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- Be specific and cite sources when possible
- If the context doesn't contain enough information, say so clearly
- Structure your answer clearly and logically
- Keep the answer concise but comprehensive

Answer:"""

        return prompt

    async def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Ollama API for answer generation."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "response": result.get("response", ""),
                            "tokens_used": result.get("tokens_used", 0),
                            "model": result.get("model", self.model_name)
                        }
                    else:
                        logger.error(f"Ollama API error: {response.status} - {await response.text()}")
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Ollama API for model {self.model_name}")
            return None
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None

    def _extract_answer(self, answer_data: Dict[str, Any]) -> str:
        """Extract and clean the generated answer."""
        if not answer_data or "response" not in answer_data:
            return "No answer generated."

        answer = answer_data["response"].strip()

        # Clean up common issues
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()

        if answer.startswith("Based on the context"):
            # Remove redundant phrases
            lines = answer.split('\n')
            if len(lines) > 1:
                answer = '\n'.join(lines[1:])

        return answer.strip()

    def generate_answer_sync(self,
                           query: str,
                           context_chunks: List[Dict[str, Any]],
                           entities: Optional[List[Dict[str, Any]]] = None,
                           relationships: Optional[List[Dict[str, Any]]] = None) -> GeneratedAnswer:
        """Synchronous version of answer generation."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async_in_thread, query, context_chunks, entities, relationships)
                return future.result()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(
                self.generate_answer(query, context_chunks, entities, relationships)
            )

    def _run_async_in_thread(self, query: str, context_chunks: List[Dict[str, Any]],
                           entities: Optional[List[Dict[str, Any]]] = None,
                           relationships: Optional[List[Dict[str, Any]]] = None) -> GeneratedAnswer:
        """Run async function in a new event loop (for use in threads)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_answer(query, context_chunks, entities, relationships)
            )
        finally:
            loop.close()

    def generate_multiple_answers(self,
                                query: str,
                                context_chunks: List[Dict[str, Any]],
                                num_answers: int = 3) -> List[GeneratedAnswer]:
        """
        Generate multiple diverse answers for the same query.

        Args:
            query: The user's query
            context_chunks: Retrieved context chunks
            num_answers: Number of different answers to generate

        Returns:
            List of GeneratedAnswer objects
        """
        answers = []

        # Vary temperature for diversity
        temperatures = [0.1, 0.3, 0.5, 0.7][:num_answers]

        for temp in temperatures:
            # Create a modified config for this generation
            original_temp = self.temperature
            self.temperature = temp

            try:
                answer = self.generate_answer_sync(query, context_chunks)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating answer with temperature {temp}: {e}")
            finally:
                # Restore original temperature
                self.temperature = original_temp

        return answers

    def evaluate_answer_quality(self, answer: GeneratedAnswer) -> Dict[str, float]:
        """
        Evaluate the quality of a generated answer.

        Args:
            answer: GeneratedAnswer to evaluate

        Returns:
            Dictionary with quality metrics
        """
        metrics = {}

        # Length metric (answers should be substantial but not too long)
        answer_length = len(answer.answer.split())
        metrics["length_score"] = min(1.0, answer_length / 100.0)  # Optimal around 100 words

        # Context utilization (how much context was used)
        if answer.context_chunks:
            # Simple heuristic: longer answers tend to use more context
            context_usage = min(1.0, len(answer.answer) / 1000.0)
            metrics["context_usage"] = context_usage
        else:
            metrics["context_usage"] = 0.0

        # Entity utilization
        if answer.entities:
            entity_mentions = len(answer.entities)
            metrics["entity_utilization"] = min(1.0, entity_mentions / 5.0)  # Optimal around 5 entities
        else:
            metrics["entity_utilization"] = 0.0

        # Overall confidence (combine metrics)
        weights = {"length_score": 0.3, "context_usage": 0.4, "entity_utilization": 0.3}
        overall_score = sum(metrics[key] * weights[key] for key in weights)

        metrics["overall_quality"] = overall_score

        return metrics


def main():
    """Example usage of the AnswerGenerator."""
    logging.basicConfig(level=logging.INFO)

    try:
        generator = AnswerGenerator()

        # Example context
        context_chunks = [
            {
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                "source": "ML_overview.md",
                "hybrid_score": 0.85
            },
            {
                "text": "Neural networks are computational models inspired by biological neural networks, consisting of layers of interconnected nodes.",
                "source": "neural_networks.md",
                "hybrid_score": 0.72
            }
        ]

        query = "What is machine learning and how does it relate to artificial intelligence?"

        # Generate answer
        answer = generator.generate_answer_sync(query, context_chunks)

        print(f"Query: {query}")
        print(f"Answer: {answer.answer}")
        print(f"Generation time: {answer.generation_time:.2f}s")
        print(f"Model used: {answer.model_used}")

        # Evaluate quality
        quality = generator.evaluate_answer_quality(answer)
        print(f"Answer quality: {quality}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
