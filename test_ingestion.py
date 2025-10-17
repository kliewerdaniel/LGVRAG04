#!/usr/bin/env python3
"""
Test script to verify ingestion works with timeout handling.
"""

import os
import sys
import logging
import signal
from contextlib import contextmanager

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.ingest_data import DataIngester

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    logger.error("Ingestion timed out after 10 minutes")
    sys.exit(1)

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def test_ingestion():
    """Test the ingestion process with timeout."""
    logger.info("Starting ingestion test...")

    try:
        # Set a 10-minute timeout for the entire ingestion process
        with timeout(600):  # 10 minutes
            ingester = DataIngester()
            logger.info("Created DataIngester")

            # Test with a small subset first - just one file
            documents_dir = "./documents"
            if os.path.exists(documents_dir):
                files = []
                for root, dirs, filenames in os.walk(documents_dir):
                    for filename in filenames:
                        if filename.endswith('.md'):
                            files.append(os.path.join(root, filename))
                            if len(files) >= 3:  # Just test first 3 files
                                break
                    if len(files) >= 3:
                        break

                logger.info(f"Testing with {len(files)} files: {files}")

                if files:
                    # Test single file ingestion first
                    stats = ingester.ingest_file(files[0])
                    logger.info(f"Single file ingestion completed: {stats}")
                else:
                    logger.warning("No markdown files found for testing")
            else:
                logger.warning(f"Documents directory not found: {documents_dir}")

    except TimeoutError as e:
        logger.error(f"Ingestion timed out: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return False

    logger.info("Ingestion test completed successfully")
    return True

if __name__ == "__main__":
    success = test_ingestion()
    sys.exit(0 if success else 1)
