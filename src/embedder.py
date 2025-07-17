import os
import cv2
import time
import uuid # Added for generating unique IDs
import json
import requests
from PIL import Image
import base64
import io
from pinecone import Pinecone, PineconeException # Modified import
import logging # Added import
from .config import (
    PINECONE_API_KEY,
    PINECONE_EMBEDDINGS_MODEL,
    PINECONE_INDEX_NAME
)

class Embedder:
    def __init__(self, ):
        self.pc = Pinecone(PINECONE_API_KEY)
        self.model = PINECONE_EMBEDDINGS_MODEL
        self.namespace = PINECONE_INDEX_NAME
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def upload_embeddings(self, filepath):
        
        try:
            
            if not os.path.exists(filepath):
                self.logger.error(f"File {filepath} not found.")
                raise FileNotFoundError(f"File {filepath} not found.")
            
            with open(filepath, 'r',encoding='utf-8') as file:
                text_content = file.read().strip()
                if not text_content:
                    self.logger.warning(f"File {filepath} is empty.")
                    raise ValueError("File is empty.")
            file_id = os.path.splitext(os.path.basename(filepath))[0]
            
            # Prepare the prompt for the VLM
            data = [{"id": file_id, "text": text_content}]
            self.logger.info(f"Generating embeddings for {filepath}...")
            embeddings = self.pc.inference.embed(
                model=self.model,
                inputs = [d['text'] for d in data],
                parameters= {
                "input_type":"passage"
                }
            )
        
            vectors = []
        
            for d, e in zip(data, embeddings):
                vectors.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text']}
                })
            
            index = self.pc.Index(self.namespace)
            self.logger.info(f"Upserting embeddings for {filepath} to Pinecone index '{self.namespace}'...")
            index.upsert(
                vectors=vectors,
                namespace=self.namespace,
            )
        
            self.logger.info(f"Successfully uploaded embeddings for {filepath} to Pinecone.")
            
        except FileNotFoundError as e: # More specific exception
            # Already logged, re-raise or handle as needed
            raise e
        except ValueError as e: # More specific exception
            # Already logged, re-raise or handle as needed
            raise e
        except PineconeException as e: # Specific Pinecone exception
            self.logger.error(f"Pinecone API error during embedding upload for {filepath}. Error: {e}")
            # Potentially re-raise or handle specific Pinecone error codes
            raise e
        except requests.exceptions.RequestException as e: # For network errors during API calls
            self.logger.error(f"Network error during embedding upload for {filepath}. Error: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while uploading embeddings for {filepath}. Error: {e}")
            raise e # Re-raise the exception after logging







