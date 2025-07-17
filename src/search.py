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
    PINECONE_INDEX_NAME,
    OUTPUT_SNAPSHOTS_PATH # Added import
)

class Searcher:
    def __init__(self):
        self.pc = Pinecone(PINECONE_API_KEY)
        self.model = PINECONE_EMBEDDINGS_MODEL
        self.namespace = PINECONE_INDEX_NAME
        self.output_snapshots_path = OUTPUT_SNAPSHOTS_PATH # Store path
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def _find_image_path_for_id(self, result_id):
        """Find the image path corresponding to a given result ID."""
        possible_extensions = [".jpg", ".png", ".jpeg"]
        image_path = None

        if result_id.startswith("scene_frame"):
            screenshots_main_dir = os.path.join(self.output_snapshots_path, "screenshots")
            self.logger.debug(f"Searching for scene_frame ID '{result_id}' in main screenshots dir: {screenshots_main_dir}")
            if os.path.exists(screenshots_main_dir):
                for video_specific_subdir_name in os.listdir(screenshots_main_dir):
                    video_specific_subdir_path = os.path.join(screenshots_main_dir, video_specific_subdir_name)
                    self.logger.debug(f"Checking subdirectory: {video_specific_subdir_path}")
                    if os.path.isdir(video_specific_subdir_path):
                        for ext in possible_extensions:
                            potential_path = os.path.join(video_specific_subdir_path, result_id + ext)
                            self.logger.debug(f"Attempting to find image at: {potential_path}")
                            if os.path.exists(potential_path):
                                image_path = potential_path
                                self.logger.info(f"Found image for ID {result_id} at {image_path}") # Changed from debug to info for successful find
                                break  # Found image with this extension
                    else:
                        self.logger.debug(f"Skipping '{video_specific_subdir_path}' as it is not a directory.")
                    if image_path:  # If found in this video_specific_subdir, no need to check others
                        break
            else:
                self.logger.warning(f"Main screenshots directory not found: {screenshots_main_dir}")
        else:
            # Likely an object snapshot, e.g., "bicycle_id1_1747127929"
            parts = result_id.split('_id', 1)
            if len(parts) > 0:
                class_name = parts[0]
                for ext in possible_extensions:
                    path = os.path.join(self.output_snapshots_path, class_name, result_id + ext)
                    if os.path.exists(path):
                        image_path = path
                        break
            else:
                # Fallback if no '_id' is found, though less likely for object snapshots
                # This part might need adjustment based on exact ID formats if they vary more
                self.logger.warning(f"Could not determine class from ID: {result_id}")

        if not image_path and result_id.startswith("scene_frame"): # Log only if not found and was a scene_frame
            self.logger.warning(f"Could not find image for scene_frame ID {result_id} after checking all subdirectories.")
        elif not image_path: # General log if not found (for object snapshots)
             self.logger.warning(f"Could not find image for ID {result_id}")
        # Removed the original combined log for not found, handled above
        return image_path

    def search_embeddings(self, query_text):
        try:
            self.logger.info(f"Searching for embeddings related to: {query_text}")
            # Use self.model and correctly access the embedding vector
            embedding_response = self.pc.inference.embed(
                model=self.model, # Use model from config
                inputs=[query_text],
                parameters={
                    "input_type": "query"
                }
            )
            
            if not embedding_response.data:
                self.logger.error("Failed to generate embedding for the query.")
                return []
            
            query_vector = embedding_response.data[0].values # Correct way to get the vector
            
            index = self.pc.Index(self.namespace)
            
            response = index.query(
                namespace=self.namespace,
                vector=query_vector, # Use the correctly extracted vector
                top_k=5,
                include_values=True,
                include_metadata=True,
            )
            
            if response and 'matches' in response:
                results = response['matches']
                self.logger.info(f"Found {len(results)} matches.")
                # Augment results with image paths
                augmented_results = []
                for match in results:
                    result_id = match.get('id') # Pinecone match ID is the file_id
                    if result_id:
                        image_path = self._find_image_path_for_id(result_id)
                        match['image_path'] = image_path
                    else:
                        match['image_path'] = None
                    augmented_results.append(match)
                return augmented_results
            else:
                self.logger.warning("No matches found.")
                return []
        
        except PineconeException as e:
            self.logger.error(f"Pinecone error: {e}")
            return []
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return []


