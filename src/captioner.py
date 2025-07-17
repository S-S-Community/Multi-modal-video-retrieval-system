import os
import cv2
import time
import uuid # Added for generating unique IDs
import json
import requests
from PIL import Image
import base64
import io
import sys
sys.path.append('..')  # Add parent directory to path
from checkcheck import FastImageCaptioner
from .config import (
    RECOGNIZED_CLASSES as TARGET_CLASSES_FOR_SNAPSHOT,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_PROMPT,
    FAST_CAPTIONER_MODEL,
)
from .embedder import Embedder

class Captioner:
    def __init__(self, vlmmodel, input_frame_path, output_dir):
        self.prompt = OLLAMA_PROMPT
        self.model = vlmmodel
        self.input_video_path = input_frame_path
        self.output_dir = output_dir # This is OUTPUT_SNAPSHOTS_PATH
        self.snapshotted_object_ids_this_video = set()
        self.current_video_screenshot_dir = "" # Will be set in analyze_video
        
        # Initialize FastImageCaptioner instead of Ollama
        print("🚀 Initializing FastImageCaptioner...")
        self.fast_captioner = FastImageCaptioner(model_name=FAST_CAPTIONER_MODEL)
        
        # Keep old variables for compatibility (but won't be used)
        self.ollama_url = OLLAMA_API_URL
        self.ollama_model = OLLAMA_MODEL
        self.ollama_temperature = OLLAMA_TEMPERATURE
        self.caption_history = {}
        # Initialize Pinecone embedder for storing captions
        self.embedder = Embedder()

    # Legacy methods kept for compatibility (no longer used)
    def encode_image_to_base64(self, image_path):
        """Convert an image to base64 string for Ollama API (LEGACY - no longer used)"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    def encode_cv2_image_to_base64(self, cv2_image):
        """Convert a cv2 image to base64 string for Ollama API (LEGACY - no longer used)"""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate_caption(self, image_path=None, cv2_image=None):
        """
        Generate a detailed description of an image using FastImageCaptioner
        
        Args:
            image_path: Path to the image file (if available)
            cv2_image: CV2 image object (if path not available)
        
        Returns:
            str: A detailed description of the image
        """
        try:
            # Use FastImageCaptioner for caption generation
            if image_path and os.path.exists(image_path):
                # Use image path directly
                result = self.fast_captioner.caption_single_image(image_path)
                return result['caption']
            elif cv2_image is not None:
                # Convert CV2 image to PIL Image for FastImageCaptioner
                rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                result = self.fast_captioner.caption_single_image(pil_image)
                return result['caption']
            else:
                return "Error: No valid image provided"
                
        except Exception as e:
            error_msg = f"Error generating caption with FastImageCaptioner: {str(e)}"
            print(error_msg)
            return error_msg
            
    def caption_and_save(self, image_path=None, cv2_image=None, caption_file_path=None):
        """
        Generate caption for an image and save it to a text file
        
        Args:
            image_path: Path to the image file (if available)
            cv2_image: CV2 image object (if path not available)
            caption_file_path: Path to save the caption (if None, will be derived from image_path)
        
        Returns:
            str: The generated caption
        """
        # Generate the caption
        caption = self.generate_caption(image_path, cv2_image)
        
        # Determine where to save the caption
        if caption_file_path is None and image_path is not None:
            caption_file_path = image_path.rsplit('.', 1)[0] + '.txt'
        elif caption_file_path is None and cv2_image is not None:
            # Generate a timestamp-based filename if we only have the CV2 image
            timestamp = int(time.time())
            caption_file_path = os.path.join(self.output_dir, f"caption_{timestamp}.txt")
        
        # Save the caption to a file (embedding will be done later)
        if caption_file_path:
            try:
                with open(caption_file_path, 'w') as f:
                    f.write(caption)
                print(f"Caption saved to {caption_file_path}")
                    
            except Exception as e:
                print(f"Error saving caption to file: {str(e)}")
        
        return caption
    
    def upload_all_captions_to_embeddings(self):
        """
        Upload all caption files to Pinecone embeddings after all captioning is complete
        """
        print("🔄 Starting batch embedding upload to Pinecone...")
        
        # Find all .txt files in the output directory and subdirectories
        caption_files = []
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith('.txt'):
                    caption_files.append(os.path.join(root, file))
        
        if not caption_files:
            print("No caption files found for embedding.")
            return
        
        print(f"Found {len(caption_files)} caption files to upload...")
        
        # Upload each caption file to Pinecone
        successful_uploads = 0
        failed_uploads = 0
        
        for caption_file in caption_files:
            try:
                self.embedder.upload_embeddings(caption_file)
                successful_uploads += 1
                print(f"✅ Uploaded: {os.path.basename(caption_file)}")
            except Exception as e:
                failed_uploads += 1
                print(f"❌ Failed to upload {os.path.basename(caption_file)}: {str(e)}")
        
        print(f"\n📊 Embedding Upload Summary:")
        print(f"   Successful uploads: {successful_uploads}")
        print(f"   Failed uploads: {failed_uploads}")
        print(f"   Total files: {len(caption_files)}")
        
        if successful_uploads > 0:
            print("✅ Embedding upload process completed!")
        else:
            print("❌ No embeddings were successfully uploaded.")
