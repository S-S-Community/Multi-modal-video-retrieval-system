# Configuration settings for the video surveillance project
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path to the directory containing input videos
INPUT_VIDEOS_PATH = os.getenv('INPUT_VIDEOS_PATH', './input_videos')

# Base output directory for snapshots
OUTPUT_SNAPSHOTS_PATH = os.getenv('OUTPUT_SNAPSHOTS_PATH', './output_snapshots')

# Interval for capturing snapshots of recognized classes (in seconds)
SNAPSHOT_INTERVAL = int(os.getenv('SNAPSHOT_INTERVAL', '1'))

# Interval for taking screenshots of the scene (in seconds)
SCREENSHOT_INTERVAL = int(os.getenv('SCREENSHOT_INTERVAL', '15'))

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

# List of recognized classes to capture
RECOGNIZED_CLASSES = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']  # Example classes, modify as needed

# Ollama settings (legacy - now using FastImageCaptioner)
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llava')
OLLAMA_TEMPERATURE = float(os.getenv('OLLAMA_TEMPERATURE', '0.7'))

# FastImageCaptioner settings
FAST_CAPTIONER_MODEL = os.getenv('FAST_CAPTIONER_MODEL', 'Salesforce/blip-image-captioning-base')
# Alternative models:
# - "Salesforce/blip-image-captioning-large" (slower, better quality)
# - "microsoft/git-base" (very fast, decent quality)
# - "nlpconnect/vit-gpt2-image-captioning" (fastest, basic quality)

#PINECONE settings
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")
    
PINECONE_EMBEDDINGS_MODEL = os.getenv('PINECONE_EMBEDDINGS_MODEL', 'llama-text-embed-v2')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'testpilot')

YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'https://hub.ultralytics.com/models/6y8UOPC2UfMyoSlxNI28')

OLLAMA_PROMPT = """You are an AI image-captioning assistant. Given a single image (either a cropped object snapshot or a full-scene screenshot) taken by a surveillance camera, which may produce black-and-white or color feed, generate a single, coherent sentence under 25 words.

If the image is a cropped object snapshot:
• Identify the object type and count (e.g., “a person”, “two cars”).
• Describe key attributes such as color, orientation, and accessories.
• Note any action or pose.

If the image is a full-scene screenshot:
• Describe the overall environment (indoor/outdoor, background elements).
• Mention the number and types of people or vehicles present.
• Include any obvious context (time of day, setting).
• Highlight any potential criminal activity or notable events (e.g., “a person vandalizing a window”, “a person being robbed”, "a person fainting", "people fighting with each other").

Be factual, concise, and omit speculation."""

