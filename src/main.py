from ultralytics import YOLO
import os
import cv2
import time
from .config import INPUT_VIDEOS_PATH, OUTPUT_SNAPSHOTS_PATH, SNAPSHOT_INTERVAL, SCREENSHOT_INTERVAL, YOLO_MODEL_PATH
from .analyzer import VideoAnalyzer
from .search import Searcher
import argparse # Added import

def check_captioning_service():
    """Check if FastImageCaptioner dependencies are available"""
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # Check if CUDA is available (optional but recommended)
        if torch.cuda.is_available():
            print(f"INFO: CUDA available - GPU acceleration enabled ({torch.cuda.get_device_name()})")
        else:
            print("INFO: CUDA not available - using CPU (will be slower)")
            
        print("INFO: FastImageCaptioner dependencies are available. Captioning enabled.")
        return True
        
    except ImportError as e:
        print(f"WARNING: FastImageCaptioner dependencies missing: {e}")
        print("Please install: pip install torch transformers")
        print("Captioning will be disabled.")
        return False
    except Exception as e:
        print(f"WARNING: Error checking captioning service: {e}")
        print("Captioning will be disabled.")
        return False

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_SNAPSHOTS_PATH):
        os.makedirs(OUTPUT_SNAPSHOTS_PATH)

    # Check if FastImageCaptioner is available for captioning
    enable_captions = check_captioning_service()

    # Load the YOLO model
    # Use the model path you specified
    model_path = YOLO_MODEL_PATH
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the model path is correct and you have a working internet connection if it's a hub model.")
        print("If the model is local, ensure 'weights/hub/6y8UOPC2UfMyoSlxNI28/best.pt' exists or provide the correct local path.")
        return

    # Store video analyzer for embedding upload later
    video_analyzer_instance = None

    # Iterate through all videos in the input directory
    for video_file in os.listdir(INPUT_VIDEOS_PATH):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')): # Added .mkv
            video_path = os.path.join(INPUT_VIDEOS_PATH, video_file)
            print(f"Analyzing video: {video_path}")

            # Initialize the video analyzer for each video
            video_analyzer_instance = VideoAnalyzer(
                model=model,
                input_video_path=video_path,
                output_dir=OUTPUT_SNAPSHOTS_PATH,
                snapshot_interval=SNAPSHOT_INTERVAL,
                screenshot_interval=SCREENSHOT_INTERVAL,
                enable_captions=enable_captions
            )
            video_analyzer_instance.analyze_video()

    # After all videos are processed, upload embeddings if captions were enabled
    if enable_captions and video_analyzer_instance is not None:
        print("\n🚀 All video analysis complete. Starting embedding upload process...")
        try:
            # Access the captioner from the last video analyzer instance
            video_analyzer_instance.captioner.upload_all_captions_to_embeddings()
        except Exception as e:
            print(f"Error during embedding upload: {str(e)}")

    print("\n✅ All processing complete!")

    # Example of how to use the Searcher after analysis
    # Removed input prompt from main analysis flow
    # if enable_captions: # Only offer search if captions (and thus embeddings) were likely generated
    #     search_query = input("Enter a search query for the video analysis (or press Enter to skip): ")
    #     if search_query:
    #         perform_search(search_query)

def perform_search(query_text):
    """Performs a similarity search and prints the results."""
    print(f"\nPerforming search for: '{query_text}'")
    searcher = Searcher()
    results = searcher.search_embeddings(query_text)
    
    if results:
        print(f"Found {len(results)} results:")
        for i, match in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  ID: {match.get('id')}")
            print(f"  Score: {match.get('score')}")
            metadata = match.get('metadata', {})
            print(f"  Text: {metadata.get('text')}")
            print(f"  Image Path: {match.get('image_path')}")
    else:
        print("No results found.")

def embed_only():
    """Run only the embedding generation and upload process for existing captions."""
    print("🔄 Starting embedding-only process...")
    
    # Check if FastImageCaptioner is available
    if not check_captioning_service():
        print("❌ FastImageCaptioner service not available. Cannot generate embeddings.")
        return
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_SNAPSHOTS_PATH):
        print(f"❌ Output directory not found: {OUTPUT_SNAPSHOTS_PATH}")
        print("Please run video processing first to generate captions.")
        return
    
    # Initialize captioner for embedding upload
    from .captioner import Captioner
    captioner = Captioner(None, None, OUTPUT_SNAPSHOTS_PATH)
    
    try:
        captioner.upload_all_captions_to_embeddings()
        print("✅ Embedding process completed successfully!")
    except Exception as e:
        print(f"❌ Error during embedding process: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Analysis and Search Tool")
    parser.add_argument("--query", type=str, help="Perform a similarity search with the given query text.")
    parser.add_argument("--embed-only", action="store_true", help="Run only the embedding generation process for existing captions.")
    
    args = parser.parse_args()
    
    if args.embed_only:
        embed_only()
    elif args.query:
        # Check FastImageCaptioner service before attempting search, as embeddings depend on it.
        # This is a good practice, though embeddings might exist from a previous run.
        if check_captioning_service(): 
            perform_search(args.query)
        else:
            print("INFO: FastImageCaptioner service not available. Search might not yield results if embeddings weren't generated previously.")
            # Optionally, you could still proceed with search if you know embeddings exist:
            # perform_search(args.query)
    else:
        main()