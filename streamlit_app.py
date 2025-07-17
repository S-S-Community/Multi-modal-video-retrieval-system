import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import cv2
import time
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import threading
import queue
import base64
from pathlib import Path

# Import your existing modules
from src.config import *
from src.analyzer import VideoAnalyzer
from src.search import Searcher
from src.captioner import Captioner
from ultralytics import YOLO

# Configure Streamlit page
st.set_page_config(
    page_title="Video Surveillance Analysis System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "Ready"
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None

def load_yolo_model():
    """Load YOLO model with caching"""
    if not st.session_state.model_loaded:
        try:
            with st.spinner("Loading YOLO model..."):
                st.session_state.yolo_model = YOLO(YOLO_MODEL_PATH)
                st.session_state.model_loaded = True
            st.success("YOLO model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to load YOLO model: {str(e)}")
            return False
    return True

def check_captioning_service():
    """Check if FastImageCaptioner dependencies are available"""
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        return True
    except ImportError:
        return False

def get_video_files():
    """Get list of video files in input directory"""
    if not os.path.exists(INPUT_VIDEOS_PATH):
        os.makedirs(INPUT_VIDEOS_PATH)
        return []
    
    video_files = []
    for file in os.listdir(INPUT_VIDEOS_PATH):
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(file)
    return video_files

def get_analysis_results():
    """Get analysis results from output directory"""
    results = []
    if not os.path.exists(OUTPUT_SNAPSHOTS_PATH):
        return results
    
    for root, dirs, files in os.walk(OUTPUT_SNAPSHOTS_PATH):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(root, file)
                # Try to find corresponding caption file
                caption_file = file_path.replace('.jpg', '_caption.txt').replace('.png', '_caption.txt').replace('.jpeg', '_caption.txt')
                caption = ""
                if os.path.exists(caption_file):
                    try:
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    except:
                        caption = "Caption not available"
                
                results.append({
                    'image_path': file_path,
                    'caption': caption,
                    'filename': file,
                    'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path))
                })
    
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)

def process_video(video_file, progress_bar, status_text):
    """Process a single video file"""
    if not st.session_state.model_loaded:
        if not load_yolo_model():
            return False
    
    try:
        video_path = os.path.join(INPUT_VIDEOS_PATH, video_file)
        status_text.text(f"Processing {video_file}...")
        
        # Initialize video analyzer
        video_analyzer = VideoAnalyzer(
            model=st.session_state.yolo_model,
            input_video_path=video_path,
            output_dir=OUTPUT_SNAPSHOTS_PATH,
            snapshot_interval=SNAPSHOT_INTERVAL,
            screenshot_interval=SCREENSHOT_INTERVAL,
            enable_captions=check_captioning_service()
        )
        
        # Process video
        video_analyzer.analyze_video()
        
        # Upload embeddings if captions are enabled
        if check_captioning_service():
            status_text.text("Uploading embeddings...")
            video_analyzer.captioner.upload_all_captions_to_embeddings()
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ {video_file} processed successfully!")
        return True
        
    except Exception as e:
        status_text.text(f"❌ Error processing {video_file}: {str(e)}")
        return False

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎥 Video Surveillance Analysis System</h1>
        <p>Analyze surveillance videos with YOLO object detection and AI-powered captioning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 System Controls")
    
    # System Status
    st.sidebar.subheader("📊 System Status")
    
    # Check service availability
    yolo_status = "✅ Available" if st.session_state.model_loaded else "⚠️ Not Loaded"
    caption_status = "✅ Available" if check_captioning_service() else "❌ Not Available"
    
    st.sidebar.info(f"""
    **YOLO Model:** {yolo_status}
    **Captioning Service:** {caption_status}
    **Processing Status:** {st.session_state.processing_status}
    """)
    
    # Load model button
    if not st.session_state.model_loaded:
        if st.sidebar.button("🚀 Load YOLO Model"):
            load_yolo_model()
    
    # Configuration
    st.sidebar.subheader("⚙️ Configuration")
    
    # Display current config
    with st.sidebar.expander("Current Settings"):
        st.write(f"**Snapshot Interval:** {SNAPSHOT_INTERVAL}s")
        st.write(f"**Screenshot Interval:** {SCREENSHOT_INTERVAL}s")
        st.write(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD}")
        st.write(f"**Model:** {YOLO_MODEL_PATH}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Video Processing", "🔍 Search & Analysis", "📊 Results Dashboard", "🛠️ System Info"])
    
    with tab1:
        st.header("Video Processing")
        
        # File upload
        st.subheader("📤 Upload Videos")
        uploaded_files = st.file_uploader(
            "Choose video files",
            type=['mp4', 'avi', 'mov', 'mkv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                if not os.path.exists(INPUT_VIDEOS_PATH):
                    os.makedirs(INPUT_VIDEOS_PATH)
                
                file_path = os.path.join(INPUT_VIDEOS_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ {uploaded_file.name} uploaded successfully")
        
        # Video list and processing
        st.subheader("📋 Available Videos")
        video_files = get_video_files()
        
        if video_files:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_videos = st.multiselect(
                    "Select videos to process:",
                    video_files,
                    default=video_files
                )
            
            with col2:
                process_all = st.button("🚀 Process Selected Videos", type="primary")
            
            if process_all and selected_videos:
                if not st.session_state.model_loaded:
                    st.warning("Please load the YOLO model first!")
                else:
                    st.session_state.processing_status = "Processing"
                    
                    # Create progress containers
                    progress_container = st.container()
                    
                    for i, video_file in enumerate(selected_videos):
                        with progress_container:
                            st.write(f"Processing video {i+1}/{len(selected_videos)}: {video_file}")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Process video
                            success = process_video(video_file, progress_bar, status_text)
                            
                            if success:
                                st.success(f"✅ {video_file} completed successfully")
                            else:
                                st.error(f"❌ Failed to process {video_file}")
                    
                    st.session_state.processing_status = "Ready"
                    st.balloons()
                    st.success("🎉 All videos processed!")
        else:
            st.info("No video files found. Please upload videos to get started.")
    
    with tab2:
        st.header("Search & Analysis")
        
        # Search functionality
        st.subheader("🔍 Semantic Search")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter search query:",
                placeholder="Example: person walking, car on road, suspicious activity..."
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and search_query:
            if not check_captioning_service():
                st.warning("Captioning service not available. Search may not work properly.")
            else:
                with st.spinner("Searching..."):
                    try:
                        searcher = Searcher()
                        results = searcher.search_embeddings(search_query)
                        st.session_state.search_results = results
                        
                        if results:
                            st.success(f"Found {len(results)} results for '{search_query}'")
                            
                            # Display results
                            for i, match in enumerate(results):
                                with st.expander(f"Result {i+1} (Score: {match.get('score', 0):.3f})"):
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        # Display image if available
                                        image_path = match.get('image_path')
                                        if image_path and os.path.exists(image_path):
                                            try:
                                                image = Image.open(image_path)
                                                st.image(image, caption=f"Match {i+1}", use_column_width=True)
                                            except Exception as e:
                                                st.warning(f"Could not load image: {e}")
                                        elif image_path: # Path was provided, but file not found by streamlit
                                            st.warning(f"Image file not found at: {image_path}")
                                        else: # image_path was None
                                            st.info("Image path not found by search module.")
                                    
                                    with col2:
                                        metadata = match.get('metadata', {})
                                        st.write(f"**Caption:** {metadata.get('text', 'No caption available')}")
                                        st.write(f"**ID:** {match.get('id', 'N/A')}")
                                        st.write(f"**Image Path:** {image_path}")
                        else:
                            st.info("No results found for your search query.")
                    
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
    
    with tab3:
        st.header("Results Dashboard")
        
        # Get analysis results
        results = get_analysis_results()
        
        if results:
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", len(results))
            
            with col2:
                captioned_count = len([r for r in results if r['caption']])
                st.metric("Captioned Images", captioned_count)
            
            with col3:
                # Group by date
                dates = [r['timestamp'].date() for r in results]
                unique_dates = len(set(dates))
                st.metric("Analysis Days", unique_dates)
            
            with col4:
                # Recent activity (last 24 hours)
                recent = [r for r in results if (datetime.now() - r['timestamp']).days == 0]
                st.metric("Today's Images", len(recent))
            
            # Timeline chart
            st.subheader("📈 Analysis Timeline")
            
            # Prepare data for timeline
            df = pd.DataFrame(results)
            df['date'] = df['timestamp'].dt.date
            timeline_data = df.groupby('date').size().reset_index(name='count')
            
            if len(timeline_data) > 0:
                fig = px.bar(timeline_data, x='date', y='count', 
                           title="Images Processed Per Day")
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent results
            st.subheader("📷 Recent Analysis Results")
            
            # Pagination
            items_per_page = 20
            total_pages = len(results) // items_per_page + (1 if len(results) % items_per_page > 0 else 0)
            
            if total_pages > 1:
                page = st.select_slider(
                    "Page",
                    options=list(range(1, total_pages + 1)),
                    value=1
                )
            else:
                page = 1
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(results))
            page_results = results[start_idx:end_idx]
            
            # Display results in grid
            cols = st.columns(4)
            for i, result in enumerate(page_results):
                col_idx = i % 4
                
                with cols[col_idx]:
                    try:
                        if os.path.exists(result['image_path']):
                            image = Image.open(result['image_path'])
                            st.image(image, caption=result['filename'], use_column_width=True)
                            
                            # Caption
                            if result['caption']:
                                st.caption(result['caption'])
                            
                            # Timestamp
                            st.caption(f"📅 {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
        
        else:
            st.info("No analysis results found. Process some videos first!")
    
    with tab4:
        st.header("System Information")
        
        # System configuration
        st.subheader("⚙️ Configuration")
        config_data = {
            "Parameter": [
                "Input Videos Path",
                "Output Snapshots Path", 
                "Snapshot Interval",
                "Screenshot Interval",
                "Confidence Threshold",
                "YOLO Model Path",
                "Captioning Model",
                "Pinecone Index"
            ],
            "Value": [
                INPUT_VIDEOS_PATH,
                OUTPUT_SNAPSHOTS_PATH,
                f"{SNAPSHOT_INTERVAL}s",
                f"{SCREENSHOT_INTERVAL}s",
                CONFIDENCE_THRESHOLD,
                YOLO_MODEL_PATH,
                FAST_CAPTIONER_MODEL,
                PINECONE_INDEX_NAME
            ]
        }
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
        
        # Directory structure
        st.subheader("📁 Directory Structure")
        
        # Check directories
        dirs_to_check = [INPUT_VIDEOS_PATH, OUTPUT_SNAPSHOTS_PATH]
        
        for dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                st.success(f"✅ {dir_path} - {file_count} files")
            else:
                st.warning(f"⚠️ {dir_path} - Directory not found")
        
        # System health check
        st.subheader("🔧 System Health")
        
        health_checks = [
            ("YOLO Model", st.session_state.model_loaded),
            ("Captioning Service", check_captioning_service()),
            ("Input Directory", os.path.exists(INPUT_VIDEOS_PATH)),
            ("Output Directory", os.path.exists(OUTPUT_SNAPSHOTS_PATH))
        ]
        
        for check_name, status in health_checks:
            if status:
                st.success(f"✅ {check_name}: OK")
            else:
                st.error(f"❌ {check_name}: Failed")
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        if results:
            # Processing rate
            total_images = len(results)
            if total_images > 0:
                time_range = max(r['timestamp'] for r in results) - min(r['timestamp'] for r in results)
                if time_range.total_seconds() > 0:
                    processing_rate = total_images / (time_range.total_seconds() / 3600)  # per hour
                    st.metric("Processing Rate", f"{processing_rate:.2f} images/hour")
        
        # Clear data button
        st.subheader("🗑️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Results", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    # Clear output directory
                    import shutil
                    if os.path.exists(OUTPUT_SNAPSHOTS_PATH):
                        shutil.rmtree(OUTPUT_SNAPSHOTS_PATH)
                        os.makedirs(OUTPUT_SNAPSHOTS_PATH)
                    st.success("All results cleared!")
                    st.session_state.confirm_clear = False
                    st.experimental_rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("Click again to confirm deletion!")
        
        with col2:
            if st.button("♻️ Refresh Data"):
                st.experimental_rerun()

if __name__ == "__main__":
    main()
