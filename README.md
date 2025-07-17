# Video Surveillance Analysis System

A comprehensive AI-powered video surveillance analysis system that combines YOLO object detection with AI captioning and semantic search capabilities.

## 🌟 Features

- **🎯 Object Detection**: Real-time YOLO-based object detection and tracking
- **🤖 AI Captioning**: Automatic image captioning using BLIP models  
- **🔍 Semantic Search**: Natural language search through video content
- **🌐 Web Interface**: User-friendly Streamlit interface
- **⚡ Batch Processing**: Process multiple videos simultaneously
- **🗄️ Vector Database**: Pinecone integration for semantic search
- **📸 Smart Snapshots**: Automatic cropping of detected objects
- **🎬 Scene Capture**: Full scene screenshots at configurable intervals
- **📊 Progress Tracking**: Real-time processing progress and statistics
- **🔄 End-to-End Pipeline**: From raw video to searchable embeddings

## 🚀 Quick Start

### Prerequisites

- Python 3.10.16
- GPU recommended for faster processing
- Pinecone account and API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd fast_search
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Create required directories**:
   ```bash
   mkdir input_videos output_snapshots
   ```

### Usage

#### Web Interface (Recommended)
```bash
python launch_streamlit.py
# or
streamlit run streamlit_app.py
```

#### Command Line
```bash
# Run from the project root directory
python -m src.main
```

#### Search
```bash
# Run from the project root directory
python -m src.main --query "person walking"
```

#### Embedding Only
```bash
# Generate embeddings for existing captions (useful if embedding failed previously)
python -m src.main --embed-only
```

## 🔄 Complete Workflow

### Step-by-Step Process

The system follows a comprehensive pipeline to transform raw video files into searchable content:

#### 1. Video Input
- Place video files in `input_videos/` directory
- Supported formats: MP4, AVI, MOV, MKV
- Videos can be uploaded via web interface or placed manually

#### 2. YOLO Object Detection
```bash
# The system automatically:
# - Loads the YOLO model (YOLOv8 from Ultralytics)
# - Processes each video frame by frame
# - Detects objects with configurable confidence threshold
# - Tracks objects across frames with unique IDs
```

**What happens during detection:**
- **Snapshot Creation**: Crops detected objects and saves them as individual images
- **Screenshot Capture**: Takes full scene screenshots at regular intervals
- **Metadata Generation**: Creates text files with detection confidence and coordinates

#### 3. AI-Powered Captioning
```bash
# For each image (snapshots + screenshots):
# - BLIP model generates descriptive captions
# - Captions saved as .txt files alongside images
# - Optimized for surveillance scenarios
```

**Caption Examples:**
- Object snapshots: "a person in blue jacket walking"
- Full scenes: "outdoor parking lot with two cars and a person"

#### 4. Vector Embedding Generation
```bash
# The system:
# - Converts captions to vector embeddings
# - Uses sentence transformers for semantic representation
# - Uploads embeddings to Pinecone vector database
```

#### 5. Semantic Search
```bash
# Search capabilities:
# - Natural language queries
# - Semantic similarity matching
# - Returns relevant images with confidence scores
```

### Detailed Processing Flow

#### Option A: Web Interface (Streamlit)

1. **Start the interface:**
   ```bash
   python launch_streamlit.py
   ```

2. **Load YOLO Model:**
   - Click "Load YOLO Model" in sidebar
   - Wait for confirmation message

3. **Upload/Select Videos:**
   - Drag & drop videos OR select from `input_videos/`
   - Choose which videos to process

4. **Process Videos:**
   - Click "Process Selected Videos"
   - Monitor real-time progress bars
   - System automatically:
     - Runs YOLO detection
     - Generates captions
     - Creates embeddings
     - Uploads to Pinecone

5. **Search Content:**
   - Enter natural language query
   - Browse results with confidence scores
   - View original images and captions

#### Option B: Command Line Interface

1. **Place videos in input directory:**
   ```bash
   # Copy your videos
   Copy-Item "C:\path\to\your\video.mp4" ".\input_videos\"
   ```

2. **Run complete processing:**
   ```bash
   # Processes all videos in input_videos/
   python -m src.main
   ```

3. **Search processed content:**
   ```bash
   # Search with natural language
   python -m src.main --query "person walking"
   python -m src.main --query "red car in parking lot"
   python -m src.main --query "suspicious activity"
   ```

4. **Generate embeddings only (if needed):**
   ```bash
   # If embedding generation failed or was skipped previously
   python -m src.main --embed-only
   ```

### Output Structure

After processing, your `output_snapshots/` directory will contain:

```
output_snapshots/
├── person/                    # Object-based folders
│   ├── person_id1_timestamp.jpg    # Cropped person detection
│   ├── person_id1_timestamp.txt    # Caption for the image
│   └── ...
├── car/
│   ├── car_id5_timestamp.jpg
│   ├── car_id5_timestamp.txt
│   └── ...
├── screenshots/               # Full scene captures
│   ├── screenshot_video1_uuid/
│   │   ├── screenshot_0001.jpg
│   │   ├── screenshot_0001.txt
│   │   └── ...
│   └── screenshot_video2_uuid/
└── [other_object_classes]/
```

### Processing Configuration

Control the processing behavior via `.env`:

```env
# Detection frequency (every N seconds)
SNAPSHOT_INTERVAL=1          # Object detection snapshots
SCREENSHOT_INTERVAL=15       # Full scene screenshots

# Detection sensitivity
CONFIDENCE_THRESHOLD=0.5     # 0.0-1.0 (higher = more strict)

# Model settings
YOLO_MODEL_PATH=https://hub.ultralytics.com/models/your_model
FAST_CAPTIONER_MODEL=Salesforce/blip-image-captioning-base
```

### Search Examples

```bash
# Find people
python -m src.main --query "person walking"
python -m src.main --query "people talking"

# Find vehicles
python -m src.main --query "red car"
python -m src.main --query "motorcycle on street"

# Find activities
python -m src.main --query "person running"
python -m src.main --query "people fighting"

# Find scenes
python -m src.main --query "outdoor parking lot"
python -m src.main --query "indoor room with chairs"

# Security-focused
python -m src.main --query "suspicious activity"
python -m src.main --query "person breaking window"
```

### Performance Optimization

#### GPU Acceleration
```bash
# Install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Batch Processing Tips
- Process videos during off-peak hours
- Adjust intervals based on video content
- Use smaller test videos first
- Monitor disk space for outputs

## 📁 Project Structure

```
fast_search/
├── src/                    # Core modules
│   ├── analyzer.py         # Video analysis
│   ├── captioner.py        # Image captioning
│   ├── config.py           # Configuration
│   ├── embedder.py         # Vector embeddings
│   ├── main.py             # Main CLI
│   └── search.py           # Search functionality
├── input_videos/           # Input video files
├── output_snapshots/       # Processed results
├── streamlit_app.py        # Web interface
├── launch_streamlit.py     # Streamlit launcher
├── requirements.txt        # Dependencies
└── .env.example           # Environment template
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# API Keys (Required)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name
YOLO_MODEL_PATH=https://hub.ultralytics.com/models/your_model_id

# Directory Paths
INPUT_VIDEOS_PATH=./input_videos
OUTPUT_SNAPSHOTS_PATH=./output_snapshots

# Processing Intervals (seconds)
SNAPSHOT_INTERVAL=1          # How often to detect objects
SCREENSHOT_INTERVAL=15       # How often to capture full scenes

# Detection Settings
CONFIDENCE_THRESHOLD=0.5     # Object detection sensitivity (0.0-1.0)

# AI Models
FAST_CAPTIONER_MODEL=Salesforce/blip-image-captioning-base
PINECONE_EMBEDDINGS_MODEL=llama-text-embed-v2
```

### Supported Object Classes

The system can detect and track these object types:
```python
['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

### Performance Tuning

**For faster processing:**
- Increase `SNAPSHOT_INTERVAL` and `SCREENSHOT_INTERVAL`
- Use GPU acceleration
- Process smaller video files first

**For better accuracy:**
- Decrease `CONFIDENCE_THRESHOLD` 
- Use larger BLIP model: `Salesforce/blip-image-captioning-large`
- Increase video resolution

**For storage efficiency:**
- Increase processing intervals
- Regularly clean old results
- Compress videos before processing

## 🔧 Development

### Setting up for development

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

1. **YOLO Model Loading Failed**:
   ```bash
   # Check internet connection for hub models
   # Verify your model path in .env
   YOLO_MODEL_PATH=https://hub.ultralytics.com/models/your_model_id
   ```

2. **Path errors**: 
   - Ensure all paths in `.env` use forward slashes or raw strings
   - Check that input_videos/ and output_snapshots/ directories exist

3. **Missing API keys**: 
   ```bash
   # Check your .env file has required keys
   PINECONE_API_KEY=your_actual_api_key
   ```

4. **GPU issues**: 
   ```bash
   # Install CUDA-compatible PyTorch for GPU acceleration
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Import Errors**:
   ```bash
   # Make sure you're in the project root directory
   cd d:\keep\htx\fast_search
   python -m src.main  # NOT: cd src && python main.py
   ```

6. **No Search Results**:
   - Ensure videos have been processed first
   - Check that captions were generated (look for .txt files)
   - Verify Pinecone connection and embeddings upload
   - Try broader search terms

7. **Processing Stuck/Slow**:
   - Check available disk space
   - Monitor GPU memory usage
   - Reduce video resolution or length for testing
   - Adjust intervals in .env (increase SNAPSHOT_INTERVAL)

8. **Captioning Service Unavailable**:
   ```bash
   # Install required packages
   pip install torch transformers
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Debug Commands

```bash
# Test environment loading
python -c "from src.config import *; print('Config loaded successfully')"

# Test YOLO model
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO works')"

# Test Pinecone connection
python -c "from pinecone import Pinecone; print('Pinecone imported')"

# Check directory structure
ls input_videos/
ls output_snapshots/
```

