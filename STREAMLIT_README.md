# Video Surveillance Analysis System - Streamlit Interface

A comprehensive web interface for your video surveillance analysis system that combines YOLO object detection with AI-powered captioning and semantic search capabilities.

## 🌟 Features

### 📁 Video Processing
- **Upload Videos**: Drag and drop video files directly into the interface
- **Batch Processing**: Process multiple videos simultaneously
- **Real-time Progress**: Monitor processing status with progress bars
- **Format Support**: MP4, AVI, MOV, MKV formats supported

### 🔍 Search & Analysis
- **Semantic Search**: Search through video content using natural language
- **AI-Powered Results**: Find specific scenes, objects, or activities
- **Visual Results**: See matching images with confidence scores
- **Caption Integration**: Leverage BLIP model captions for search

### 📊 Results Dashboard
- **Timeline Visualization**: Track analysis activity over time
- **Statistics Overview**: View processing metrics and counts
- **Image Gallery**: Browse all processed images with captions
- **Pagination**: Navigate through large result sets efficiently

### 🛠️ System Management
- **Health Monitoring**: Check system component status
- **Configuration View**: Review current settings
- **Data Management**: Clear results and refresh data
- **Performance Metrics**: Monitor processing rates

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Launch the Interface
```bash
python launch_streamlit.py
```
or
```bash
streamlit run streamlit_app.py
```

### 3. Access the Interface
Open your browser and navigate to: `http://localhost:8501`

## 📋 Prerequisites

### Required Services
- **YOLO Model**: Ultralytics YOLO model for object detection
- **FastImageCaptioner**: BLIP model for image captioning
- **Pinecone**: Vector database for semantic search

### System Requirements
- Python 3.8+
- GPU recommended for faster processing
- Sufficient disk space for video files and results

## 🎯 Usage Guide

### Processing Videos

1. **Upload Videos**:
   - Use the file uploader in the "Video Processing" tab
   - Or place videos in the `input_videos/` directory

2. **Load YOLO Model**:
   - Click "Load YOLO Model" in the sidebar
   - Wait for successful loading confirmation

3. **Start Processing**:
   - Select videos to process
   - Click "Process Selected Videos"
   - Monitor progress in real-time

### Searching Content

1. **Enter Search Query**:
   - Use natural language descriptions
   - Examples: "person walking", "car on road", "suspicious activity"

2. **Review Results**:
   - Browse matching images with confidence scores
   - Read AI-generated captions
   - Access original image files

### Monitoring Results

1. **Dashboard Overview**:
   - Check processing statistics
   - View timeline charts
   - Browse recent results

2. **System Health**:
   - Monitor component status
   - Check directory structure
   - Review performance metrics

## ⚙️ Configuration

### Key Settings
- **Snapshot Interval**: Object detection frequency (default: 1s)
- **Screenshot Interval**: Full scene capture frequency (default: 15s)
- **Confidence Threshold**: Object detection sensitivity (default: 0.5)
- **Model Paths**: YOLO and captioning model configurations

### Directory Structure
```
fast_search/
├── input_videos/          # Place video files here
├── output_snapshots/      # Processed results stored here
├── src/                   # Core system modules
├── streamlit_app.py       # Main interface
├── launch_streamlit.py    # Launcher script
└── requirements_streamlit.txt
```

## 🔧 Troubleshooting

### Common Issues

1. **YOLO Model Loading Failed**:
   - Check internet connection for hub models
   - Verify Ultralytics API key is set
   - Ensure model path is correct

2. **Captioning Service Unavailable**:
   - Install required packages: `pip install torch transformers`
   - Check GPU availability for faster processing

3. **Search Not Working**:
   - Ensure Pinecone API key is configured
   - Check if embeddings have been generated
   - Verify captioning service is working

4. **No Videos Found**:
   - Check `input_videos/` directory exists
   - Verify video file formats are supported
   - Ensure proper file permissions

### Performance Tips

1. **GPU Acceleration**:
   - Install CUDA-compatible PyTorch
   - Monitor GPU memory usage
   - Use appropriate batch sizes

2. **Storage Management**:
   - Regularly clear old results
   - Monitor disk space usage
   - Compress video files if needed

3. **Processing Optimization**:
   - Adjust snapshot intervals based on needs
   - Process videos during off-peak hours
   - Use smaller video files for testing

### Troubleshooting

#### OSError: [WinError 123] The filename, directory name, or volume label syntax is incorrect

This error means your path string contains an invalid character, often a hidden control character like `\x0c` (form feed).



## 🛡️ Security Considerations

- Store API keys securely (use environment variables)
- Restrict access to surveillance footage
- Regular backup of analysis results
- Monitor system resource usage

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review system logs in the interface
3. Verify all dependencies are installed
4. Check configuration settings

## 🔄 Updates

To update the interface:
1. Pull latest changes
2. Update dependencies: `pip install -r requirements_streamlit.txt`
3. Restart the Streamlit server

---

**Note**: This interface provides a user-friendly way to interact with your surveillance analysis system. For advanced configuration or troubleshooting, refer to the main system documentation.
