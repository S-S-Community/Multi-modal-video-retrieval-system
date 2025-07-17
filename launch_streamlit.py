#!/usr/bin/env python3
"""
Launcher script for the Video Surveillance Analysis System Streamlit interface.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        from PIL import Image
        print("✅ All Streamlit dependencies are available.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements_streamlit.txt")
        return False

def main():
    """Main launcher function."""
    print("🎥 Video Surveillance Analysis System")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "streamlit_app.py").exists():
        print("❌ streamlit_app.py not found in current directory.")
        print("Please run this script from the fast_search directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    directories = ["input_videos", "output_snapshots"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    print("\n🚀 Launching Streamlit interface...")
    print("The interface will open in your default web browser.")
    print("If it doesn't open automatically, visit: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down the interface. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
