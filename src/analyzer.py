import os
import cv2
import time
import uuid # Added for generating unique IDs
from .config import RECOGNIZED_CLASSES as TARGET_CLASSES_FOR_SNAPSHOT
from .captioner import Captioner  # Import the Captioner class

class VideoAnalyzer:
    def __init__(self, model, input_video_path, output_dir, snapshot_interval=1, screenshot_interval=15, enable_captions=True):
        self.model = model
        self.input_video_path = input_video_path
        self.output_dir = output_dir # This is OUTPUT_SNAPSHOTS_PATH
        self.snapshot_interval = snapshot_interval
        self.screenshot_interval = screenshot_interval
        self.snapshotted_object_ids_this_video = set()
        self.current_video_screenshot_dir = "" # Will be set in analyze_video
        self.enable_captions = enable_captions
        # Initialize captioner if captions are enabled
        if enable_captions:
            self.captioner = Captioner(None, input_video_path, output_dir)
        else:
            self.captioner = None

    def analyze_video(self):
        print(f"INFO: Resetting predictor for video: {self.input_video_path}")
        if hasattr(self.model, 'predictor'):
            self.model.predictor = None
            print(f"INFO: self.model.predictor set to None for {self.input_video_path}")
        else:
            print(f"WARNING: self.model does not have a 'predictor' attribute to reset for {self.input_video_path}")

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.input_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: # Handle case where FPS is not available or video is invalid
            print(f"WARNING: Could not get valid FPS for video {self.input_video_path} (got {fps}). Defaulting to 25 FPS for interval calculations.")
            fps = 25 # Default FPS

        # Calculate how many frames to process for each interval
        # Process at least every frame if interval calculation results in less than 1
        snapshot_frames_to_process_every = max(1, int(round(fps * self.snapshot_interval)))
        screenshot_frames_to_process_every = max(1, int(round(fps * self.screenshot_interval)))

        # Create dedicated screenshot directory for this video
        video_filename = os.path.basename(self.input_video_path)
        video_name_without_ext, _ = os.path.splitext(video_filename)
        unique_id = str(uuid.uuid4())[:8]  # Shortened UUID for directory name
        
        base_screenshots_folder = os.path.join(self.output_dir, "screenshots")
        self.current_video_screenshot_dir = os.path.join(
            base_screenshots_folder,
            f"screenshot_{video_name_without_ext}_{unique_id}"
        )
        os.makedirs(self.current_video_screenshot_dir, exist_ok=True)
        print(f"INFO: Screenshots for {video_filename} will be saved to: {self.current_video_screenshot_dir}")
            
        self.snapshotted_object_ids_this_video = set() # Reset for each new video
        
        actual_frames_read = 0
        processed_for_snapshot_count = 0
        processed_for_screenshot_count = 0
        
        next_snapshot_at_frame = 0
        next_screenshot_at_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"INFO: End of video or error reading frame for {self.input_video_path} after {actual_frames_read} frames read.")
                break

            # Check if current frame should be processed for snapshots
            if actual_frames_read >= next_snapshot_at_frame:
                self.process_frame(frame)
                processed_for_snapshot_count += 1
                next_snapshot_at_frame += snapshot_frames_to_process_every

            # Check if current frame should be processed for screenshots
            if actual_frames_read >= next_screenshot_at_frame:
                # Using a more descriptive screenshot name including frame number
                self.take_screenshot(frame, f"scene_frame{actual_frames_read}_{int(time.time())}")
                processed_for_screenshot_count += 1
                next_screenshot_at_frame += screenshot_frames_to_process_every
            
            actual_frames_read += 1
            
        cap.release()
        print(f"INFO: Finished analyzing {self.input_video_path}. Total frames read: {actual_frames_read}. Frames processed for snapshot: {processed_for_snapshot_count} (target interval: every {snapshot_frames_to_process_every} frames). Frames processed for screenshot: {processed_for_screenshot_count} (target interval: every {screenshot_frames_to_process_every} frames).")

    def process_frame(self, frame):
        results_list = self.model.track(source=frame, show=False, conf=0.5, persist=True) 
        
        if not results_list: 
            return

        result = results_list[0]

        if result.boxes is not None and hasattr(result.boxes, 'cls') and result.boxes.cls is not None and \
           hasattr(result.boxes, 'id') and result.boxes.id is not None:
            
            detected_class_indices = result.boxes.cls.int().tolist()
            all_boxes_coords = result.boxes.xyxy.int().tolist()
            detected_track_ids = result.boxes.id.int().tolist()

            for i, cls_idx in enumerate(detected_class_indices):
                class_name = result.names.get(cls_idx)
                track_id = detected_track_ids[i] 
                
                if class_name and class_name in TARGET_CLASSES_FOR_SNAPSHOT:
                    object_key = (class_name, track_id)
                    if object_key not in self.snapshotted_object_ids_this_video:
                        self.snapshotted_object_ids_this_video.add(object_key)
                        current_box_coords = all_boxes_coords[i]
                        self.save_snapshot(frame, class_name, current_box_coords, track_id)
    
    def save_snapshot(self, frame, class_name_str, box_coords, track_id=None):
        x1, y1, x2, y2 = box_coords
        
        if x1 < x2 and y1 < y2:
            cropped_image = frame[y1:y2, x1:x2]
        else:
            print(f"Warning: Invalid bounding box coordinates for {class_name_str} ID {track_id}: {box_coords}. Skipping crop.")
            cropped_image = frame 

        if cropped_image.size == 0:
            print(f"Warning: Cropped image for {class_name_str} ID {track_id} is empty. Box coordinates: {box_coords}. Skipping snapshot.")
            return

        class_dir = os.path.join(self.output_dir, class_name_str)
        os.makedirs(class_dir, exist_ok=True)
        
        if track_id is not None:
            snapshot_filename = f"{class_name_str}_id{track_id}_{int(time.time())}.jpg"
        else:
            snapshot_filename = f"{class_name_str}_{int(time.time())}.jpg" # Fallback
            
        snapshot_path = os.path.join(class_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, cropped_image)
        
        # Generate caption if enabled
        if self.enable_captions and self.captioner:
            print(f"Generating caption for {snapshot_filename}...")
            self.captioner.caption_and_save(image_path=snapshot_path)

    def take_screenshot(self, frame, screenshot_name_suffix_str):
        # Save to the video-specific screenshot directory created in analyze_video
        # self.current_video_screenshot_dir is already created with os.makedirs
        screenshot_filename = f"{screenshot_name_suffix_str}.jpg"
        screenshot_path = os.path.join(self.current_video_screenshot_dir, screenshot_filename)
        cv2.imwrite(screenshot_path, frame)
        
        # Generate caption if enabled
        if self.enable_captions and self.captioner:
            print(f"Generating caption for full scene {screenshot_filename}...")
            self.captioner.caption_and_save(image_path=screenshot_path)