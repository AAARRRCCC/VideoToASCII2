import subprocess
import os
import cv2
import sys

class VideoProcessor:
    def __init__(self):
        self._validate_ffmpeg()
    
    def _validate_ffmpeg(self):
        """Validate that ffmpeg is installed and accessible."""
        try:
            # Use shell=True for Windows command execution
            subprocess.run('ffmpeg -version', check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("ffmpeg is not installed or not accessible in PATH")
    
    def downscale_video(self, input_path, output_path, width, height):
        """
        Downscale the input video using ffmpeg while preserving aspect ratio.
        
        Args:
            input_path (str): Path to input video file
            output_path (str): Path to save downscaled video
            width (int): Target width
            height (int): Target height
            
        Returns:
            str: Path to downscaled video
            tuple: Actual dimensions (width, height) after preserving aspect ratio
        """
        # Ensure input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Get original video dimensions
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_path}")
        
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate new dimensions that preserve aspect ratio
        original_aspect = original_width / original_height
        
        # If both width and height are specified, prioritize width for aspect ratio
        new_width = width
        new_height = int(new_width / original_aspect)
        
        # If calculated height exceeds specified height, recalculate based on height
        if new_height > height:
            new_height = height
            new_width = int(new_height * original_aspect)
        
        print(f"Original dimensions: {original_width}x{original_height}")
        print(f"New dimensions (preserving aspect ratio): {new_width}x{new_height}")
        
        # Create ffmpeg command for downscaling with Windows-safe quoting
        cmd = (
            f'ffmpeg -i "{input_path}" '
            f'-vf "scale={new_width}:{new_height}" '
            f'-c:v libx264 -crf 23 -preset fast -y "{output_path}"'
        )
        
        # Execute ffmpeg command
        try:
            # Use shell=True for Windows command execution
            process = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, shell=True)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Error downscaling video: {error_message}")
        
        return output_path, (new_width, new_height)
    
    def extract_frames(self, video_path):
        """
        Extract frames from the video as numpy arrays.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            list: List of frames as numpy arrays
            float: FPS of the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, fps