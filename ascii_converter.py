import numpy as np
import cv2
from character_mapper import CharacterMapper
from tqdm import tqdm

class ASCIIConverter:
    def __init__(self):
        self.character_mapper = CharacterMapper()
    
    def convert_video_to_ascii(self, video_path, width, height):
        """
        Convert video frames to ASCII art.
        
        Args:
            video_path (str): Path to video file
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            
        Returns:
            list: List of ASCII art frames
            tuple: Actual dimensions (width, height) used for ASCII art
        """
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        # Get video dimensions
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate actual dimensions to use for ASCII art (preserving aspect ratio)
        aspect_ratio = video_width / video_height
        
        # Use the provided width and calculate height based on aspect ratio
        actual_width = width
        actual_height = int(actual_width / aspect_ratio)
        
        # If calculated height exceeds specified height, recalculate based on height
        if actual_height > height:
            actual_height = height
            actual_width = int(actual_height * aspect_ratio)
        
        print(f"ASCII dimensions (preserving aspect ratio): {actual_width}x{actual_height}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ascii_frames = []
        
        for _ in tqdm(range(total_frames), desc="Converting frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to ASCII
            ascii_frame = self.convert_frame_to_ascii(frame, actual_width, actual_height)
            ascii_frames.append(ascii_frame)
        
        cap.release()
        return ascii_frames, (actual_width, actual_height)
    
    def convert_frame_to_ascii(self, frame, width, height):
        """
        Convert a single video frame to ASCII art.
        
        Args:
            frame (numpy.ndarray): Input video frame
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            
        Returns:
            list: 2D list of ASCII characters
        """
        # Resize frame to match target dimensions while preserving aspect ratio
        resized_frame = cv2.resize(frame, (width, height))
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Create ASCII frame
        ascii_frame = []
        for y in range(height):
            ascii_row = []
            for x in range(width):
                pixel_value = gray_frame[y, x]
                character = self.character_mapper.map_pixel_to_character(pixel_value)
                ascii_row.append(character)
            ascii_frame.append(ascii_row)
        
        return ascii_frame