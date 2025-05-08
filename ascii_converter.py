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
        """
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ascii_frames = []
        
        for _ in tqdm(range(total_frames), desc="Converting frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to ASCII
            ascii_frame = self.convert_frame_to_ascii(frame, width, height)
            ascii_frames.append(ascii_frame)
        
        cap.release()
        return ascii_frames
    
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
        # Resize frame to match target dimensions
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