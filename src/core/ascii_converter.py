import numpy as np
import cv2
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from .character_mapper import CharacterMapper
from tqdm import tqdm

class ASCIIConverter:
    def __init__(self, num_processes=None):
        self.character_mapper = CharacterMapper()
        # Set default number of processes to CPU count if not specified
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
    
    def convert_video_to_ascii(self, video_path, width, height, batch_size=10):
        """
        Convert video frames to ASCII art using parallel processing.
        
        Args:
            video_path (str): Path to video file
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            batch_size (int): Number of frames to process in each batch
            
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
        print(f"Using {self.num_processes} processes for parallel conversion")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read all frames into memory (for large videos, we would use batch processing)
        frames = []
        frame_indices = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_indices.append(i)
        
        cap.release()
        
        # Process frames in parallel using batches to manage memory
        ascii_frames = [None] * len(frames)  # Pre-allocate list to maintain frame order
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Process frames in batches
            for batch_start in tqdm(range(0, len(frames), batch_size), desc="Converting frame batches"):
                batch_end = min(batch_start + batch_size, len(frames))
                batch_frames = frames[batch_start:batch_end]
                batch_indices = frame_indices[batch_start:batch_end]
                
                # Submit batch for parallel processing
                future_to_index = {
                    executor.submit(
                        self.convert_frame_to_ascii,
                        frame,
                        actual_width,
                        actual_height
                    ): idx
                    for frame, idx in zip(batch_frames, batch_indices)
                }
                
                # Collect results while preserving order
                for future in future_to_index:
                    try:
                        ascii_frame = future.result()
                        idx = future_to_index[future]
                        ascii_frames[idx] = ascii_frame
                    except Exception as e:
                        print(f"Error processing frame: {e}")
        
        return ascii_frames, (actual_width, actual_height)
    
    def convert_frame_to_ascii(self, frame, width, height):
        """
        Convert a single video frame to ASCII art using NumPy vectorization.
        
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
        
        # Create ASCII frame using vectorized operations
        ascii_frame = []
        
        # Vectorized approach for determining characters
        black_threshold = self.character_mapper.black_threshold
        black_char = self.character_mapper.black_char
        brightness_step = self.character_mapper.brightness_step
        japanese_chars = self.character_mapper.japanese_chars
        
        # Process each row
        for y in range(height):
            row_pixels = gray_frame[y, :]
            ascii_row = []
            
            # Apply the mapping logic to each pixel in the row
            for pixel_value in row_pixels:
                if pixel_value <= black_threshold:
                    ascii_row.append(black_char)
                else:
                    adjusted_value = pixel_value - black_threshold
                    char_index = min(
                        int(adjusted_value / brightness_step),
                        len(japanese_chars) - 1
                    )
                    ascii_row.append(japanese_chars[char_index])
            
            ascii_frame.append(ascii_row)
        
        return ascii_frame