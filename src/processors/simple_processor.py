import cv2
import time
import os
from ..core.ascii_converter import ASCIIConverter

class SimpleProcessor:
    """
    A simple processor for converting videos to ASCII art.
    This is the basic implementation without any parallelism or optimizations.
    """
    
    def __init__(self):
        self.ascii_converter = ASCIIConverter()
    
    def process_video(self, video_path, output_path=None, width=100, height=30, fps_target=None, invert=False, colored=False):
        """
        Process a video file and convert it to ASCII art.
        
        Args:
            video_path (str): Path to the input video file.
            output_path (str, optional): Path to save the output ASCII video. If None, display in terminal.
            width (int, optional): Width of the ASCII output in characters. Defaults to 100.
            height (int, optional): Height of the ASCII output in characters. Defaults to 30.
            fps_target (int, optional): Target FPS for playback. If None, use original video FPS.
            invert (bool, optional): Whether to invert the brightness. Defaults to False.
            colored (bool, optional): Whether to use colored ASCII. Defaults to False.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use original FPS if target not specified
        fps = fps_target if fps_target is not None else original_fps
        frame_time = 1.0 / fps
        
        print(f"Processing video: {video_path}")
        print(f"Original FPS: {original_fps}, Target FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"ASCII dimensions: {width}x{height}")
        
        # Create output directory if needed
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_file = open(output_path, 'w', encoding='utf-8')
        
        # Process each frame
        frame_number = 0
        start_time = time.time()
        
        try:
            while True:
                # Measure frame processing time
                frame_start = time.time()
                
                # Read a frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to ASCII
                ascii_frame = self.ascii_converter.convert_frame_to_ascii(frame, width, height)
                
                # Convert 2D list to string
                ascii_str = '\n'.join([''.join(row) for row in ascii_frame])
                
                # Display or save the ASCII frame
                if output_path:
                    # Add frame delimiter for saved output
                    output_file.write(f"FRAME {frame_number}\n")
                    output_file.write(ascii_str)
                    output_file.write("\n\n")
                else:
                    # Clear terminal and display
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(ascii_str)
                
                frame_number += 1
                
                # Calculate processing time and sleep if needed
                process_time = time.time() - frame_start
                sleep_time = max(0, frame_time - process_time)
                
                if not output_path:  # Only sleep for real-time playback
                    time.sleep(sleep_time)
                
                # Print progress
                if frame_number % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_number / elapsed if elapsed > 0 else 0
                    print(f"\rProcessed {frame_number}/{frame_count} frames ({fps_actual:.2f} fps)", end="")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        # Clean up
        cap.release()
        if output_path:
            output_file.close()
        
        # Print final stats
        total_time = time.time() - start_time
        avg_fps = frame_number / total_time if total_time > 0 else 0
        print(f"\nProcessing complete: {frame_number} frames in {total_time:.2f} seconds ({avg_fps:.2f} fps)")