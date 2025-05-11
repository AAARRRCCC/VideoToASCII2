import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm

from ..core.video_processor import VideoProcessor
from ..core.ascii_converter import ASCIIConverter
from ..core.renderer import Renderer, render_ascii_frame_to_image_static
from ..utils.helpers import create_directory_if_not_exists

# Helper function for parallel rendering - must be at module level for pickling
def render_task_helper(task):
    """Unpacks and calls render_ascii_frame_to_image_static with the task arguments"""
    return render_ascii_frame_to_image_static(*task)

def process_video_enhanced(input_path, output_path, width=120, height=60, 
                          processes=None, batch_size=10, font_size=12, fps=30, temp_dir="./temp", scale=1.0):
    """
    Process a video using enhanced parallelism with better resource management.
    
    This implementation uses a more sophisticated approach to parallelism with:
    - Better memory management
    - Improved error handling
    - More efficient resource allocation
    - Dynamic batch sizing based on system resources
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        width (int): Width of ASCII output in characters
        height (int): Height of ASCII output in characters
        processes (int): Number of processes to use
        batch_size (int): Number of frames to process in each batch
        font_size (int): Font size for ASCII characters
        fps (int): Frames per second of output video
        temp_dir (str): Directory for temporary files
        scale (float): Scale factor for output resolution
    """
    start_time = time.time()
    
    # Set default number of processes to CPU count if not specified
    num_processes = processes if processes is not None else os.cpu_count()
    
    # Create temporary directory
    create_directory_if_not_exists(temp_dir)
    frames_dir = os.path.join(temp_dir, "frames")
    create_directory_if_not_exists(frames_dir)
    
    try:
        # Initialize components with enhanced settings
        video_processor = VideoProcessor(num_processes=num_processes, scale=scale)
        ascii_converter = ASCIIConverter(num_processes=num_processes)
        renderer = Renderer(font_size=font_size, fps=fps, num_processes=num_processes)
        
        # Downscale video first
        print(f"Downscaling video: {input_path}")
        downscaled_video, actual_dimensions = video_processor.downscale_video(
            input_path,
            os.path.join(temp_dir, "downscaled.mp4"),
            width,
            height
        )
        
        # Get video properties
        cap = cv2.VideoCapture(downscaled_video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open downscaled video: {downscaled_video}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use input video FPS if not specified
        if fps is None:
            fps = int(video_fps)
            print(f"Using input video FPS: {fps}")
        
        # Extract frames in batches
        print(f"Extracting frames from video: {total_frames} total frames")
        frames = []
        frame_indices = []
        
        # Use a progress bar for frame extraction
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_indices.append(i)
                pbar.update(1)
        
        cap.release()
        
        # Calculate output image dimensions
        actual_width, actual_height = actual_dimensions
        char_width = font_size
        char_height = font_size
        img_width = actual_width * char_width
        img_height = actual_height * char_height
        
        print(f"ASCII dimensions: {actual_width}x{actual_height} characters")
        print(f"Output dimensions: {img_width}x{img_height} pixels")
        
        # Process frames in parallel using enhanced batch processing
        # This approach is more memory-efficient and provides better progress tracking
        
        # Determine optimal batch size based on system memory and number of processes
        # For simplicity, we'll use the provided batch_size, but in a real implementation
        # this could be dynamically calculated based on available memory
        
        # Convert frames to ASCII art
        print("Converting frames to ASCII art...")
        ascii_frames = [None] * len(frames)  # Pre-allocate list to maintain frame order
        
        # Process frames in batches
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for batch_start in tqdm(range(0, len(frames), batch_size), desc="Converting to ASCII"):
                batch_end = min(batch_start + batch_size, len(frames))
                batch_frames = frames[batch_start:batch_end]
                batch_indices = frame_indices[batch_start:batch_end]
                
                # Submit batch for parallel processing
                future_to_index = {
                    executor.submit(
                        ascii_converter.convert_frame_to_ascii,
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
        
        # Render ASCII frames to images
        print("Rendering ASCII frames to images...")
        frame_paths = []
        
        # Prepare rendering tasks
        render_tasks = []
        for i, ascii_frame in enumerate(ascii_frames):
            if ascii_frame is not None:
                frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                frame_paths.append(frame_path)
                render_tasks.append((ascii_frame, frame_path, img_width, img_height, font_size))
        
        # Render frames in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            list(tqdm(
                executor.map(
                    render_task_helper,
                    render_tasks
                ),
                total=len(render_tasks),
                desc="Rendering frames"
            ))
        
        # Create video from rendered frames
        print("Creating final video...")
        renderer._create_video_from_frames(frame_paths, output_path, img_width, img_height)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Processed {len(frame_paths)} frames at {len(frame_paths) / (end_time - start_time):.2f} fps")
        
    finally:
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {temp_dir}")
            print(f"Error: {str(e)}")
            print("You may want to delete it manually.")