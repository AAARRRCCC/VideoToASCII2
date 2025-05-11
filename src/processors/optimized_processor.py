import os
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ..core.video_processor import VideoProcessor
from ..core.ascii_converter import ASCIIConverter
from ..core.renderer import Renderer, render_ascii_frame_to_image_static
from ..utils.helpers import create_directory_if_not_exists

# Helper function for parallel rendering - must be at module level for pickling
def render_task_helper(task):
    """Unpacks and calls render_ascii_frame_to_image_static with the task arguments"""
    return render_ascii_frame_to_image_static(*task)

def process_video_optimized(input_path, output_path, width=120, height=60, 
                           processes=None, batch_size=10, font_size=12, fps=None, temp_dir="./temp", scale=1.0):
    """
    Process a video using optimized parallelism.
    
    This implementation focuses on optimizing performance through:
    - Efficient memory usage
    - Minimizing disk I/O
    - Optimized parallel processing
    - Better error handling
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        width (int): Width of ASCII output in characters
        height (int): Height of ASCII output in characters
        processes (int): Number of processes to use
        batch_size (int): Number of frames to process in each batch
        font_size (int): Font size for ASCII characters
        fps (int): Frames per second of output video (if None, use input video's FPS)
        temp_dir (str): Directory for temporary files
        scale (float): Scale factor for output resolution
    """
    import sys
    print("[DEBUG] Starting process_video_optimized")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Set default number of processes to CPU count if not specified
    num_processes = processes if processes is not None else os.cpu_count()
    
    # Create temporary directory
    create_directory_if_not_exists(temp_dir)
    frames_dir = os.path.join(temp_dir, "frames")
    create_directory_if_not_exists(frames_dir)
    
    try:
        print("[DEBUG] Initializing components")
        sys.stdout.flush()
        
        # Initialize components with optimized settings
        video_processor = VideoProcessor(num_processes=num_processes, scale=scale)
        ascii_converter = ASCIIConverter(num_processes=num_processes)
        renderer = Renderer(font_size=font_size, fps=fps if fps is not None else 30, num_processes=num_processes)
        
        # Downscale video first
        print(f"Downscaling video: {input_path}")
        sys.stdout.flush()
        
        downscaled_video, actual_dimensions = video_processor.downscale_video(
            input_path,
            os.path.join(temp_dir, "downscaled.mp4"),
            width,
            height
        )
        
        print("[DEBUG] Video downscaled successfully")
        sys.stdout.flush()
        
        # Get video properties
        cap = cv2.VideoCapture(downscaled_video)
        if not cap.isOpened():
            print(f"[DEBUG] Could not open downscaled video: {downscaled_video}")
            sys.stdout.flush()
            raise RuntimeError(f"Could not open downscaled video: {downscaled_video}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[DEBUG] Video properties: {total_frames} frames, {video_fps} fps")
        sys.stdout.flush()
        
        # Use input video FPS if not specified
        if fps is None:
            fps = int(video_fps)
            print(f"Using input video FPS: {fps}")
            renderer.fps = fps
        
        # Calculate output image dimensions
        actual_width, actual_height = actual_dimensions
        char_width = font_size
        char_height = font_size
        img_width = actual_width * char_width
        img_height = actual_height * char_height
        
        print(f"ASCII dimensions: {actual_width}x{actual_height} characters")
        print(f"Output dimensions: {img_width}x{img_height} pixels")
        sys.stdout.flush()
        
        # Optimized frame extraction and processing
        # Instead of loading all frames into memory at once, we process them in batches
        # This reduces memory usage and improves performance
        
        # Calculate total number of batches
        num_batches = (total_frames + batch_size - 1) // batch_size
        
        # Process each batch
        all_frame_paths = []
        
        print(f"Processing {total_frames} frames in {num_batches} batches of size {batch_size}")
        sys.stdout.flush()
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_frames)
            batch_size_actual = batch_end - batch_start
            
            # Extract frames for this batch
            frames = []
            frame_indices = []
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
            for i in range(batch_size_actual):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_indices.append(batch_start + i)
            
            # Convert frames to ASCII art
            ascii_frames = [None] * len(frames)
            
            # Process frames in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit batch for parallel processing
                future_to_index = {
                    executor.submit(
                        ascii_converter.convert_frame_to_ascii,
                        frame,
                        actual_width,
                        actual_height
                    ): idx
                    for idx, frame in enumerate(frames)
                }
                
                # Collect results while preserving order
                for future in future_to_index:
                    try:
                        ascii_frame = future.result()
                        idx = future_to_index[future]
                        ascii_frames[idx] = ascii_frame
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        sys.stdout.flush()
            
            # Render ASCII frames to images
            frame_paths = []
            render_tasks = []
            
            for i, ascii_frame in enumerate(ascii_frames):
                if ascii_frame is not None:
                    frame_idx = frame_indices[i]
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                    frame_paths.append((frame_idx, frame_path))
                    render_tasks.append((ascii_frame, frame_path, img_width, img_height, font_size))
            
            # Render frames in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Process all rendering tasks using the module-level helper function
                for _ in executor.map(render_task_helper, render_tasks):
                    pass
            
            # Add frame paths to the complete list
            all_frame_paths.extend(frame_paths)
        
        # Close the video capture
        cap.release()
        
        # Sort frame paths by frame index
        all_frame_paths.sort(key=lambda x: x[0])
        sorted_frame_paths = [path for _, path in all_frame_paths]
        
        # Create video from rendered frames
        print("Creating final video...")
        sys.stdout.flush()
        
        renderer._create_video_from_frames(sorted_frame_paths, output_path, img_width, img_height)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Processed {len(sorted_frame_paths)} frames at {len(sorted_frame_paths) / (end_time - start_time):.2f} fps")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[DEBUG] Error in process_video_optimized: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
        
    finally:
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {temp_dir}")
            print(f"Error: {str(e)}")
            print("You may want to delete it manually.")
            sys.stdout.flush()