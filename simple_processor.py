import os
import time
import subprocess
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import tempfile
import shutil
from tqdm import tqdm

from video_processor import VideoProcessor
from ascii_converter import ASCIIConverter
from renderer import Renderer, render_ascii_frame_to_image_static
from utils import create_directory_if_not_exists

class SimpleProcessor:
    """
    A simplified processor that avoids using multiprocessing.Manager
    """
    
    def __init__(self, num_processes=None, batch_size=10, font_size=12, fps=30):
        """
        Initialize the simplified processor.
        
        Args:
            num_processes (int): Number of processes to use for parallel processing
            batch_size (int): Number of frames to process in each batch
            font_size (int): Font size for ASCII characters
            fps (int): Frames per second of output video
        """
        print("[DEBUG] Initializing SimpleProcessor")
        
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.batch_size = batch_size
        self.font_size = font_size
        self.fps = fps
        
        print(f"[DEBUG] Parameters: num_processes={self.num_processes}, batch_size={self.batch_size}")
        
        # Initialize components
        print("[DEBUG] Initializing VideoProcessor")
        self.video_processor = VideoProcessor(num_processes=self.num_processes)
        
        print("[DEBUG] Initializing ASCIIConverter")
        self.ascii_converter = ASCIIConverter(num_processes=self.num_processes)
        
        print("[DEBUG] Initializing Renderer")
        self.renderer = Renderer(font_size=self.font_size, fps=self.fps, num_processes=self.num_processes)
        
        print(f"[DEBUG] SimpleProcessor initialized with {self.num_processes} processes and batch size {self.batch_size}")
    
    def process_video(self, input_path, output_path, width=120, height=60, temp_dir=None):
        """
        Process a video using a simplified approach.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            temp_dir (str): Directory for temporary files
        """
        start_time = time.time()
        
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            create_directory_if_not_exists(temp_dir)
            
        frames_dir = os.path.join(temp_dir, "frames")
        create_directory_if_not_exists(frames_dir)
        
        try:
            # Step 1: Downscale video using ffmpeg
            print(f"Downscaling video: {input_path}")
            downscaled_video, actual_dimensions = self.video_processor.downscale_video(
                input_path,
                os.path.join(temp_dir, "downscaled.mp4"),
                width,
                height
            )
            
            # Step 2: Extract video information
            print("[DEBUG] Opening video file")
            cap = cv2.VideoCapture(downscaled_video)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {downscaled_video}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[DEBUG] Video has {total_frames} frames at {fps} fps")
            cap.release()
            
            # Calculate output image dimensions
            char_width = self.font_size
            char_height = self.font_size
            img_width = width * char_width
            img_height = height * char_height
            
            # Step 3: Extract all frames first
            print("[DEBUG] Extracting all frames")
            frames = []
            cap = cv2.VideoCapture(downscaled_video)
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append((i, frame))
            cap.release()
            print(f"[DEBUG] Extracted {len(frames)} frames")
            
            # Step 4: Process frames in parallel
            print("[DEBUG] Processing frames in parallel")
            frame_paths = [None] * total_frames
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Process frames in batches
                batch_size = min(self.batch_size, 10)  # Use a smaller batch size
                batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
                
                with tqdm(total=total_frames, desc="Processing frames") as pbar:
                    for batch in batches:
                        futures = []
                        for frame_idx, frame in batch:
                            future = executor.submit(
                                self._process_frame,
                                frame_idx,
                                frame,
                                width,
                                height,
                                img_width,
                                img_height,
                                self.font_size,
                                frames_dir
                            )
                            futures.append((frame_idx, future))
                        
                        # Wait for batch to complete
                        for frame_idx, future in futures:
                            frame_path = future.result()
                            frame_paths[frame_idx] = frame_path
                            pbar.update(1)
            
            # Step 5: Create video from frames
            print("[DEBUG] Creating video from frames")
            frame_paths = [path for path in frame_paths if path is not None]
            self._create_video_from_frames(
                frame_paths,
                output_path,
                img_width,
                img_height
            )
            
            end_time = time.time()
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {temp_dir}")
                print(f"Error: {str(e)}")
    
    def _process_frame(self, frame_idx, frame, width, height, img_width, img_height, font_size, frames_dir):
        """Process a single frame: convert to ASCII and render to image."""
        try:
            # Convert frame to ASCII
            ascii_frame = self.ascii_converter.convert_frame_to_ascii(frame, width, height)
            
            # Render ASCII frame to image
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            render_ascii_frame_to_image_static(ascii_frame, frame_path, img_width, img_height, font_size)
            
            return frame_path
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return None
    
    def _create_video_from_frames(self, frame_paths, output_path, width, height):
        """Create a video from a list of frame image paths using ffmpeg directly."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print("Creating final video...")
        
        # Create a temporary file listing all frames
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            frames_list_path = f.name
            for frame_path in frame_paths:
                f.write(f"file '{os.path.abspath(frame_path)}'\n")
        
        try:
            # Use ffmpeg directly
            cmd = (
                f'ffmpeg -y -r {self.fps} -f concat -safe 0 -i "{frames_list_path}" '
                f'-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast "{output_path}"'
            )
            
            # Execute ffmpeg command
            subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Video saved to {output_path}")
        finally:
            # Remove temporary file
            os.unlink(frames_list_path)


def process_video_simple(input_path, output_path, width=120, height=60, 
                        processes=None, batch_size=10, font_size=12, fps=30, temp_dir="./temp"):
    """
    Process a video using a simplified approach.
    
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
    """
    print(f"[DEBUG] process_video_simple called with input={input_path}, output={output_path}, processes={processes}, batch_size={batch_size}")
    
    try:
        print("[DEBUG] Creating SimpleProcessor instance")
        processor = SimpleProcessor(
            num_processes=processes,
            batch_size=batch_size,
            font_size=font_size,
            fps=fps
        )
        print("[DEBUG] SimpleProcessor instance created successfully")
        
        print("[DEBUG] Calling process_video method")
        processor.process_video(
            input_path=input_path,
            output_path=output_path,
            width=width,
            height=height,
            temp_dir=temp_dir
        )
        print("[DEBUG] process_video completed successfully")
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception in process_video_simple: {str(e)}")
        traceback.print_exc()
        raise