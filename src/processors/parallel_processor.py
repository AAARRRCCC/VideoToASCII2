import os
import time
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm

from ..core.video_processor import VideoProcessor
from ..core.ascii_converter import ASCIIConverter
from ..core.renderer import Renderer, render_ascii_frame_to_image_static
from ..utils.helpers import create_directory_if_not_exists

class ParallelProcessor:
    """
    A class that implements pipeline parallelism for the VideoToASCII conversion process.
    This allows different stages of the pipeline to run concurrently on different frames.
    """
    
    def __init__(self, num_processes=None, batch_size=10, font_size=12, fps=30):
        """
        Initialize the parallel processor.
        
        Args:
            num_processes (int): Number of processes to use for parallel processing
            batch_size (int): Number of frames to process in each batch
            font_size (int): Font size for ASCII characters
            fps (int): Frames per second of output video
        """
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.batch_size = batch_size
        self.font_size = font_size
        self.fps = fps
        
        # Initialize components
        self.video_processor = VideoProcessor(num_processes=self.num_processes)
        self.ascii_converter = ASCIIConverter(num_processes=self.num_processes)
        self.renderer = Renderer(font_size=self.font_size, fps=self.fps, num_processes=self.num_processes)
        
        # Create queues for pipeline stages
        self.frame_queue = mp.Queue(maxsize=self.batch_size * 2)
        self.ascii_queue = mp.Queue(maxsize=self.batch_size * 2)
        self.render_queue = mp.Queue(maxsize=self.batch_size * 2)
        
        # Create events for signaling
        self.extraction_done = mp.Event()
        self.conversion_done = mp.Event()
        self.rendering_done = mp.Event()
        
    def process_video(self, input_path, output_path, width=120, height=60, temp_dir="./temp"):
        """
        Process a video using pipeline parallelism.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            temp_dir (str): Directory for temporary files
        """
        start_time = time.time()
        
        # Create temporary directory
        create_directory_if_not_exists(temp_dir)
        frames_dir = os.path.join(temp_dir, "frames")
        create_directory_if_not_exists(frames_dir)
        
        try:
            # Downscale video first (this is still sequential as it uses ffmpeg)
            print(f"Downscaling video: {input_path}")
            downscaled_video, actual_dimensions = self.video_processor.downscale_video(
                input_path,
                os.path.join(temp_dir, "downscaled.mp4"),
                width,
                height
            )
            
            # Get video properties
            cap = cv2.VideoCapture(downscaled_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Calculate output image dimensions
            char_width = self.font_size
            char_height = self.font_size
            img_width = width * char_width
            img_height = height * char_height
            
            # Start pipeline processes
            print("Starting parallel processing pipeline...")
            
            # Start frame extraction process
            extraction_process = mp.Process(
                target=self._extract_frames_worker,
                args=(downscaled_video, total_frames)
            )
            extraction_process.start()
            
            # Start ASCII conversion processes
            conversion_processes = []
            for _ in range(max(1, self.num_processes // 3)):
                p = mp.Process(
                    target=self._convert_frames_worker,
                    args=(width, height)
                )
                p.start()
                conversion_processes.append(p)
            
            # Start rendering processes
            rendering_processes = []
            for _ in range(max(1, self.num_processes // 3)):
                p = mp.Process(
                    target=self._render_frames_worker,
                    args=(frames_dir, img_width, img_height, self.font_size)
                )
                p.start()
                rendering_processes.append(p)
            
            # Start progress monitoring in a separate thread
            progress_thread = threading.Thread(
                target=self._monitor_progress,
                args=(total_frames,)
            )
            progress_thread.start()
            
            # Wait for all processes to complete
            extraction_process.join()
            for p in conversion_processes:
                p.join()
            for p in rendering_processes:
                p.join()
            progress_thread.join()
            
            # Create video from rendered frames
            frame_paths = [os.path.join(frames_dir, f"frame_{i:06d}.png") for i in range(total_frames)]
            self._create_video_from_frames(frame_paths, output_path, img_width, img_height)
            
            end_time = time.time()
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            
        finally:
            # Clean up temporary files
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {temp_dir}")
                print(f"Error: {str(e)}")
    
    def _extract_frames_worker(self, video_path, total_frames):
        """Worker process for extracting frames from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_queue.put((i, frame))
            
            cap.release()
        except Exception as e:
            print(f"Error in frame extraction: {e}")
        finally:
            # Signal that extraction is done
            self.extraction_done.set()
    
    def _convert_frames_worker(self, width, height):
        """Worker process for converting frames to ASCII."""
        try:
            while not (self.extraction_done.is_set() and self.frame_queue.empty()):
                try:
                    # Get frame from queue with timeout
                    frame_data = self.frame_queue.get(timeout=0.1)
                    if frame_data is None:
                        continue
                    
                    frame_idx, frame = frame_data
                    
                    # Convert frame to ASCII
                    ascii_frame = self.ascii_converter.convert_frame_to_ascii(frame, width, height)
                    
                    # Put ASCII frame in queue
                    self.ascii_queue.put((frame_idx, ascii_frame))
                    
                except queue.Empty:
                    # Queue is empty but extraction might not be done
                    continue
        except Exception as e:
            print(f"Error in ASCII conversion: {e}")
        finally:
            # Signal that this conversion process is done
            if self.extraction_done.is_set() and self.frame_queue.empty():
                self.conversion_done.set()
    
    def _render_frames_worker(self, output_dir, img_width, img_height, font_size):
        """Worker process for rendering ASCII frames to images."""
        try:
            while not (self.conversion_done.is_set() and self.ascii_queue.empty()):
                try:
                    # Get ASCII frame from queue with timeout
                    ascii_data = self.ascii_queue.get(timeout=0.1)
                    if ascii_data is None:
                        continue
                    
                    frame_idx, ascii_frame = ascii_data
                    
                    # Render ASCII frame to image
                    frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                    render_ascii_frame_to_image_static(ascii_frame, frame_path, img_width, img_height, font_size)
                    
                    # Put rendered frame path in queue
                    self.render_queue.put((frame_idx, frame_path))
                    
                except queue.Empty:
                    # Queue is empty but conversion might not be done
                    continue
        except Exception as e:
            print(f"Error in frame rendering: {e}")
        finally:
            # Signal that this rendering process is done
            if self.conversion_done.is_set() and self.ascii_queue.empty():
                self.rendering_done.set()
    
    def _monitor_progress(self, total_frames):
        """Monitor progress of the pipeline and display a progress bar."""
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            completed = 0
            while completed < total_frames:
                # Count items in render queue
                try:
                    # Non-blocking get
                    self.render_queue.get(timeout=0.1)
                    completed += 1
                    pbar.update(1)
                except queue.Empty:
                    # If all processes are done and queue is empty, break
                    if self.rendering_done.is_set():
                        break
                    time.sleep(0.1)
    
    def _create_video_from_frames(self, frame_paths, output_path, width, height):
        """Create a video from a list of frame image paths."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print("Creating final video...")
        # For Windows compatibility, use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create video writer for {output_path}")
        
        for frame_path in tqdm(frame_paths, desc="Creating video"):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to {output_path}")


def process_video_parallel(input_path, output_path, width=120, height=60, 
                          processes=None, batch_size=10, font_size=12, fps=30, temp_dir="./temp"):
    """
    Process a video using pipeline parallelism.
    
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
    processor = ParallelProcessor(
        num_processes=processes,
        batch_size=batch_size,
        font_size=font_size,
        fps=fps
    )
    
    processor.process_video(
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        temp_dir=temp_dir
    )