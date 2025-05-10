import os
import time
import multiprocessing as mp
import queue
import threading
import tempfile
import subprocess
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import psutil
import mmap
from contextlib import contextmanager

from video_processor import VideoProcessor
from ascii_converter import ASCIIConverter
from renderer import Renderer, render_ascii_frame_to_image_static
from utils import create_directory_if_not_exists

class EnhancedParallelProcessor:
    """
    An enhanced parallel processor that addresses multiple bottlenecks:
    1. More efficient process allocation strategy
    2. Reduced inter-process communication overhead
    3. Better handling of I/O bottlenecks
    4. Optimized memory usage patterns
    5. Minimized process creation overhead
    """
    
    def __init__(self, num_processes=None, batch_size=10, font_size=12, fps=30):
        """
        Initialize the enhanced parallel processor.
        
        Args:
            num_processes (int): Number of processes to use for parallel processing
            batch_size (int): Number of frames to process in each batch
            font_size (int): Font size for ASCII characters
            fps (int): Frames per second of output video
        """
        # Determine optimal number of processes based on system resources
        self.system_cores = os.cpu_count()
        
        # If num_processes is not specified, use an optimal allocation strategy
        if num_processes is None:
            # Use 75% of available cores for better resource utilization
            # This avoids oversubscription while still using most available cores
            self.num_processes = max(2, int(self.system_cores * 0.75))
        else:
            self.num_processes = num_processes
            
        # Dynamically adjust batch size based on available memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        memory_per_process = available_memory / self.num_processes
        
        # If user specified a batch size, use it as a minimum
        self.batch_size = max(batch_size, min(50, int(memory_per_process / 10)))
        
        self.font_size = font_size
        self.fps = fps
        
        # Initialize components
        self.video_processor = VideoProcessor(num_processes=self.num_processes)
        self.ascii_converter = ASCIIConverter(num_processes=self.num_processes)
        self.renderer = Renderer(font_size=self.font_size, fps=self.fps, num_processes=self.num_processes)
        
        # Create a manager for shared resources
        self.manager = mp.Manager()
        
        # Allocate process pools based on task type
        # More processes for CPU-bound tasks, fewer for I/O-bound tasks
        self.io_processes = max(2, self.num_processes // 4)
        self.cpu_processes = self.num_processes - self.io_processes
        
        print(f"Enhanced processor configuration:")
        print(f"- Total processes: {self.num_processes} (out of {self.system_cores} cores)")
        print(f"- I/O processes: {self.io_processes}")
        print(f"- CPU processes: {self.cpu_processes}")
        print(f"- Batch size: {self.batch_size}")
    
    def process_video(self, input_path, output_path, width=120, height=60, temp_dir=None):
        """
        Process a video using enhanced parallelism.
        
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
            # Step 1: Downscale video using ffmpeg (this is still sequential but optimized)
            print(f"Downscaling video: {input_path}")
            downscaled_video, actual_dimensions = self.video_processor.downscale_video(
                input_path,
                os.path.join(temp_dir, "downscaled.mp4"),
                width,
                height
            )
            
            # Step 2: Extract video information
            cap = cv2.VideoCapture(downscaled_video)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {downscaled_video}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Calculate output image dimensions
            char_width = self.font_size
            char_height = self.font_size
            img_width = width * char_width
            img_height = height * char_height
            
            # Step 3: Create shared data structures with optimized queue sizes
            # Use larger queue sizes to reduce blocking
            frame_queue = self.manager.Queue(maxsize=self.batch_size * 10)  # Increased queue size
            result_queue = self.manager.Queue(maxsize=self.batch_size * 10)  # Increased queue size
            
            # Create events for signaling
            extraction_done = self.manager.Event()
            processing_done = self.manager.Event()
            
            # Create a shared counter for active workers
            active_workers = self.manager.Value('i', 0)
            
            # Step 4: Start the enhanced pipeline
            print("Starting enhanced processing pipeline...")
            
            # Use thread pool for I/O-bound tasks
            with ThreadPoolExecutor(max_workers=self.io_processes) as io_executor:
                # Start frame extraction in a thread
                extraction_future = io_executor.submit(
                    self._extract_frames_worker,
                    downscaled_video,
                    total_frames,
                    frame_queue,
                    extraction_done,
                    batch_size=self.batch_size
                )
                
                # Use process pool for CPU-bound tasks with work stealing
                with ProcessPoolExecutor(max_workers=self.cpu_processes) as process_executor:
                    # Submit batch processing tasks
                    processing_futures = []
                    
                    # Create a partial function with fixed parameters
                    process_func = partial(
                        self._process_frame_batch,
                        width=width,
                        height=height,
                        img_width=img_width,
                        img_height=img_height,
                        font_size=self.font_size,
                        frames_dir=frames_dir
                    )
                    
                    # Submit initial batch processing tasks
                    num_workers = self.cpu_processes
                    print(f"Starting {num_workers} worker processes")
                    
                    # Set initial active worker count
                    with active_workers.get_lock():
                        active_workers.value = num_workers
                    
                    for _ in range(num_workers):
                        future = process_executor.submit(
                            self._process_frames_worker,
                            frame_queue,
                            result_queue,
                            extraction_done,
                            processing_done,
                            process_func,
                            self.batch_size,
                            active_workers
                        )
                        processing_futures.append(future)
                    
                    # Monitor progress and collect rendered frame paths
                    frame_paths = [None] * total_frames
                    with tqdm(total=total_frames, desc="Processing frames") as pbar:
                        completed = 0
                        stall_counter = 0
                        last_completed = 0
                        timeout_counter = 0
                        max_timeout = 300  # 30 seconds with 0.1s sleep
                        
                        print(f"Starting progress monitoring loop, expecting {total_frames} frames")
                        while completed < total_frames:
                            try:
                                # Non-blocking get
                                frame_idx, frame_path = result_queue.get(timeout=0.1)
                                frame_paths[frame_idx] = frame_path
                                completed += 1
                                pbar.update(1)
                                stall_counter = 0  # Reset stall counter when we make progress
                                timeout_counter = 0  # Reset timeout counter
                            except queue.Empty:
                                # Check if all processes are done
                                if extraction_done.is_set() and processing_done.is_set() and result_queue.empty():
                                    # Double-check if we have all frames
                                    missing_frames = frame_paths.count(None)
                                    print(f"All processes done, missing frames: {missing_frames}")
                                    if None not in frame_paths:
                                        print(f"All frames received, breaking loop")
                                        break
                                    else:
                                        # If we've waited long enough, proceed anyway
                                        timeout_counter += 1
                                        if timeout_counter >= max_timeout:
                                            print(f"Timeout reached, proceeding with {missing_frames} missing frames")
                                            break
                                
                                # Check for stalls
                                if completed == last_completed:
                                    stall_counter += 1
                                    if stall_counter % 100 == 0:  # Log every ~10 seconds
                                        print(f"Possible stall detected: No progress for {stall_counter} iterations")
                                        print(f"Extraction done: {extraction_done.is_set()}, Processing done: {processing_done.is_set()}")
                                        print(f"Active workers: {active_workers.value}")
                                        
                                        # Force break if stalled for too long (30 seconds)
                                        if stall_counter >= 3000:
                                            print(f"Stall timeout reached, breaking loop")
                                            break
                                else:
                                    last_completed = completed
                                    stall_counter = 0
                                
                                time.sleep(0.01)
                    
                    # Wait for all futures to complete
                    for future in processing_futures:
                        future.result()
                    
                    # Create video from rendered frames using a more efficient approach
                    frame_paths = [path for path in frame_paths if path is not None]
                    video_future = io_executor.submit(
                        self._create_video_from_frames_efficient,
                        frame_paths,
                        output_path,
                        img_width,
                        img_height
                    )
                    
                    # Wait for video creation to complete
                    video_future.result()
            
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
    
    def _extract_frames_worker(self, video_path, total_frames, frame_queue, extraction_done, batch_size=10):
        """
        Worker thread for extracting frames from video with optimized I/O.
        Uses batched reading to reduce I/O overhead.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            
            # Read frames in batches to reduce I/O overhead
            batch = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch.append((i, frame))
                
                # When batch is full, put it in the queue
                if len(batch) >= batch_size:
                    frame_queue.put(batch)
                    batch = []
            
            # Put any remaining frames
            if batch:
                frame_queue.put(batch)
            
            cap.release()
        except Exception as e:
            print(f"Error in frame extraction: {e}")
        finally:
            # Signal that extraction is done
            extraction_done.set()
    
    def _process_frames_worker(self, frame_queue, result_queue, extraction_done, processing_done, process_func, batch_size, active_workers):
        """
        Worker process that processes batches of frames.
        Uses work stealing to ensure all processes are kept busy.
        """
        worker_id = mp.current_process().name
        print(f"Worker {worker_id} started")
        frames_processed = 0
        
        try:
            while True:
                # First check if we should exit
                if extraction_done.is_set() and frame_queue.empty():
                    print(f"Worker {worker_id}: Extraction done and queue empty, breaking loop")
                    break
                
                try:
                    # Get batch from queue with timeout
                    batch = frame_queue.get(timeout=0.5)  # Increased timeout
                    if not batch:
                        print(f"Worker {worker_id}: Received empty batch, skipping")
                        continue
                    
                    # Process each frame in the batch
                    for frame_data in batch:
                        frame_idx, frame = frame_data
                        frame_path = process_func(frame_idx, frame)
                        result_queue.put((frame_idx, frame_path))
                        frames_processed += 1
                    
                    print(f"Worker {worker_id}: Processed batch, total frames: {frames_processed}")
                    
                except queue.Empty:
                    # Queue is empty but extraction might not be done
                    if extraction_done.is_set():
                        print(f"Worker {worker_id}: Queue empty and extraction done, will check again")
                    continue
        except Exception as e:
            print(f"Error in frame processing: {e}")
        finally:
            # Decrement active worker count
            with active_workers.get_lock():
                active_workers.value -= 1
                remaining = active_workers.value
                print(f"Worker {worker_id}: Exiting, {remaining} workers still active")
                
                # Last worker sets the processing_done event
                if remaining == 0 and extraction_done.is_set():
                    print(f"Worker {worker_id}: Last worker setting processing_done event")
                    processing_done.set()
    
    def _process_frame_batch(self, frame_idx, frame, width, height, img_width, img_height, font_size, frames_dir):
        """
        Process a single frame: convert to ASCII and render to image.
        This function combines conversion and rendering to reduce inter-process communication.
        """
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
    
    def _create_video_from_frames_efficient(self, frame_paths, output_path, width, height):
        """
        Create a video from a list of frame image paths using ffmpeg directly with optimized settings.
        Uses hardware acceleration if available and optimized encoding parameters.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print("Creating final video...")
        
        # Create a temporary file listing all frames
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            frames_list_path = f.name
            for frame_path in frame_paths:
                f.write(f"file '{os.path.abspath(frame_path)}'\n")
        
        try:
            # Check for hardware acceleration support
            hw_accel = ""
            try:
                # Check for NVIDIA GPU support
                nvidia_check = subprocess.run(
                    "nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if nvidia_check.returncode == 0:
                    hw_accel = "-hwaccel cuda -hwaccel_output_format cuda"
            except:
                pass
            
            # Use ffmpeg with optimized settings
            # -threads 0: Use all available CPU threads
            # -preset ultrafast: Fastest encoding speed
            # -tune zerolatency: Optimize for low latency
            cmd = (
                f'ffmpeg {hw_accel} -y -r {self.fps} -f concat -safe 0 -i "{frames_list_path}" '
                f'-c:v libx264 -pix_fmt yuv420p -crf 23 -preset ultrafast -tune zerolatency '
                f'-threads 0 "{output_path}"'
            )
            
            # Execute ffmpeg command
            subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Video saved to {output_path}")
        finally:
            # Remove temporary file
            os.unlink(frames_list_path)


def process_video_enhanced(input_path, output_path, width=120, height=60, 
                          processes=None, batch_size=10, font_size=12, fps=30, temp_dir="./temp"):
    """
    Process a video using enhanced parallelism.
    
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
    processor = EnhancedParallelProcessor(
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