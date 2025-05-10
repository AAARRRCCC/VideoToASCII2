import os
import time
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm
import tempfile
import subprocess
import sys
import traceback

from video_processor import VideoProcessor
from ascii_converter import ASCIIConverter
from renderer import Renderer, render_ascii_frame_to_image_static
from utils import create_directory_if_not_exists

# Define a standalone worker function that can be pickled
def worker_process_func(worker_id, width, height, img_width, img_height, font_size,
                       frames_dir, frame_queue, render_queue, extraction_done, conversion_done, active_workers):
    import os
    import sys
    import time
    import queue
    import traceback
    
    # Import these inside the function to ensure they're available in the worker process
    from ascii_converter import ASCIIConverter
    from renderer import render_ascii_frame_to_image_static
    
    print(f"[DEBUG] Worker {worker_id} started with PID: {os.getpid()}")
    sys.stdout.flush()
    
    # Create a local ASCII converter for this process
    try:
        ascii_converter = ASCIIConverter()
        print(f"[DEBUG] Worker {worker_id}: Created local ASCIIConverter")
        sys.stdout.flush()
    except Exception as e:
        print(f"[DEBUG] Worker {worker_id}: Error creating ASCIIConverter: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return
    
    frames_processed = 0
    
    try:
        while True:
            # Check if we should exit
            if extraction_done.is_set() and frame_queue.empty():
                print(f"[DEBUG] Worker {worker_id}: Extraction done and queue empty, exiting")
                sys.stdout.flush()
                break
            
            try:
                # Get frame from queue with timeout
                frame_data = frame_queue.get(timeout=2.0)
                if frame_data is None:
                    print(f"[DEBUG] Worker {worker_id}: Received None frame, exiting")
                    sys.stdout.flush()
                    break
                
                frame_idx, frame = frame_data
                print(f"[DEBUG] Worker {worker_id}: Processing frame {frame_idx}")
                sys.stdout.flush()
                
                # Convert frame to ASCII
                ascii_frame = ascii_converter.convert_frame_to_ascii(frame, width, height)
                
                # Render ASCII frame to image
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                render_ascii_frame_to_image_static(ascii_frame, frame_path, img_width, img_height, font_size)
                
                # Put rendered frame path in queue
                try:
                    render_queue.put((frame_idx, frame_path), timeout=5)
                except queue.Full:
                    print(f"[DEBUG] Worker {worker_id}: Render queue full, skipping frame {frame_idx}")
                    sys.stdout.flush()
                
                frames_processed += 1
                print(f"[DEBUG] Worker {worker_id}: Completed frame {frame_idx}, total: {frames_processed}")
                sys.stdout.flush()
                
            except queue.Empty:
                # Just continue if the queue is empty
                continue
            except Exception as e:
                print(f"[DEBUG] Worker {worker_id}: Error processing frame: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                continue
    
    except Exception as e:
        print(f"[DEBUG] Worker {worker_id}: Unexpected error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
    
    finally:
        # Decrement active worker count
        active_workers.value -= 1
        remaining = active_workers.value
        print(f"[DEBUG] Worker {worker_id}: Exiting, {remaining} workers still active")
        sys.stdout.flush()
        
        # Last worker sets the conversion_done event
        if remaining == 0 and extraction_done.is_set():
            print(f"[DEBUG] Worker {worker_id}: Last worker setting conversion_done event")
            sys.stdout.flush()
            conversion_done.set()

class OptimizedProcessor:
    """
    An optimized processor that combines data parallelism with pipeline parallelism
    and uses shared memory for better performance.
    """
    
    def __init__(self, num_processes=None, batch_size=10, font_size=12, fps=None, scale=1.0):
        """
        Initialize the optimized processor.
        
        Args:
            num_processes (int): Number of processes to use for parallel processing
            batch_size (int): Number of frames to process in each batch
            font_size (int): Font size for ASCII characters
            fps (int): Frames per second of output video (if None, will use input video's FPS)
            scale (float): Scale factor for output resolution (e.g., 0.5 for half size, 2.0 for double size)
                           Values above 10.0 will be capped to prevent performance issues.
        """
        import sys
        print("[DEBUG] Initializing OptimizedProcessor")
        sys.stdout.flush()
        
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.batch_size = batch_size
        self.font_size = font_size
        self.fps = fps
        self.scale = scale  # This will be validated by VideoProcessor
        
        print(f"[DEBUG] Parameters: num_processes={self.num_processes}, batch_size={self.batch_size}, scale={self.scale}")
        sys.stdout.flush()
        
        # Initialize components
        print("[DEBUG] Initializing VideoProcessor")
        sys.stdout.flush()
        self.video_processor = VideoProcessor(num_processes=self.num_processes, scale=self.scale)
        
        print("[DEBUG] Initializing ASCIIConverter")
        sys.stdout.flush()
        self.ascii_converter = ASCIIConverter(num_processes=self.num_processes)
        
        print("[DEBUG] Initializing Renderer")
        sys.stdout.flush()
        self.renderer = Renderer(font_size=self.font_size, fps=self.fps, num_processes=self.num_processes)
        
        # Create a manager for shared resources
        print("[DEBUG] Creating multiprocessing Manager")
        sys.stdout.flush()
        self.manager = mp.Manager()
        
        print(f"[DEBUG] OptimizedProcessor initialized with {self.num_processes} processes and batch size {self.batch_size}")
        sys.stdout.flush()
        
    def process_video(self, input_path, output_path, width=120, height=60, temp_dir=None):
        """
        Process a video using optimized parallelism.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            temp_dir (str): Directory for temporary files
        """
        import sys
        print("[DEBUG] Entering process_video method")
        sys.stdout.flush()
        
        start_time = time.time()
        
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            create_directory_if_not_exists(temp_dir)
            
        frames_dir = os.path.join(temp_dir, "frames")
        create_directory_if_not_exists(frames_dir)
        
        print("[DEBUG] Temporary directories created")
        sys.stdout.flush()
        
        try:
            # Step 1: Downscale video using ffmpeg (this is still sequential)
            print(f"Downscaling video: {input_path}")
            print(f"[DEBUG] Starting video processing with temp_dir: {temp_dir}")
            sys.stdout.flush()
            
            print("[DEBUG] About to call downscale_video")
            sys.stdout.flush()
            
            downscaled_video, actual_dimensions = self.video_processor.downscale_video(
                input_path,
                os.path.join(temp_dir, "downscaled.mp4"),
                width,
                height
            )
            
            print("[DEBUG] Video downscaling completed")
            sys.stdout.flush()
            
            # Step 2: Extract video information
            print("[DEBUG] About to extract video information")
            sys.stdout.flush()
            
            cap = cv2.VideoCapture(downscaled_video)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {downscaled_video}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[DEBUG] Video has {total_frames} frames at {input_fps} fps")
            sys.stdout.flush()
            cap.release()
            
            # Use input video's FPS if not specified
            if self.fps is None:
                self.fps = input_fps
                print(f"[DEBUG] Using input video's FPS: {self.fps}")
            else:
                print(f"[DEBUG] Using specified FPS: {self.fps}")
            sys.stdout.flush()
            
            print("[DEBUG] Video information extracted successfully")
            sys.stdout.flush()
            
            # Calculate output image dimensions
            char_width = self.font_size
            char_height = self.font_size
            img_width = int(width * char_width * self.scale)
            img_height = int(height * char_height * self.scale)
            
            print(f"[DEBUG] Output image dimensions with scale {self.scale}: {img_width}x{img_height}")
            sys.stdout.flush()
            
            # Step 3: Create shared data structures
            # Increase queue sizes significantly to prevent blocking
            frame_queue = self.manager.Queue(maxsize=self.batch_size * 50)
            ascii_queue = self.manager.Queue(maxsize=self.batch_size * 50)
            render_queue = self.manager.Queue(maxsize=self.batch_size * 50)
            
            print(f"[DEBUG] Created queues with size {self.batch_size * 50}")
            
            # Create events for signaling
            extraction_done = self.manager.Event()
            conversion_done = self.manager.Event()
            
            # Create a shared counter for active workers
            active_workers = self.manager.Value('i', 0)
            
            # Step 4: Start the pipeline
            print("Starting optimized processing pipeline...")
            print(f"[DEBUG] Starting extraction thread")
            sys.stdout.flush()
            
            # Use a thread for frame extraction
            print("[DEBUG] Creating extraction thread")
            sys.stdout.flush()
            
            extraction_thread = threading.Thread(
                target=self._extract_frames_worker,
                args=(downscaled_video, total_frames, frame_queue, extraction_done)
            )
            extraction_thread.daemon = True
            extraction_thread.start()
            
            print("[DEBUG] Extraction thread started")
            sys.stdout.flush()
            
            # Use direct multiprocessing for worker processes
            print(f"[DEBUG] Creating {self.num_processes} worker processes")
            sys.stdout.flush()
            
            # Create worker processes
            worker_processes = []
            num_workers = max(1, self.num_processes // 2)
            print(f"Starting {num_workers} worker processes")
            sys.stdout.flush()
            
            # Set initial active worker count
            active_workers.value = num_workers
            
            # Use our simplified worker function from debug_optimized.py
            from ascii_converter import ASCIIConverter
            from renderer import render_ascii_frame_to_image_static
            
            # Start worker processes
            for i in range(num_workers):
                p = mp.Process(
                    target=worker_process_func,  # Use the module-level function
                    args=(
                        i+1, width, height, img_width, img_height, self.font_size,
                        frames_dir, frame_queue, render_queue, extraction_done, conversion_done, active_workers
                    )
                )
                p.daemon = True
                p.start()
                worker_processes.append(p)
                print(f"[DEBUG] Started worker {i+1} with PID: {p.pid}")
                sys.stdout.flush()
            
            # Monitor progress and collect rendered frame paths
            frame_paths = [None] * total_frames
            print(f"[DEBUG] Starting progress monitoring loop")
            sys.stdout.flush()
            
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                completed = 0
                stall_counter = 0
                last_completed = 0
                timeout_counter = 0
                max_timeout = 300  # 30 seconds with 0.1s sleep
                
                print(f"[DEBUG] Starting progress monitoring loop, expecting {total_frames} frames")
                sys.stdout.flush()
                
                while completed < total_frames:
                    try:
                        # Non-blocking get with longer timeout
                        frame_idx, frame_path = render_queue.get(timeout=0.5)
                        frame_paths[frame_idx] = frame_path
                        completed += 1
                        pbar.update(1)
                        
                        # Reduce logging frequency
                        if completed % 5 == 0 or completed == total_frames:
                            print(f"[DEBUG] Received frame {frame_idx}, completed: {completed}/{total_frames}")
                            sys.stdout.flush()
                        
                        stall_counter = 0  # Reset stall counter when we make progress
                        timeout_counter = 0  # Reset timeout counter
                    except queue.Empty:
                        # Check if all processes are done
                        if extraction_done.is_set() and conversion_done.is_set() and render_queue.empty():
                            # Double-check if we have all frames
                            missing_frames = frame_paths.count(None)
                            print(f"[DEBUG] All processes done, missing frames: {missing_frames}")
                            sys.stdout.flush()
                            
                            # Wait a bit longer to ensure all frames are processed
                            time.sleep(1.0)
                            
                            if None not in frame_paths:
                                print(f"[DEBUG] All frames received, breaking loop")
                                sys.stdout.flush()
                                break
                            else:
                                # Print the indices of missing frames
                                missing_indices = [i for i, path in enumerate(frame_paths) if path is None]
                                print(f"[DEBUG] Missing frames at indices: {missing_indices[:10]}...")
                                sys.stdout.flush()
                                
                                # If we've waited long enough, proceed anyway
                                timeout_counter += 1
                                if timeout_counter >= max_timeout:
                                    print(f"[DEBUG] Timeout reached, proceeding with {missing_frames} missing frames")
                                    sys.stdout.flush()
                                    break
                        
                        # Check for stalls
                        if completed == last_completed:
                            stall_counter += 1
                            # Log less frequently to reduce output
                            if stall_counter % 200 == 0:  # Log every ~20 seconds
                                print(f"[DEBUG] Possible stall detected: No progress for {stall_counter} iterations")
                                print(f"[DEBUG] Extraction done: {extraction_done.is_set()}, Conversion done: {conversion_done.is_set()}")
                                print(f"[DEBUG] Active workers: {active_workers.value}")
                                print(f"[DEBUG] Frame queue size: ~{frame_queue.qsize()}, Render queue size: ~{render_queue.qsize()}")
                                sys.stdout.flush()
                                
                                # Force break if stalled for too long (reduced from 30 to 15 seconds)
                                if stall_counter >= 1500:
                                    print(f"[DEBUG] Stall timeout reached, breaking loop")
                                    sys.stdout.flush()
                                    break
                        else:
                            last_completed = completed
                            stall_counter = 0
                        
                        # Longer sleep to reduce CPU usage
                        time.sleep(0.1)
            
            # Wait for extraction thread to complete
            extraction_thread.join(timeout=5)
            if extraction_thread.is_alive():
                print("[DEBUG] Extraction thread did not exit cleanly")
                sys.stdout.flush()
            
            # Wait for worker processes to complete
            for i, p in enumerate(worker_processes):
                p.join(timeout=5)
                if p.is_alive():
                    print(f"[DEBUG] Worker {i+1} did not exit cleanly, terminating")
                    sys.stdout.flush()
                    p.terminate()
            
            # Create video from rendered frames
            frame_paths = [path for path in frame_paths if path is not None]
            
            # Use a thread for video creation
            video_future = threading.Thread(
                target=self._create_video_from_frames,
                args=(
                    frame_paths,
                    output_path,
                    img_width,
                    img_height
                )
            )
            video_future.start()
            video_future.join()
            
            print("[DEBUG] Video creation completed")
            sys.stdout.flush()
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
    
    def _extract_frames_worker(self, video_path, total_frames, frame_queue, extraction_done):
        """Worker thread for extracting frames from video."""
        try:
            print(f"[DEBUG] Starting frame extraction, total frames: {total_frames}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            
            # Only log every 10th frame to reduce output
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"[DEBUG] Failed to read frame {i}, breaking extraction loop")
                    break
                
                # Reduce logging frequency
                if i % 10 == 0 or i == total_frames - 1:
                    print(f"[DEBUG] Extracted frame {i}/{total_frames}")
                
                # Add a small sleep to prevent overwhelming the queue
                if frame_queue.qsize() > self.batch_size * 40:
                    time.sleep(0.01)
                
                # Instead of skipping frames, wait until there's space in the queue
                while True:
                    try:
                        # Use a timeout to prevent blocking indefinitely
                        frame_queue.put((i, frame), timeout=5)
                        break  # Successfully added to queue, exit the loop
                    except queue.Full:
                        print(f"[DEBUG] Frame queue full, waiting to add frame {i}")
                        # Sleep briefly before retrying
                        time.sleep(0.5)
            
            cap.release()
            print(f"[DEBUG] Frame extraction completed, processed {i+1}/{total_frames} frames")
        except Exception as e:
            print(f"Error in frame extraction: {e}")
        finally:
            # Signal that extraction is done
            extraction_done.set()
            print(f"[DEBUG] Extraction done event set")
    
    def _convert_and_render_worker(self, width, height, img_width, img_height, font_size,
                                   frames_dir, frame_queue, render_queue, extraction_done, conversion_done, active_workers):
        """Worker process that combines ASCII conversion and rendering for better efficiency."""
        worker_id = mp.current_process().name
        print(f"[DEBUG] Worker {worker_id} started")
        frames_processed = 0
        last_log_time = time.time()
        
        try:
            while True:
                # First check if we should exit
                if extraction_done.is_set() and frame_queue.empty():
                    print(f"[DEBUG] Worker {worker_id}: Extraction done and queue empty, breaking loop")
                    # Double-check after a short wait to avoid race conditions
                    time.sleep(0.5)
                    if frame_queue.empty():
                        break
                
                try:
                    # Get frame from queue with longer timeout
                    frame_data = frame_queue.get(timeout=2.0)  # Increased timeout
                    if frame_data is None:
                        print(f"[DEBUG] Worker {worker_id}: Received None frame, skipping")
                        continue
                    
                    frame_idx, frame = frame_data
                    
                    # Reduce logging frequency - only log every 5 frames or every 5 seconds
                    current_time = time.time()
                    if frames_processed % 5 == 0 or current_time - last_log_time > 5:
                        print(f"[DEBUG] Worker {worker_id}: Processing frame {frame_idx}")
                        last_log_time = current_time
                    
                    # Convert frame to ASCII
                    ascii_frame = self.ascii_converter.convert_frame_to_ascii(frame, width, height)
                    
                    # Render ASCII frame to image
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                    render_ascii_frame_to_image_static(ascii_frame, frame_path, img_width, img_height, font_size)
                    
                    # Put rendered frame path in queue with timeout to prevent blocking
                    try:
                        render_queue.put((frame_idx, frame_path), timeout=5)
                    except queue.Full:
                        print(f"[DEBUG] Worker {worker_id}: Render queue full, retrying for frame {frame_idx}")
                        # Try again with a longer timeout
                        try:
                            render_queue.put((frame_idx, frame_path), timeout=30)
                        except queue.Full:
                            print(f"[DEBUG] Worker {worker_id}: Render queue still full after retry, skipping frame {frame_idx}")
                    
                    frames_processed += 1
                    
                    # Only log completion periodically
                    if frames_processed % 5 == 0:
                        print(f"[DEBUG] Worker {worker_id}: Completed frame {frame_idx}, total processed: {frames_processed}")
                    
                except queue.Empty:
                    # Queue is empty but extraction might not be done
                    if extraction_done.is_set():
                        # Only log occasionally to reduce output
                        if time.time() - last_log_time > 5:
                            print(f"[DEBUG] Worker {worker_id}: Queue empty and extraction done, will check again")
                            last_log_time = time.time()
                    # Add a small sleep to prevent CPU spinning
                    time.sleep(0.1)
                    continue
        except Exception as e:
            print(f"Error in conversion and rendering: {e}")
        finally:
            # Decrement active worker count
            with active_workers.get_lock():
                active_workers.value -= 1
                remaining = active_workers.value
                print(f"[DEBUG] Worker {worker_id}: Exiting, {remaining} workers still active")
                
                # Last worker sets the conversion_done event
                if remaining == 0 and extraction_done.is_set():
                    print(f"[DEBUG] Worker {worker_id}: Last worker setting conversion_done event")
                    conversion_done.set()
    
    def _create_video_from_frames(self, frame_paths, output_path, width, height):
        """Create a video from a list of frame image paths using ffmpeg directly for better performance."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print("Creating final video...")
        
        # Create a temporary file listing all frames
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            frames_list_path = f.name
            for frame_path in frame_paths:
                f.write(f"file '{os.path.abspath(frame_path)}'\n")
        
        try:
            # Use ffmpeg directly for better performance
            cmd = (
                f'ffmpeg -y -r {self.fps} -f concat -safe 0 -i "{frames_list_path}" '
                f'-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast '
                f'-filter:v "setpts=1.0*PTS" "{output_path}"'  # Ensure correct timing
            )
            
            print(f"[DEBUG] Creating video with FPS: {self.fps}")
            
            # Execute ffmpeg command
            subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Video saved to {output_path}")
        finally:
            # Remove temporary file
            os.unlink(frames_list_path)


def process_video_optimized(input_path, output_path, width=120, height=60,
                            processes=None, batch_size=10, font_size=12, fps=None, temp_dir="./temp", scale=1.0):
    """
    Process a video using optimized parallelism.
    
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
        scale (float): Scale factor for output resolution (e.g., 0.5 for half size, 2.0 for double size)
                       Values above 10.0 will be capped to prevent performance issues.
    """
    processor = OptimizedProcessor(
        num_processes=processes,
        batch_size=batch_size,
        font_size=font_size,
        fps=fps,
        scale=scale
    )
    
    processor.process_video(
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        temp_dir=temp_dir
    )