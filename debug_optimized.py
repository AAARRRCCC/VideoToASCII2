import os
import time
import sys
import cv2
import multiprocessing as mp
import queue
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import traceback

def convert_frame_to_ascii(frame, width, height):
    """Simplified ASCII conversion for debugging"""
    # Resize frame to match target dimensions
    resized_frame = cv2.resize(frame, (width, height))
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Create ASCII frame using simple mapping
    ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    ascii_frame = []
    
    for y in range(height):
        row = []
        for x in range(width):
            pixel_value = gray_frame[y, x]
            # Map pixel value (0-255) to ASCII character
            char_idx = min(int(pixel_value / 25.5), 9)
            row.append(ascii_chars[char_idx])
        ascii_frame.append(row)
    
    return ascii_frame

def render_ascii_frame(ascii_frame, output_path, font_size=12):
    """Render ASCII frame to image"""
    width = len(ascii_frame[0])
    height = len(ascii_frame)
    img_width = width * font_size
    img_height = height * font_size
    
    # Create a black image
    image = Image.new("RGB", (img_width, img_height), color="black")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\consola.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw each character
    for y, row in enumerate(ascii_frame):
        for x, char in enumerate(row):
            draw.text(
                (x * font_size, y * font_size),
                char,
                font=font,
                fill="white"
            )
    
    # Save the image
    image.save(output_path)

def worker_process(worker_id, frame_queue, result_queue):
    """Worker process that processes frames"""
    print(f"[DEBUG] Worker {worker_id} started with PID: {os.getpid()}")
    sys.stdout.flush()
    
    frames_processed = 0
    
    try:
        while True:
            try:
                # Get frame from queue with timeout
                frame_data = frame_queue.get(timeout=5.0)
                if frame_data is None:
                    print(f"[DEBUG] Worker {worker_id}: Received None, exiting")
                    sys.stdout.flush()
                    break
                
                frame_idx, frame = frame_data
                print(f"[DEBUG] Worker {worker_id}: Processing frame {frame_idx}")
                sys.stdout.flush()
                
                # Convert frame to ASCII
                ascii_frame = convert_frame_to_ascii(frame, 80, 40)
                
                # Render ASCII frame to image
                frame_path = f"debug_output/frame_{frame_idx:06d}.png"
                render_ascii_frame(ascii_frame, frame_path)
                
                # Put result in result queue
                result_queue.put((frame_idx, frame_path))
                frames_processed += 1
                print(f"[DEBUG] Worker {worker_id}: Completed frame {frame_idx}, total: {frames_processed}")
                sys.stdout.flush()
                
            except queue.Empty:
                print(f"[DEBUG] Worker {worker_id}: Queue empty, waiting...")
                sys.stdout.flush()
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
        print(f"[DEBUG] Worker {worker_id}: Exiting after processing {frames_processed} frames")
        sys.stdout.flush()

def extract_frames(video_path, frame_queue, max_frames=100):
    """Extract frames from video and put them in the queue"""
    print(f"[DEBUG] Starting frame extraction from {video_path}")
    sys.stdout.flush()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_extract = min(total_frames, max_frames)
    
    print(f"[DEBUG] Video has {total_frames} frames at {fps} FPS, will extract {frames_to_extract}")
    sys.stdout.flush()
    
    for i in range(frames_to_extract):
        ret, frame = cap.read()
        if not ret:
            print(f"[DEBUG] Failed to read frame {i}")
            sys.stdout.flush()
            break
        
        if i % 10 == 0:
            print(f"[DEBUG] Extracted frame {i}/{frames_to_extract}")
            sys.stdout.flush()
        
        # Instead of skipping frames, wait until there's space in the queue
        while True:
            try:
                frame_queue.put((i, frame), timeout=5)
                break  # Successfully added to queue, exit the loop
            except queue.Full:
                print(f"[DEBUG] Frame queue full, waiting to add frame {i}")
                sys.stdout.flush()
                # Sleep briefly before retrying
                time.sleep(0.5)
    
    cap.release()
    print(f"[DEBUG] Frame extraction completed, extracted {frames_to_extract} frames")
    sys.stdout.flush()
    
    # Signal end of extraction by putting None in the queue for each worker
    return frames_to_extract

def main():
    # Create output directory
    os.makedirs("debug_output", exist_ok=True)
    
    # Create queues for communication
    frame_queue = mp.Queue(maxsize=50)
    result_queue = mp.Queue()
    
    # Start worker processes
    num_workers = 2
    workers = []
    
    print(f"[DEBUG] Starting {num_workers} worker processes")
    sys.stdout.flush()
    
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i+1, frame_queue, result_queue))
        p.daemon = True
        p.start()
        workers.append(p)
        print(f"[DEBUG] Started worker {i+1} with PID: {p.pid}")
        sys.stdout.flush()
    
    # Extract frames from video
    input_video = "input_short.mp4"
    max_frames = 50  # Limit to 50 frames for testing
    
    try:
        num_frames = extract_frames(input_video, frame_queue, max_frames)
        
        # Signal workers to exit
        for _ in range(num_workers):
            frame_queue.put(None)
        
        # Collect results
        results = []
        timeout_counter = 0
        
        print(f"[DEBUG] Waiting for {num_frames} results")
        sys.stdout.flush()
        
        while len(results) < num_frames and timeout_counter < 30:
            try:
                result = result_queue.get(timeout=1)
                results.append(result)
                if len(results) % 10 == 0:
                    print(f"[DEBUG] Received {len(results)}/{num_frames} results")
                    sys.stdout.flush()
            except queue.Empty:
                timeout_counter += 1
                print(f"[DEBUG] No results for {timeout_counter} seconds")
                sys.stdout.flush()
        
        print(f"[DEBUG] Received {len(results)}/{num_frames} results")
        sys.stdout.flush()
        
        # Wait for workers to finish
        for i, p in enumerate(workers):
            p.join(timeout=5)
            if p.is_alive():
                print(f"[DEBUG] Worker {i+1} did not exit cleanly, terminating")
                sys.stdout.flush()
                p.terminate()
        
        print("[DEBUG] All workers have exited")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[DEBUG] Error in main process: {e}")
        traceback.print_exc()
        sys.stdout.flush()
    
    print("[DEBUG] Processing completed")
    sys.stdout.flush()

if __name__ == "__main__":
    main()