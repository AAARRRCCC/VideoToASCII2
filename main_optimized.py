# Add debug prints at the very beginning
import sys
print("[DEBUG] Starting main_optimized.py")
sys.stdout.flush()

import argparse
import os
import traceback
from optimized_processor import process_video_optimized
from utils import check_ffmpeg_installed, create_directory_if_not_exists

print("[DEBUG] All modules imported successfully")
sys.stdout.flush()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert video to Japanese ASCII art using optimized parallelism')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to output video file')
    parser.add_argument('--width', type=int, default=120, help='Maximum width of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--height', type=int, default=60, help='Maximum height of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for output resolution (e.g., 0.5 for half size, 2.0 for double size, max 10.0)')
    parser.add_argument('--fps', type=int, default=None, help='Frames per second of output video (default: use input video\'s FPS)')
    parser.add_argument('--font-size', type=int, default=12, help='Font size for ASCII characters')
    parser.add_argument('--temp-dir', type=str, default='.\\temp', help='Directory for temporary files')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel processing (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of frames to process in each batch (default: 10)')
    parser.add_argument('--force', action='store_true', help='Force processing without confirmation for large scale factors')
    return parser.parse_args()

def main():
    print("[DEBUG] Entering main function")
    sys.stdout.flush()
    
    try:
        # Check for ffmpeg installation first
        print("[DEBUG] Checking for ffmpeg")
        sys.stdout.flush()
        if not check_ffmpeg_installed():
            print("Error: ffmpeg is not installed or not accessible in PATH.")
            print("Please install ffmpeg from https://ffmpeg.org/download.html")
            print("Make sure to add it to your PATH environment variable.")
            sys.exit(1)
        
        print("[DEBUG] Parsing arguments")
        sys.stdout.flush()
        args = parse_arguments()
    
        # Convert backslashes to forward slashes for consistent path handling
        args.input_path = args.input_path.replace('\\', '/')
        args.output_path = args.output_path.replace('\\', '/')
        args.temp_dir = args.temp_dir.replace('\\', '/')
        
        print(f"[DEBUG] Input path: {args.input_path}")
        print(f"[DEBUG] Output path: {args.output_path}")
        print(f"[DEBUG] Temp dir: {args.temp_dir}")
        print(f"[DEBUG] Processes: {args.processes}")
        print(f"[DEBUG] Batch size: {args.batch_size}")
        print(f"[DEBUG] Scale factor: {args.scale}")
        sys.stdout.flush()
        
        # Check if scale factor is large and prompt for confirmation if needed
        LARGE_SCALE_THRESHOLD = 5.0
        if args.scale > LARGE_SCALE_THRESHOLD and not args.force:
            print(f"Warning: You've specified a large scale factor of {args.scale}.")
            print("Large scale factors can cause high memory usage and slow processing.")
            print("Recommended scale factors are between 0.5 and 5.0.")
            
            confirm = input(f"Do you want to continue with scale factor {args.scale}? (y/n): ")
            if confirm.lower() != 'y':
                print("Conversion cancelled.")
                sys.exit(0)
            print("Continuing with large scale factor as confirmed by user.")
        
        # Create temporary directory if it doesn't exist
        print("[DEBUG] Creating temporary directory")
        sys.stdout.flush()
        create_directory_if_not_exists(args.temp_dir)
    
        try:
            # Process video using optimized parallelism
            print(f"Processing video with optimized parallelism: {args.input_path}")
            sys.stdout.flush()
            
            # Check if input file exists
            if not os.path.exists(args.input_path):
                print(f"[DEBUG] ERROR: Input file does not exist: {args.input_path}")
                sys.stdout.flush()
                sys.exit(1)
                
            print("[DEBUG] About to call process_video_optimized")
            sys.stdout.flush()
            
            # Add a timeout mechanism
            import threading
            import ctypes
            
            def process_with_timeout():
                import traceback  # Import traceback inside the function
                try:
                    process_video_optimized(
                        input_path=args.input_path,
                        output_path=args.output_path,
                        width=args.width,
                        height=args.height,
                        processes=args.processes,
                        batch_size=args.batch_size,
                        font_size=args.font_size,
                        fps=args.fps,
                        temp_dir=args.temp_dir,
                        scale=args.scale
                    )
                    print("[DEBUG] process_video_optimized completed successfully")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[DEBUG] Exception in process_video_optimized: {e}")
                    traceback.print_exc()
                    sys.stdout.flush()
            
            # Start processing in a separate thread
            processing_thread = threading.Thread(target=process_with_timeout)
            processing_thread.daemon = True
            processing_thread.start()
            
            # Wait for the thread to complete with a timeout
            processing_thread.join(timeout=120)  # 2 minute timeout
            
            # Check if the thread is still alive (meaning it timed out)
            if processing_thread.is_alive():
                print("[DEBUG] TIMEOUT: process_video_optimized did not complete within the timeout period")
                print("[DEBUG] Attempting to diagnose the issue...")
                sys.stdout.flush()
                
                # Try to get information about what's happening
                import traceback
                import inspect
                import gc
                
                # Print information about active threads
                print("[DEBUG] Active threads:")
                for thread in threading.enumerate():
                    print(f"[DEBUG] - Thread: {thread.name}, Daemon: {thread.daemon}")
                
                print("[DEBUG] Exiting due to timeout")
                sys.exit(1)
            
            print("Conversion complete!")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"Error: {str(e)}")
            print("[DEBUG] Exception traceback:")
            traceback.print_exc()
            sys.stdout.flush()
            sys.exit(1)
    except Exception as e:
        print(f"[DEBUG] Unexpected error in main function: {str(e)}")
        print("[DEBUG] Exception traceback:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
    
    finally:
        # Clean up temporary files
        if os.path.exists(args.temp_dir):
            import shutil
            try:
                shutil.rmtree(args.temp_dir)
            except PermissionError:
                print(f"Warning: Could not remove temporary directory: {args.temp_dir}")
                print("You may want to delete it manually.")

if __name__ == "__main__":
    main()