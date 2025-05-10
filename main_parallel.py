import argparse
import os
import sys
from parallel_processor import process_video_parallel
from utils import check_ffmpeg_installed, create_directory_if_not_exists

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert video to Japanese ASCII art using pipeline parallelism')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to output video file')
    parser.add_argument('--width', type=int, default=120, help='Maximum width of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--height', type=int, default=60, help='Maximum height of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of output video')
    parser.add_argument('--font-size', type=int, default=12, help='Font size for ASCII characters')
    parser.add_argument('--temp-dir', type=str, default='.\\temp', help='Directory for temporary files')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel processing (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of frames to process in each batch (default: 10)')
    return parser.parse_args()

def main():
    # Check for ffmpeg installation first
    if not check_ffmpeg_installed():
        print("Error: ffmpeg is not installed or not accessible in PATH.")
        print("Please install ffmpeg from https://ffmpeg.org/download.html")
        print("Make sure to add it to your PATH environment variable.")
        sys.exit(1)
    
    args = parse_arguments()
    
    # Convert backslashes to forward slashes for consistent path handling
    args.input_path = args.input_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')
    args.temp_dir = args.temp_dir.replace('\\', '/')
    
    # Create temporary directory if it doesn't exist
    create_directory_if_not_exists(args.temp_dir)
    
    try:
        # Process video using pipeline parallelism
        print(f"Processing video with pipeline parallelism: {args.input_path}")
        process_video_parallel(
            input_path=args.input_path,
            output_path=args.output_path,
            width=args.width,
            height=args.height,
            processes=args.processes,
            batch_size=args.batch_size,
            font_size=args.font_size,
            fps=args.fps,
            temp_dir=args.temp_dir
        )
        
        print("Conversion complete!")
    
    except Exception as e:
        print(f"Error: {str(e)}")
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