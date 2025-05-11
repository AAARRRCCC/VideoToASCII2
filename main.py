#!/usr/bin/env python3
"""
VideoToASCII - A tool for converting videos to ASCII art

This is the main entry point for the VideoToASCII converter.
It provides a unified interface to access all the different implementations.
"""

import argparse
import os
import sys
import time
import tkinter as tk

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import check_ffmpeg_installed, create_directory_if_not_exists
from src.processors.simple_processor import SimpleProcessor
from src.processors.parallel_processor import process_video_parallel
from src.processors.enhanced_parallel_processor import process_video_enhanced
from src.processors.optimized_processor import process_video_optimized
from src.gui.gui import VideoToASCIIGUI


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert video to ASCII art with various processing options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input and output paths
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, nargs='?', default=None, 
                        help='Path to output video file (if not provided, display in terminal)')
    
    # ASCII dimensions
    parser.add_argument('--width', '-w', type=int, default=120, 
                        help='Width of ASCII output in characters')
    parser.add_argument('--height', '-ht', type=int, default=60, 
                        help='Height of ASCII output in characters')
    
    # Processing options
    parser.add_argument('--fps', '-f', type=int, default=None, 
                        help='Frames per second of output video (defaults to input video FPS)')
    parser.add_argument('--font-size', '-fs', type=int, default=12, 
                        help='Font size for ASCII characters')
    parser.add_argument('--temp-dir', '-t', type=str, default='./temp', 
                        help='Directory for temporary files')
    parser.add_argument('--processes', '-p', type=int, default=None, 
                        help='Number of processes to use (default: number of CPU cores)')
    parser.add_argument('--batch-size', '-b', type=int, default=10, 
                        help='Number of frames to process in each batch')
    
    # Implementation selection
    parser.add_argument('--implementation', '-i', type=str, default='optimized', 
                        choices=['simple', 'parallel', 'enhanced', 'optimized'],
                        help='Processing implementation to use')
    
    # Scale factor
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for output resolution (e.g., 0.5 for half size, 2.0 for double size, max 10.0)')
    
    # Force option (skip confirmation for large scale factors)
    parser.add_argument('--force', action='store_true',
                        help='Force processing without confirmation for large scale factors')
    
    # GUI mode
    parser.add_argument('--gui', '-g', action='store_true', 
                        help='Launch the graphical user interface')
    
    # Terminal mode options
    parser.add_argument('--invert', '-inv', action='store_true', 
                        help='Invert brightness (only for terminal output)')
    parser.add_argument('--colored', '-c', action='store_true', 
                        help='Use colored ASCII (only for terminal output)')
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Launch GUI if requested
    if args.gui:
        root = tk.Tk()
        app = VideoToASCIIGUI(root)
        root.mainloop()
        return
    
    # Check for ffmpeg installation first
    if not check_ffmpeg_installed():
        print("Error: ffmpeg is not installed or not accessible in PATH.")
        print("Please install ffmpeg from https://ffmpeg.org/download.html")
        print("Make sure to add it to your PATH environment variable.")
        sys.exit(1)
    
    # Convert backslashes to forward slashes for consistent path handling
    args.input_path = args.input_path.replace('\\', '/')
    if args.output_path:
        args.output_path = args.output_path.replace('\\', '/')
    args.temp_dir = args.temp_dir.replace('\\', '/')
    
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
    if args.implementation != 'simple' or args.output_path:
        create_directory_if_not_exists(args.temp_dir)
    
    # Process video using the selected implementation
    start_time = time.time()
    
    try:
        if args.implementation == 'simple':
            # Simple implementation (good for terminal output)
            processor = SimpleProcessor()
            processor.process_video(
                args.input_path,
                args.output_path,
                args.width,
                args.height,
                args.fps,
                args.invert,
                args.colored
            )
        
        elif args.implementation == 'parallel':
            # Parallel implementation (pipeline parallelism)
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
        
        elif args.implementation == 'enhanced':
            # Enhanced parallel implementation
            process_video_enhanced(
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
        
        else:  # 'optimized' (default)
            # Optimized implementation
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
        
        # Print final stats
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()