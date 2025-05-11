import argparse
import os
import sys
import time
from video_processor import VideoProcessor
from ascii_converter import ASCIIConverter
from renderer import Renderer
from utils import check_ffmpeg_installed, create_directory_if_not_exists

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert video to Japanese ASCII art')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to output video file')
    parser.add_argument('--width', type=int, default=120, help='Maximum width of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--height', type=int, default=60, help='Maximum height of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of output video')
    parser.add_argument('--font-size', type=int, default=12, help='Font size for ASCII characters')
    parser.add_argument('--temp-dir', type=str, default='.\\temp', help='Directory for temporary files')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use when mode is set to "parallel" (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of frames to process in each batch (default: 10)')
    parser.add_argument('--compare', action='store_true', help='Create a side-by-side comparison video of the original and ASCII versions')
    parser.add_argument('--mode', type=str, choices=['sequential', 'parallel'], default='parallel', help='Processing mode: "sequential" or "parallel" (default: "parallel")')
    parser.add_argument('--scale', type=int, default=1, help='Scaling factor for ASCII render resolution (default: 1)')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
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
    
    # If the output directory doesn't exist, create it
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    create_directory_if_not_exists(output_dir)
    
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # Initialize components based on processing mode
        if args.mode == 'sequential':
            print("Using sequential processing mode.")
            # In sequential mode, num_processes is effectively 1
            video_processor = VideoProcessor(num_processes=1)
            ascii_converter = ASCIIConverter(num_processes=1)
            renderer = Renderer(font_size=args.font_size, fps=args.fps, num_processes=1)
        else: # Default to parallel
            print("Using parallel processing mode.")
            video_processor = VideoProcessor(num_processes=args.processes)
            ascii_converter = ASCIIConverter(num_processes=args.processes)
            renderer = Renderer(font_size=args.font_size, fps=args.fps, num_processes=args.processes)
        
        # Process video
        print(f"Processing video: {args.input_path}")
        start_time = time.time()
        downscaled_video, actual_dimensions = video_processor.downscale_video(
            args.input_path,
            os.path.join(args.temp_dir, "downscaled.mp4"),
            args.width,
            args.height
        )
        end_time = time.time()
        print(f"Downscaling video took: {end_time - start_time:.2f} seconds")

        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            print("\n--- Downscaling Profiling Results ---")
            stats.print_stats()
            print("-------------------------------------")
            profiler.enable()
        
        # Convert to ASCII frames
        print("Converting to ASCII art...")
        start_time = time.time()
        ascii_frames, ascii_dimensions = ascii_converter.convert_video_to_ascii(
            downscaled_video,
            args.width * args.scale,
            args.height * args.scale,
            batch_size=args.batch_size
        )
        end_time = time.time()
        print(f"Converting to ASCII art took: {end_time - start_time:.2f} seconds")

        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            print("\n--- ASCII Conversion Profiling Results ---")
            stats.print_stats()
            print("------------------------------------------")
            profiler.enable()
        
        # Render and save output
        print(f"Rendering output to: {args.output_path}")
        start_time = time.time()
        renderer.render_ascii_frames(
            ascii_frames,
            args.output_path,
            ascii_dimensions, # ascii_dimensions already reflects the scaled size
            batch_size=args.batch_size
        )
        end_time = time.time()
        print(f"Rendering output took: {end_time - start_time:.2f} seconds")

        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            print("\n--- Rendering Profiling Results ---")
            stats.print_stats()
            print("-----------------------------------")
            profiler.enable()
        
        # If compare flag is set, create side-by-side comparison video
        if args.compare:
            print("Creating side-by-side comparison video...")
            start_time = time.time()
            comparison_path = os.path.splitext(args.output_path)[0] + "_comparison.mp4"
            video_processor.create_comparison_video(
                args.input_path,
                args.output_path,
                comparison_path,
                args.scale
            )
            end_time = time.time()
            print(f"Creating comparison video took: {end_time - start_time:.2f} seconds")

            if args.profile:
                profiler.disable()
                stats = pstats.Stats(profiler).sort_stats('cumulative')
                print("\n--- Comparison Video Profiling Results ---")
                stats.print_stats()
                print("------------------------------------------")
                profiler.enable()
        
        print("Conversion complete!")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) # Ensure exit on error even with profiling

    finally:
        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            print("\n--- Profiling Results ---")
            stats.print_stats()
            print("-------------------------")

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