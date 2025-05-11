import argparse
import os
import sys
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
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel processing (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of frames to process in each batch (default: 10)')
    parser.add_argument('--compare', action='store_true', help='Create a side-by-side comparison video of the original and ASCII versions')
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
    
    try:
        # Initialize components
        video_processor = VideoProcessor(num_processes=args.processes)
        ascii_converter = ASCIIConverter(num_processes=args.processes)
        renderer = Renderer(font_size=args.font_size, fps=args.fps, num_processes=args.processes)
        
        # Process video
        print(f"Processing video: {args.input_path}")
        downscaled_video, actual_dimensions = video_processor.downscale_video(
            args.input_path,
            os.path.join(args.temp_dir, "downscaled.mp4"),
            args.width,
            args.height
        )
        
        # Convert to ASCII frames
        print("Converting to ASCII art...")
        ascii_frames, ascii_dimensions = ascii_converter.convert_video_to_ascii(
            downscaled_video,
            args.width,
            args.height,
            batch_size=args.batch_size
        )
        
        # Render and save output
        print(f"Rendering output to: {args.output_path}")
        renderer.render_ascii_frames(
            ascii_frames,
            args.output_path,
            ascii_dimensions,
            batch_size=args.batch_size
        )
        
        # If compare flag is set, create side-by-side comparison video
        if args.compare:
            print("Creating side-by-side comparison video...")
            comparison_path = os.path.splitext(args.output_path)[0] + "_comparison.mp4"
            video_processor.create_comparison_video(
                args.input_path,
                args.output_path,
                comparison_path
            )
            print(f"Comparison video saved to: {comparison_path}")
        
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