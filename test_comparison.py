import os
import argparse
import sys
import subprocess
from datetime import datetime

def run_test(test_video, width=60, height=30, compare=True):
    """
    Run a test of the VideoToASCII converter with the comparison feature.
    
    Args:
        test_video (str): Path to the test video file
        width (int): Width of ASCII output in characters
        height (int): Height of ASCII output in characters
        compare (bool): Whether to create a comparison video
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_output/comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    ascii_output = os.path.join(output_dir, "ascii_output.mp4")
    
    # Build command
    cmd = [
        "python", "main.py",
        test_video,
        ascii_output,
        "--width", str(width),
        "--height", str(height)
    ]
    
    if compare:
        cmd.append("--compare")
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
        
        print("\nTest completed successfully!")
        print(f"ASCII output saved to: {ascii_output}")
        
        if compare:
            comparison_output = os.path.join(output_dir, "ascii_output_comparison.mp4")
            print(f"Comparison video saved to: {comparison_output}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the VideoToASCII comparison feature')
    parser.add_argument('--video', type=str, default='test_videos/small.mp4', 
                        help='Path to test video file (default: test_videos/small.mp4)')
    parser.add_argument('--width', type=int, default=60, 
                        help='Width of ASCII output in characters (default: 60)')
    parser.add_argument('--height', type=int, default=30, 
                        help='Height of ASCII output in characters (default: 30)')
    parser.add_argument('--no-compare', action='store_true', 
                        help='Disable comparison video creation (default: comparison enabled)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Check if test video exists
    if not os.path.exists(args.video):
        print(f"Error: Test video not found: {args.video}")
        print(f"Available test videos: {os.listdir('test_videos')}")
        sys.exit(1)
    
    run_test(
        test_video=args.video,
        width=args.width,
        height=args.height,
        compare=not args.no_compare
    )