import os
import time
import sys
from simple_processor import process_video_simple

def main():
    # Input and output paths
    input_video = "input_short.mp4"
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "output_simple.mp4")
    
    # Parameters
    processes = 4  # Use fewer processes for debugging
    batch_size = 5  # Use a small batch size
    
    print(f"Processing video: {input_video}")
    print(f"Output video: {output_video}")
    print(f"Processes: {processes}")
    print(f"Batch size: {batch_size}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Process video using simple processor
        process_video_simple(
            input_path=input_video,
            output_path=output_video,
            width=120,
            height=60,
            processes=processes,
            batch_size=batch_size,
            font_size=12,
            fps=30,
            temp_dir="./temp_simple"
        )
        
        # Record end time
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        print(f"Output saved to: {output_video}")
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()