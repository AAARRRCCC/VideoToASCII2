import os
import time
import sys
from src.processors.simple_processor import SimpleProcessor

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
        processor = SimpleProcessor()
        processor.process_video(
            video_path=input_video,
            output_path=output_video,
            width=120,
            height=60,
            fps_target=30
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