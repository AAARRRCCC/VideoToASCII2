import os
import time
import subprocess
import psutil
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import shutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def run_conversion(input_video, output_video, processes, batch_size, width=120, height=60):
    """
    Run the video to ASCII conversion with specified parameters and measure execution time.
    
    Args:
        input_video (str): Path to input video
        output_video (str): Path to output video
        processes (int): Number of processes to use
        batch_size (int): Batch size for processing
        width (int): Width of ASCII output
        height (int): Height of ASCII output
        
    Returns:
        float: Execution time in seconds
        dict: Resource usage statistics
    """
    # Prepare command
    cmd = [
        "python", "main.py",
        input_video, output_video,
        "--width", str(width),
        "--height", str(height),
        "--processes", str(processes),
        "--batch-size", str(batch_size)
    ]
    
    # Start monitoring resources
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Record start time
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Record end time
    end_time = time.time()
    
    # Calculate execution time
    execution_time = end_time - start_time
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Collect resource statistics
    resource_stats = {
        "memory_increase_mb": memory_increase,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }
    
    return execution_time, resource_stats

def compare_images(image1_path, image2_path):
    """
    Compare two images and calculate similarity metrics.
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        
    Returns:
        float: Structural similarity index (higher is more similar)
    """
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity index
    from skimage.metrics import structural_similarity as ssim
    similarity = ssim(gray1, gray2)
    
    return similarity

def extract_frames(video_path, output_dir, num_frames=5):
    """
    Extract frames from a video for quality comparison.
    
    Args:
        video_path (str): Path to video
        output_dir (str): Directory to save frames
        num_frames (int): Number of frames to extract
        
    Returns:
        list: Paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Extract frames
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths

def run_performance_tests(test_videos, process_counts, batch_sizes):
    """
    Run performance tests with different configurations.
    
    Args:
        test_videos (list): List of test video paths
        process_counts (list): List of process counts to test
        batch_sizes (list): List of batch sizes to test
        
    Returns:
        dict: Test results
    """
    results = {}
    
    # Create output directories
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_output/frames", exist_ok=True)
    
    for video_path in test_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        results[video_name] = {}
        
        # Reference output (sequential processing)
        reference_output = f"test_output/{video_name}_sequential.mp4"
        print(f"\nProcessing {video_name} with sequential processing (processes=1, batch_size=1)...")
        seq_time, seq_stats = run_conversion(video_path, reference_output, 1, 1)
        results[video_name]["sequential"] = {
            "time": seq_time,
            "stats": seq_stats
        }
        print(f"Sequential processing time: {seq_time:.2f} seconds")
        
        # Extract reference frames
        reference_frames_dir = f"test_output/frames/{video_name}_sequential"
        reference_frames = extract_frames(reference_output, reference_frames_dir)
        
        # Test different process counts
        for processes in process_counts:
            if processes == 1:  # Skip, already done as reference
                continue
                
            results[video_name][f"processes_{processes}"] = {}
            
            for batch_size in batch_sizes:
                test_name = f"p{processes}_b{batch_size}"
                output_path = f"test_output/{video_name}_{test_name}.mp4"
                
                print(f"Processing {video_name} with {processes} processes, batch size {batch_size}...")
                exec_time, stats = run_conversion(video_path, output_path, processes, batch_size)
                
                # Extract frames for quality comparison
                test_frames_dir = f"test_output/frames/{video_name}_{test_name}"
                test_frames = extract_frames(output_path, test_frames_dir)
                
                # Compare frames with reference
                similarities = []
                for ref_frame, test_frame in zip(reference_frames, test_frames):
                    similarity = compare_images(ref_frame, test_frame)
                    similarities.append(similarity)
                
                # Calculate speedup
                speedup = seq_time / exec_time
                
                results[video_name][f"processes_{processes}"][f"batch_{batch_size}"] = {
                    "time": exec_time,
                    "speedup": speedup,
                    "similarity": np.mean(similarities),
                    "stats": stats
                }
                
                print(f"Execution time: {exec_time:.2f} seconds")
                print(f"Speedup factor: {speedup:.2f}x")
                print(f"Average frame similarity: {np.mean(similarities):.4f}")
    
    return results

def plot_results(results):
    """
    Plot performance test results.
    
    Args:
        results (dict): Test results
    """
    os.makedirs("test_output/plots", exist_ok=True)
    
    for video_name, video_results in results.items():
        # Extract sequential time
        seq_time = video_results["sequential"]["time"]
        
        # Prepare data for plotting
        process_counts = []
        batch_sizes = []
        speedups = []
        similarities = []
        execution_times = []
        
        # Add sequential run data (for absolute time comparison)
        process_counts.append(1)
        batch_sizes.append(1)
        speedups.append(1.0)  # By definition, speedup is 1.0 for sequential
        similarities.append(1.0)  # By definition, similarity is 1.0 for sequential
        execution_times.append(seq_time)
        
        for key, value in video_results.items():
            if key == "sequential":
                continue
            
            # Extract process count
            if key.startswith("processes_"):
                process_count = int(key.split("_")[1])
                
                for batch_key, batch_value in value.items():
                    if batch_key.startswith("batch_"):
                        batch_size = int(batch_key.split("_")[1])
                        
                        process_counts.append(process_count)
                        batch_sizes.append(batch_size)
                        speedups.append(batch_value["speedup"])
                        similarities.append(batch_value["similarity"])
                        execution_times.append(batch_value["time"])
        
        # Create execution time vs. process count plot (absolute times)
        plt.figure(figsize=(10, 6))
        
        # Add sequential run as a separate point with special marker
        plt.plot(1, seq_time, marker='*', markersize=15, color='red', label="Sequential (Non-Parallel)")
        
        # Group by batch size (excluding sequential point which we already plotted)
        unique_batch_sizes = sorted(set(batch_sizes[1:]))
        for batch_size in unique_batch_sizes:
            batch_indices = [i for i, b in enumerate(batch_sizes) if b == batch_size and i > 0]  # Skip sequential
            plt.plot(
                [process_counts[i] for i in batch_indices],
                [execution_times[i] for i in batch_indices],
                marker='o',
                label=f"Batch Size: {batch_size}"
            )
        
        plt.xlabel("Number of Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Execution Time vs. Process Count for {video_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_output/plots/{video_name}_time_vs_processes.png")
        
        # Create speedup vs. process count plot
        plt.figure(figsize=(10, 6))
        
        # Add sequential run as a reference point
        plt.plot(1, 1.0, marker='*', markersize=15, color='red', label="Sequential (Non-Parallel)")
        
        # Group by batch size (excluding sequential point which we already plotted)
        for batch_size in unique_batch_sizes:
            batch_indices = [i for i, b in enumerate(batch_sizes) if b == batch_size and i > 0]  # Skip sequential
            plt.plot(
                [process_counts[i] for i in batch_indices],
                [speedups[i] for i in batch_indices],
                marker='o',
                label=f"Batch Size: {batch_size}"
            )
        
        plt.xlabel("Number of Processes")
        plt.ylabel("Speedup Factor")
        plt.title(f"Speedup vs. Process Count for {video_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_output/plots/{video_name}_speedup_vs_processes.png")
        
        # Create speedup vs. batch size plot
        plt.figure(figsize=(10, 6))
        
        # Add sequential run as a reference point
        plt.plot(1, 1.0, marker='*', markersize=15, color='red', label="Sequential (Non-Parallel)")
        
        # Group by process count (excluding sequential point which we already plotted)
        unique_process_counts = sorted(set(process_counts[1:]))  # Skip sequential
        for process_count in unique_process_counts:
            process_indices = [i for i, p in enumerate(process_counts) if p == process_count and i > 0]  # Skip sequential
            plt.plot(
                [batch_sizes[i] for i in process_indices],
                [speedups[i] for i in process_indices],
                marker='o',
                label=f"Processes: {process_count}"
            )
        
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup Factor")
        plt.title(f"Speedup vs. Batch Size for {video_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_output/plots/{video_name}_speedup_vs_batch_size.png")
        
        # Create similarity plot
        plt.figure(figsize=(10, 6))
        
        # Add sequential run as a reference point
        plt.scatter(1.0, 1.0, s=200, c='red', marker='*', label="Sequential (Non-Parallel)")
        
        # Plot other points (excluding sequential)
        sc = plt.scatter(speedups[1:], similarities[1:], c=process_counts[1:], cmap='viridis')
        plt.colorbar(sc, label="Number of Processes")
        plt.xlabel("Speedup Factor")
        plt.ylabel("Frame Similarity")
        plt.title(f"Quality vs. Performance for {video_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_output/plots/{video_name}_quality_vs_performance.png")
        
        # Create a comparison bar chart of sequential vs parallel execution times
        plt.figure(figsize=(12, 6))
        
        # Find the best parallel configuration
        best_parallel_time = min(execution_times[1:])
        best_parallel_index = execution_times.index(best_parallel_time)
        best_parallel_config = f"P{process_counts[best_parallel_index]}, B{batch_sizes[best_parallel_index]}"
        
        # Create bar chart comparing sequential vs best parallel
        labels = ['Sequential (Non-Parallel)', f'Best Parallel ({best_parallel_config})']
        times = [seq_time, best_parallel_time]
        speedup_text = f"{seq_time/best_parallel_time:.2f}x faster"
        
        bars = plt.bar(labels, times, color=['red', 'green'])
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Sequential vs Parallel Execution Time for {video_name}')
        
        # Add execution time labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Add speedup annotation
        plt.annotate(speedup_text,
                    xy=(1, best_parallel_time),
                    xytext=(0.5, (seq_time + best_parallel_time)/2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center')
        
        plt.tight_layout()
        plt.savefig(f"test_output/plots/{video_name}_sequential_vs_parallel.png")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test parallelism in VideoToASCII converter")
    parser.add_argument("--skip-video-creation", action="store_true", help="Skip test video creation")
    args = parser.parse_args()
    
    # Create test videos if needed
    if not args.skip_video_creation:
        print("Creating test videos...")
        os.makedirs("test_videos", exist_ok=True)
        subprocess.run(["python", "create_test_video.py"])
    
    # Define test parameters
    test_videos = [
        "test_videos/small.mp4",
        "test_videos/medium.mp4",
        "test_videos/large.mp4"
    ]
    
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} CPU cores")
    
    # Define process counts to test (2, half of cores, all cores)
    # Note: Process count 1 will be used for the sequential baseline
    process_counts = [2, max(2, cpu_count // 2), cpu_count]
    process_counts = sorted(list(set(process_counts)))  # Remove duplicates
    
    # Define batch sizes to test
    batch_sizes = [1, 5, 10, 20, 50]
    
    print(f"Testing with process counts: {process_counts}")
    print(f"Testing with batch sizes: {batch_sizes}")
    print(f"Sequential baseline will use: processes=1, batch_size=1")
    
    # Run performance tests
    print("\nRunning performance tests...")
    results = run_performance_tests(test_videos, process_counts, batch_sizes)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results)
    
    # Print summary
    print("\nTest Summary:")
    for video_name, video_results in results.items():
        seq_time = video_results['sequential']['time']
        print(f"\n{video_name}:")
        print(f"  Sequential (Non-Parallel) processing time: {seq_time:.2f} seconds")
        
        # Find best configuration
        best_speedup = 0
        best_config = None
        best_time = float('inf')
        
        for key, value in video_results.items():
            if key == "sequential":
                continue
            
            for batch_key, batch_value in value.items():
                if batch_value["speedup"] > best_speedup:
                    best_speedup = batch_value["speedup"]
                    best_time = batch_value["time"]
                    process_count = int(key.split("_")[1])
                    batch_size = int(batch_key.split("_")[1])
                    best_config = (process_count, batch_size)
        
        if best_config:
            print(f"  Best parallel configuration: {best_config[0]} processes, batch size {best_config[1]}")
            print(f"  Best parallel processing time: {best_time:.2f} seconds")
            print(f"  Speedup over sequential: {best_speedup:.2f}x")
            print(f"  Time reduction: {(seq_time - best_time):.2f} seconds ({(seq_time - best_time)/seq_time*100:.1f}%)")
    
    print("\nDetailed results and plots saved to test_output directory")

if __name__ == "__main__":
    main()