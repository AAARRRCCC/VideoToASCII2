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

def run_original_conversion(input_video, output_video, processes, batch_size, width=120, height=60):
    """
    Run the original video to ASCII conversion with specified parameters and measure execution time.
    
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

def run_pipeline_conversion(input_video, output_video, processes, batch_size, width=120, height=60):
    """
    Run the pipeline parallelism video to ASCII conversion with specified parameters and measure execution time.
    
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
        "python", "main_parallel.py",
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

def compare_implementations(test_videos, process_counts, batch_sizes):
    """
    Compare original and pipeline implementations with different configurations.
    
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
    os.makedirs("test_output/original", exist_ok=True)
    os.makedirs("test_output/pipeline", exist_ok=True)
    
    for video_path in test_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        results[video_name] = {
            "original": {},
            "pipeline": {}
        }
        
        # Test different process counts
        for processes in process_counts:
            results[video_name]["original"][f"processes_{processes}"] = {}
            results[video_name]["pipeline"][f"processes_{processes}"] = {}
            
            for batch_size in batch_sizes:
                test_name = f"p{processes}_b{batch_size}"
                
                # Run original implementation
                original_output = f"test_output/original/{video_name}_{test_name}.mp4"
                print(f"\nProcessing {video_name} with original implementation (processes={processes}, batch_size={batch_size})...")
                original_time, original_stats = run_original_conversion(
                    video_path, original_output, processes, batch_size
                )
                
                results[video_name]["original"][f"processes_{processes}"][f"batch_{batch_size}"] = {
                    "time": original_time,
                    "stats": original_stats
                }
                
                print(f"Original implementation time: {original_time:.2f} seconds")
                
                # Run pipeline implementation
                pipeline_output = f"test_output/pipeline/{video_name}_{test_name}.mp4"
                print(f"Processing {video_name} with pipeline implementation (processes={processes}, batch_size={batch_size})...")
                pipeline_time, pipeline_stats = run_pipeline_conversion(
                    video_path, pipeline_output, processes, batch_size
                )
                
                results[video_name]["pipeline"][f"processes_{processes}"][f"batch_{batch_size}"] = {
                    "time": pipeline_time,
                    "stats": pipeline_stats,
                    "speedup": original_time / pipeline_time if pipeline_time > 0 else 0
                }
                
                print(f"Pipeline implementation time: {pipeline_time:.2f} seconds")
                print(f"Speedup factor: {original_time / pipeline_time:.2f}x")
    
    return results

def plot_comparison_results(results):
    """
    Plot comparison results between original and pipeline implementations.
    
    Args:
        results (dict): Test results
    """
    os.makedirs("test_output/plots", exist_ok=True)
    
    for video_name, video_results in results.items():
        # Prepare data for plotting
        process_counts = []
        batch_sizes = []
        original_times = []
        pipeline_times = []
        speedups = []
        
        for proc_key, proc_value in video_results["original"].items():
            if proc_key.startswith("processes_"):
                process_count = int(proc_key.split("_")[1])
                
                for batch_key, batch_value in proc_value.items():
                    if batch_key.startswith("batch_"):
                        batch_size = int(batch_key.split("_")[1])
                        
                        # Get original time
                        original_time = batch_value["time"]
                        
                        # Get pipeline time
                        pipeline_time = video_results["pipeline"][proc_key][batch_key]["time"]
                        
                        # Calculate speedup
                        speedup = original_time / pipeline_time if pipeline_time > 0 else 0
                        
                        process_counts.append(process_count)
                        batch_sizes.append(batch_size)
                        original_times.append(original_time)
                        pipeline_times.append(pipeline_time)
                        speedups.append(speedup)
        
        # Create execution time comparison plot
        plt.figure(figsize=(12, 8))
        
        # Create a unique identifier for each configuration
        config_labels = [f"P{p},B{b}" for p, b in zip(process_counts, batch_sizes)]
        x = np.arange(len(config_labels))
        width = 0.35
        
        # Plot original and pipeline times as grouped bars
        plt.bar(x - width/2, original_times, width, label='Original Implementation')
        plt.bar(x + width/2, pipeline_times, width, label='Pipeline Implementation')
        
        plt.xlabel('Configuration (Processes, Batch Size)')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Original vs Pipeline Implementation for {video_name}')
        plt.xticks(x, config_labels, rotation=45)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Add speedup text above each pair of bars
        for i, (orig, pipe, speedup) in enumerate(zip(original_times, pipeline_times, speedups)):
            plt.text(i, max(orig, pipe) + 0.3, f"{speedup:.2f}x", 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.savefig(f"test_output/plots/{video_name}_implementation_comparison.png")
        
        # Create speedup vs. process count plot
        plt.figure(figsize=(10, 6))
        
        # Group by batch size
        unique_batch_sizes = sorted(set(batch_sizes))
        for batch_size in unique_batch_sizes:
            batch_indices = [i for i, b in enumerate(batch_sizes) if b == batch_size]
            plt.plot(
                [process_counts[i] for i in batch_indices],
                [speedups[i] for i in batch_indices],
                marker='o',
                label=f"Batch Size: {batch_size}"
            )
        
        plt.xlabel("Number of Processes")
        plt.ylabel("Speedup Factor (Original/Pipeline)")
        plt.title(f"Pipeline Speedup vs. Process Count for {video_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_output/plots/{video_name}_pipeline_speedup_vs_processes.png")
        
        # Create speedup vs. batch size plot
        plt.figure(figsize=(10, 6))
        
        # Group by process count
        unique_process_counts = sorted(set(process_counts))
        for process_count in unique_process_counts:
            process_indices = [i for i, p in enumerate(process_counts) if p == process_count]
            plt.plot(
                [batch_sizes[i] for i in process_indices],
                [speedups[i] for i in process_indices],
                marker='o',
                label=f"Processes: {process_count}"
            )
        
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup Factor (Original/Pipeline)")
        plt.title(f"Pipeline Speedup vs. Batch Size for {video_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"test_output/plots/{video_name}_pipeline_speedup_vs_batch_size.png")
        
        # Find best configuration
        best_speedup_idx = speedups.index(max(speedups))
        best_process_count = process_counts[best_speedup_idx]
        best_batch_size = batch_sizes[best_speedup_idx]
        best_original_time = original_times[best_speedup_idx]
        best_pipeline_time = pipeline_times[best_speedup_idx]
        best_speedup = speedups[best_speedup_idx]
        
        # Create a comparison bar chart for the best configuration
        plt.figure(figsize=(10, 6))
        
        labels = ['Original Implementation', 'Pipeline Implementation']
        times = [best_original_time, best_pipeline_time]
        
        bars = plt.bar(labels, times, color=['blue', 'green'])
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Best Configuration Comparison for {video_name} (P={best_process_count}, B={best_batch_size})')
        
        # Add execution time labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Add speedup annotation
        plt.annotate(f"{best_speedup:.2f}x faster", 
                    xy=(1, best_pipeline_time), 
                    xytext=(0.5, (best_original_time + best_pipeline_time)/2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center')
        
        plt.tight_layout()
        plt.savefig(f"test_output/plots/{video_name}_best_configuration_comparison.png")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare original and pipeline implementations")
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
    
    # Define process counts to test (half of cores, all cores)
    process_counts = [max(2, cpu_count // 2), cpu_count]
    
    # Define batch sizes to test
    batch_sizes = [10, 20]
    
    # Run comparison tests
    print("\nComparing original and pipeline implementations...")
    results = compare_implementations(test_videos, process_counts, batch_sizes)
    
    # Plot results
    print("\nPlotting comparison results...")
    plot_comparison_results(results)
    
    # Print summary
    print("\nComparison Summary:")
    for video_name, video_results in results.items():
        print(f"\n{video_name}:")
        
        # Find best configuration
        best_speedup = 0
        best_config = None
        best_original_time = 0
        best_pipeline_time = 0
        
        for proc_key, proc_value in video_results["original"].items():
            if proc_key.startswith("processes_"):
                process_count = int(proc_key.split("_")[1])
                
                for batch_key, batch_value in proc_value.items():
                    if batch_key.startswith("batch_"):
                        batch_size = int(batch_key.split("_")[1])
                        
                        original_time = batch_value["time"]
                        pipeline_time = video_results["pipeline"][proc_key][batch_key]["time"]
                        speedup = original_time / pipeline_time if pipeline_time > 0 else 0
                        
                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_config = (process_count, batch_size)
                            best_original_time = original_time
                            best_pipeline_time = pipeline_time
        
        if best_config:
            print(f"  Best configuration: {best_config[0]} processes, batch size {best_config[1]}")
            print(f"  Original implementation time: {best_original_time:.2f} seconds")
            print(f"  Pipeline implementation time: {best_pipeline_time:.2f} seconds")
            print(f"  Speedup: {best_speedup:.2f}x")
            print(f"  Time reduction: {best_original_time - best_pipeline_time:.2f} seconds ({(best_original_time - best_pipeline_time)/best_original_time*100:.1f}%)")
    
    print("\nDetailed results and plots saved to test_output directory")

if __name__ == "__main__":
    main()