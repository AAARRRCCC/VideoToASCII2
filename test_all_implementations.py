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

def run_implementation(implementation, input_video, output_video, processes, batch_size, width=120, height=60):
    """
    Run a specific implementation of the video to ASCII conversion with specified parameters and measure execution time.
    
    Args:
        implementation (str): Implementation to run ('original', 'pipeline', or 'optimized')
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
    # Determine which script to run
    if implementation == 'original':
        script = "main.py"
    elif implementation == 'pipeline':
        script = "main_parallel.py"
    elif implementation == 'optimized':
        script = "main_optimized.py"
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    # Prepare command
    cmd = [
        "python", script,
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
    Compare all implementations with different configurations.
    
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
    os.makedirs("test_output/optimized", exist_ok=True)
    
    implementations = ['original', 'pipeline', 'optimized']
    
    for video_path in test_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        results[video_name] = {}
        
        for implementation in implementations:
            results[video_name][implementation] = {}
        
        # Test different process counts
        for processes in process_counts:
            for implementation in implementations:
                results[video_name][implementation][f"processes_{processes}"] = {}
            
            for batch_size in batch_sizes:
                test_name = f"p{processes}_b{batch_size}"
                
                for implementation in implementations:
                    output_path = f"test_output/{implementation}/{video_name}_{test_name}.mp4"
                    print(f"\nProcessing {video_name} with {implementation} implementation (processes={processes}, batch_size={batch_size})...")
                    
                    exec_time, stats = run_implementation(
                        implementation, video_path, output_path, processes, batch_size
                    )
                    
                    results[video_name][implementation][f"processes_{processes}"][f"batch_{batch_size}"] = {
                        "time": exec_time,
                        "stats": stats
                    }
                    
                    print(f"{implementation.capitalize()} implementation time: {exec_time:.2f} seconds")
    
    # Calculate speedups relative to original implementation
    for video_name in results:
        for processes in process_counts:
            for batch_size in batch_sizes:
                proc_key = f"processes_{processes}"
                batch_key = f"batch_{batch_size}"
                
                # Get original time
                original_time = results[video_name]['original'][proc_key][batch_key]['time']
                
                # Calculate speedups for other implementations
                for implementation in ['pipeline', 'optimized']:
                    impl_time = results[video_name][implementation][proc_key][batch_key]['time']
                    speedup = original_time / impl_time if impl_time > 0 else 0
                    results[video_name][implementation][proc_key][batch_key]['speedup'] = speedup
    
    return results

def plot_comparison_results(results):
    """
    Plot comparison results between all implementations.
    
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
        optimized_times = []
        pipeline_speedups = []
        optimized_speedups = []
        
        for proc_key, proc_value in video_results["original"].items():
            if proc_key.startswith("processes_"):
                process_count = int(proc_key.split("_")[1])
                
                for batch_key, batch_value in proc_value.items():
                    if batch_key.startswith("batch_"):
                        batch_size = int(batch_key.split("_")[1])
                        
                        # Get times for each implementation
                        original_time = batch_value["time"]
                        pipeline_time = video_results["pipeline"][proc_key][batch_key]["time"]
                        optimized_time = video_results["optimized"][proc_key][batch_key]["time"]
                        
                        # Get speedups
                        pipeline_speedup = video_results["pipeline"][proc_key][batch_key].get("speedup", 0)
                        optimized_speedup = video_results["optimized"][proc_key][batch_key].get("speedup", 0)
                        
                        process_counts.append(process_count)
                        batch_sizes.append(batch_size)
                        original_times.append(original_time)
                        pipeline_times.append(pipeline_time)
                        optimized_times.append(optimized_time)
                        pipeline_speedups.append(pipeline_speedup)
                        optimized_speedups.append(optimized_speedup)
        
        # Create execution time comparison plot
        plt.figure(figsize=(14, 8))
        
        # Create a unique identifier for each configuration
        config_labels = [f"P{p},B{b}" for p, b in zip(process_counts, batch_sizes)]
        x = np.arange(len(config_labels))
        width = 0.25
        
        # Plot times for all implementations as grouped bars
        plt.bar(x - width, original_times, width, label='Original Implementation')
        plt.bar(x, pipeline_times, width, label='Pipeline Implementation')
        plt.bar(x + width, optimized_times, width, label='Optimized Implementation')
        
        plt.xlabel('Configuration (Processes, Batch Size)')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Implementation Comparison for {video_name}')
        plt.xticks(x, config_labels, rotation=45)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(f"test_output/plots/{video_name}_all_implementations_comparison.png")
        
        # Create speedup comparison plot
        plt.figure(figsize=(14, 8))
        
        # Plot speedups for pipeline and optimized implementations
        plt.bar(x - width/2, pipeline_speedups, width, label='Pipeline Speedup')
        plt.bar(x + width/2, optimized_speedups, width, label='Optimized Speedup')
        
        plt.xlabel('Configuration (Processes, Batch Size)')
        plt.ylabel('Speedup Factor (vs Original)')
        plt.title(f'Speedup Comparison for {video_name}')
        plt.xticks(x, config_labels, rotation=45)
        plt.axhline(y=1.0, color='r', linestyle='-', label='Baseline (Original)')
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(f"test_output/plots/{video_name}_speedup_comparison.png")
        
        # Find best configurations for each implementation
        best_configs = {}
        for implementation in ['original', 'pipeline', 'optimized']:
            best_time = float('inf')
            best_config = None
            
            for i, (proc, batch) in enumerate(zip(process_counts, batch_sizes)):
                time_value = [original_times, pipeline_times, optimized_times][
                    ['original', 'pipeline', 'optimized'].index(implementation)
                ][i]
                
                if time_value < best_time:
                    best_time = time_value
                    best_config = (proc, batch)
            
            best_configs[implementation] = {
                'config': best_config,
                'time': best_time
            }
        
        # Create a comparison bar chart for the best configurations
        plt.figure(figsize=(10, 6))
        
        labels = ['Original', 'Pipeline', 'Optimized']
        best_times = [best_configs[impl]['time'] for impl in ['original', 'pipeline', 'optimized']]
        best_configs_text = [f"P={best_configs[impl]['config'][0]}, B={best_configs[impl]['config'][1]}" 
                            for impl in ['original', 'pipeline', 'optimized']]
        
        bars = plt.bar(labels, best_times, color=['blue', 'green', 'red'])
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Best Configuration Comparison for {video_name}')
        
        # Add execution time and configuration labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s\n{best_configs_text[i]}', 
                    ha='center', va='bottom')
        
        # Add speedup annotations
        original_best_time = best_configs['original']['time']
        for i, impl in enumerate(['pipeline', 'optimized']):
            if i > 0:  # Skip original
                best_time = best_configs[impl]['time']
                speedup = original_best_time / best_time if best_time > 0 else 0
                
                plt.annotate(f"{speedup:.2f}x faster", 
                            xy=(i, best_time), 
                            xytext=(i - 0.5, (original_best_time + best_time)/2),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            ha='center')
        
        plt.tight_layout()
        plt.savefig(f"test_output/plots/{video_name}_best_configs_comparison.png")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare all implementations")
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
    print("\nComparing all implementations...")
    results = compare_implementations(test_videos, process_counts, batch_sizes)
    
    # Plot results
    print("\nPlotting comparison results...")
    plot_comparison_results(results)
    
    # Print summary
    print("\nComparison Summary:")
    for video_name, video_results in results.items():
        print(f"\n{video_name}:")
        
        # Find best configuration for each implementation
        for implementation in ['original', 'pipeline', 'optimized']:
            best_time = float('inf')
            best_config = None
            
            for proc_key, proc_value in video_results[implementation].items():
                if proc_key.startswith("processes_"):
                    process_count = int(proc_key.split("_")[1])
                    
                    for batch_key, batch_value in proc_value.items():
                        if batch_key.startswith("batch_"):
                            batch_size = int(batch_key.split("_")[1])
                            
                            time_value = batch_value["time"]
                            
                            if time_value < best_time:
                                best_time = time_value
                                best_config = (process_count, batch_size)
            
            print(f"  Best {implementation} configuration: {best_config[0]} processes, batch size {best_config[1]}")
            print(f"  Best {implementation} time: {best_time:.2f} seconds")
            
            # Calculate speedup compared to original (for non-original implementations)
            if implementation != 'original':
                # Find best original time
                best_original_time = float('inf')
                for proc_key, proc_value in video_results['original'].items():
                    for batch_key, batch_value in proc_value.items():
                        time_value = batch_value["time"]
                        if time_value < best_original_time:
                            best_original_time = time_value
                
                speedup = best_original_time / best_time if best_time > 0 else 0
                print(f"  Speedup over best original: {speedup:.2f}x")
                print(f"  Time reduction: {best_original_time - best_time:.2f} seconds ({(best_original_time - best_time)/best_original_time*100:.1f}%)")
            
            print("")
    
    print("Detailed results and plots saved to test_output directory")

if __name__ == "__main__":
    main()