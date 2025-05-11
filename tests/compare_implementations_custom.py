import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pandas as pd
import seaborn as sns
from datetime import datetime
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_implementation(implementation, input_video, output_video, processes=None, batch_size=10, width=120, height=60, timeout=300):
    """
    Run a specific implementation of the video to ASCII conversion with specified parameters and measure execution time.
    
    Args:
        implementation (str): Implementation to run ('original', 'parallel', 'optimized', or 'enhanced')
        input_video (str): Path to input video
        output_video (str): Path to output video
        processes (int): Number of processes to use
        batch_size (int): Batch size for processing
        width (int): Width of ASCII output
        height (int): Height of ASCII output
        timeout (int): Maximum time to wait for implementation to complete (in seconds)
        
    Returns:
        float: Execution time in seconds
    """
    # Determine which script to run
    script = "main.py"
    
    # Prepare command
    cmd = [
        "python", script,
        input_video, output_video,
        "--width", str(width),
        "--height", str(height),
        "--implementation", implementation
    ]
    
    if processes is not None:
        cmd.extend(["--processes", str(processes)])
    
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run the command with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Record end time
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        # Print any errors
        if result.returncode != 0:
            print(f"Error running {implementation} implementation:")
            print(result.stderr)
        
        return execution_time
    
    except subprocess.TimeoutExpired:
        print(f"\n*** TIMEOUT: {implementation} implementation took longer than {timeout} seconds ***")
        print("Terminating process and continuing with next test...")
        
        # Return a placeholder value
        return float('inf')

def run_comparison(input_video, output_dir, process_counts, batch_sizes):
    """
    Run a comparison of all implementations with different configurations.
    
    Args:
        input_video (str): Path to input video
        output_dir (str): Directory to save output videos and graphs
        process_counts (list): List of process counts to test
        batch_sizes (list): List of batch sizes to test
    
    Returns:
        dict: Results of the comparison
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each implementation
    for impl in ['simple', 'parallel', 'enhanced', 'optimized']:
        os.makedirs(os.path.join(output_dir, impl), exist_ok=True)
    
    # Create a directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Store results
    results = {}
    
    # Get video name
    video_name = os.path.basename(input_video).split('.')[0]
    results[video_name] = {}
    
    implementations = ['simple', 'parallel', 'enhanced', 'optimized']
    
    for implementation in implementations:
        results[video_name][implementation] = {}
    
    # Test different process counts
    for processes in process_counts:
        for implementation in implementations:
            results[video_name][implementation][f"processes_{processes}"] = {}
        
        for batch_size in batch_sizes:
            test_name = f"p{processes}_b{batch_size}"
            
            for implementation in implementations:
                output_path = os.path.join(output_dir, implementation, f"{video_name}_{test_name}.mp4")
                print(f"\nProcessing {video_name} with {implementation} implementation (processes={processes}, batch_size={batch_size})...")
                
                # Use a shorter timeout for testing
                exec_time = run_implementation(
                    implementation, input_video, output_path, processes, batch_size,
                    timeout=180  # 3 minutes timeout
                )
                
                results[video_name][implementation][f"processes_{processes}"][f"batch_{batch_size}"] = {
                    "time": exec_time
                }
                
                if exec_time == float('inf'):
                    print(f"{implementation.capitalize()} implementation: TIMEOUT")
                else:
                    print(f"{implementation.capitalize()} implementation time: {exec_time:.2f} seconds")
    
    # Calculate speedups relative to simple implementation
    for processes in process_counts:
        for batch_size in batch_sizes:
            proc_key = f"processes_{processes}"
            batch_key = f"batch_{batch_size}"
            
            # Get simple time
            simple_time = results[video_name]['simple'][proc_key][batch_key]['time']
            
            # Calculate speedups for other implementations
            for implementation in ['parallel', 'enhanced', 'optimized']:
                impl_time = results[video_name][implementation][proc_key][batch_key]['time']
                
                # Skip timeout cases
                if impl_time == float('inf'):
                    results[video_name][implementation][proc_key][batch_key]['speedup'] = 0
                    continue
                
                speedup = simple_time / impl_time if impl_time > 0 else 0
                results[video_name][implementation][proc_key][batch_key]['speedup'] = speedup
    
    return results

def plot_results(results, output_dir):
    """
    Plot the results of the comparison.
    
    Args:
        results (dict): Results of the comparison
        output_dir (str): Directory to save plots
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for video_name, video_results in results.items():
        # Prepare data for plotting
        process_counts = []
        batch_sizes = []
        simple_times = []
        parallel_times = []
        optimized_times = []
        enhanced_times = []
        parallel_speedups = []
        optimized_speedups = []
        enhanced_speedups = []
        
        for proc_key, proc_value in video_results["simple"].items():
            if proc_key.startswith("processes_"):
                process_count = int(proc_key.split("_")[1])
                
                for batch_key, batch_value in proc_value.items():
                    if batch_key.startswith("batch_"):
                        batch_size = int(batch_key.split("_")[1])
                        
                        # Get times for each implementation
                        simple_time = batch_value["time"]
                        parallel_time = video_results["parallel"][proc_key][batch_key]["time"]
                        optimized_time = video_results["optimized"][proc_key][batch_key]["time"]
                        enhanced_time = video_results["enhanced"][proc_key][batch_key]["time"]
                        
                        # Skip configurations where any implementation timed out
                        if (simple_time == float('inf') or parallel_time == float('inf') or
                            optimized_time == float('inf') or enhanced_time == float('inf')):
                            print(f"Skipping configuration P{process_count},B{batch_size} due to timeout")
                            continue
                        
                        # Get speedups
                        parallel_speedup = video_results["parallel"][proc_key][batch_key].get("speedup", 0)
                        optimized_speedup = video_results["optimized"][proc_key][batch_key].get("speedup", 0)
                        enhanced_speedup = video_results["enhanced"][proc_key][batch_key].get("speedup", 0)
                        
                        process_counts.append(process_count)
                        batch_sizes.append(batch_size)
                        simple_times.append(simple_time)
                        parallel_times.append(parallel_time)
                        optimized_times.append(optimized_time)
                        enhanced_times.append(enhanced_time)
                        parallel_speedups.append(parallel_speedup)
                        optimized_speedups.append(optimized_speedup)
                        enhanced_speedups.append(enhanced_speedup)
        
        # Create execution time comparison plot
        plt.figure(figsize=(14, 8))
        
        # Create a unique identifier for each configuration
        config_labels = [f"P{p},B{b}" for p, b in zip(process_counts, batch_sizes)]
        x = np.arange(len(config_labels))
        width = 0.2
        
        # Plot times for all implementations as grouped bars
        plt.bar(x - width*1.5, simple_times, width, label='Simple Implementation')
        plt.bar(x - width*0.5, parallel_times, width, label='Parallel Implementation')
        plt.bar(x + width*0.5, optimized_times, width, label='Optimized Implementation')
        plt.bar(x + width*1.5, enhanced_times, width, label='Enhanced Implementation')
        
        plt.xlabel('Configuration (Processes, Batch Size)')
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Implementation Comparison for {video_name}')
        plt.xticks(x, config_labels, rotation=45)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, f"{video_name}_all_implementations_comparison.png"))
        
        # Create speedup comparison plot
        plt.figure(figsize=(14, 8))
        
        # Plot speedups for parallel, optimized and enhanced implementations
        plt.bar(x - width, parallel_speedups, width, label='Parallel Speedup')
        plt.bar(x, optimized_speedups, width, label='Optimized Speedup')
        plt.bar(x + width, enhanced_speedups, width, label='Enhanced Speedup')
        
        plt.xlabel('Configuration (Processes, Batch Size)')
        plt.ylabel('Speedup Factor (vs Simple)')
        plt.title(f'Speedup Comparison for {video_name}')
        plt.xticks(x, config_labels, rotation=45)
        plt.axhline(y=1.0, color='r', linestyle='-', label='Baseline (Simple)')
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, f"{video_name}_speedup_comparison.png"))
        
        # Create a heatmap of speedups
        plt.figure(figsize=(12, 8))
        
        # Reshape data for heatmap
        unique_processes = sorted(set(process_counts))
        unique_batches = sorted(set(batch_sizes))
        
        heatmap_data = np.zeros((len(unique_processes), len(unique_batches)))
        
        for i, p in enumerate(unique_processes):
            for j, b in enumerate(unique_batches):
                # Find the index for this process/batch combination
                idx = [k for k, (proc, batch) in enumerate(zip(process_counts, batch_sizes)) 
                      if proc == p and batch == b]
                
                if idx:
                    heatmap_data[i, j] = enhanced_speedups[idx[0]]
        
        # Create heatmap
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=unique_batches, yticklabels=unique_processes)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Number of Processes')
        plt.title(f'Enhanced Implementation Speedup Heatmap for {video_name}')
        
        plt.savefig(os.path.join(plots_dir, f"{video_name}_enhanced_speedup_heatmap.png"))

def print_summary(results, output_dir):
    """
    Print a summary of the results and save to a file.
    
    Args:
        results (dict): Results of the comparison
        output_dir (str): Directory to save summary
    """
    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    
    with open(summary_path, "w") as f:
        for video_name, video_results in results.items():
            f.write(f"Summary for {video_name}:\n")
            f.write("=" * 50 + "\n\n")
            
            # Find best configuration for each implementation
            for implementation in ['simple', 'parallel', 'enhanced', 'optimized']:
                best_time = float('inf')
                best_config = None
                
                for proc_key, proc_value in video_results[implementation].items():
                    for batch_key, batch_value in proc_value.items():
                        time_value = batch_value["time"]
                        
                        if time_value < best_time and time_value != float('inf'):
                            best_time = time_value
                            process_count = int(proc_key.split("_")[1])
                            batch_size = int(batch_key.split("_")[1])
                            best_config = (process_count, batch_size)
                
                if best_config:
                    f.write(f"{implementation.capitalize()} Implementation:\n")
                    f.write(f"  Best configuration: Processes={best_config[0]}, Batch Size={best_config[1]}\n")
                    f.write(f"  Execution time: {best_time:.2f} seconds\n")
                    
                    if implementation != 'simple':
                        # Calculate speedup compared to simple
                        simple_best_time = float('inf')
                        for proc_key, proc_value in video_results['simple'].items():
                            for batch_key, batch_value in proc_value.items():
                                time_value = batch_value["time"]
                                if time_value < simple_best_time and time_value != float('inf'):
                                    simple_best_time = time_value
                        
                        speedup = simple_best_time / best_time if best_time > 0 else 0
                        f.write(f"  Speedup vs. Simple: {speedup:.2f}x\n")
                    
                    f.write("\n")
            
            f.write("\n\n")
    
    print(f"Summary saved to {summary_path}")

def main():
    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"custom_comparison_{timestamp}"
    
    # Define test parameters
    input_video = "test_videos/medium.mp4"  # Use a medium-sized test video
    process_counts = [1, 2, 4]  # Test with 1, 2, and 4 processes
    batch_sizes = [5, 10, 20]  # Test with different batch sizes
    
    # Run the comparison
    print(f"Starting comparison with input video: {input_video}")
    print(f"Output directory: {output_dir}")
    print(f"Process counts: {process_counts}")
    print(f"Batch sizes: {batch_sizes}")
    
    results = run_comparison(input_video, output_dir, process_counts, batch_sizes)
    
    # Plot the results
    plot_results(results, output_dir)
    
    # Print summary
    print_summary(results, output_dir)
    
    print(f"Comparison completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()