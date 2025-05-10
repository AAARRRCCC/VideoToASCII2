import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pandas as pd
import seaborn as sns
from datetime import datetime

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
    if implementation == 'original':
        script = "main.py"
    elif implementation == 'parallel':
        script = "main_parallel.py"
    elif implementation == 'optimized':
        script = "main_optimized.py"
    elif implementation == 'enhanced':
        # Create a temporary script for the enhanced implementation
        with open("main_enhanced.py", "w") as f:
            f.write("""
import argparse
import os
import sys
from enhanced_parallel_processor import process_video_enhanced
from utils import check_ffmpeg_installed, create_directory_if_not_exists

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert video to ASCII art using enhanced parallelism')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to output video file')
    parser.add_argument('--width', type=int, default=120, help='Maximum width of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--height', type=int, default=60, help='Maximum height of ASCII output in characters (aspect ratio will be preserved)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of output video')
    parser.add_argument('--font-size', type=int, default=12, help='Font size for ASCII characters')
    parser.add_argument('--temp-dir', type=str, default='./temp', help='Directory for temporary files')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use for parallel processing (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of frames to process in each batch (default: 10)')
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
    args.input_path = args.input_path.replace('\\\\', '/')
    args.output_path = args.output_path.replace('\\\\', '/')
    args.temp_dir = args.temp_dir.replace('\\\\', '/')
    
    # Create temporary directory if it doesn't exist
    create_directory_if_not_exists(args.temp_dir)
    
    try:
        # Process video using enhanced parallelism
        print(f"Processing video with enhanced parallelism: {args.input_path}")
        process_video_enhanced(
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
            """)
        script = "main_enhanced.py"
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    # Prepare command
    cmd = [
        "python", script,
        input_video, output_video,
        "--width", str(width),
        "--height", str(height)
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
    for impl in ['original', 'parallel', 'optimized', 'enhanced']:
        os.makedirs(os.path.join(output_dir, impl), exist_ok=True)
    
    # Create a directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Store results
    results = {}
    
    # Get video name
    video_name = os.path.basename(input_video).split('.')[0]
    results[video_name] = {}
    
    implementations = ['original', 'parallel', 'optimized', 'enhanced']
    
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
    
    # Calculate speedups relative to original implementation
    for processes in process_counts:
        for batch_size in batch_sizes:
            proc_key = f"processes_{processes}"
            batch_key = f"batch_{batch_size}"
            
            # Get original time
            original_time = results[video_name]['original'][proc_key][batch_key]['time']
            
            # Calculate speedups for other implementations
            for implementation in ['parallel', 'optimized', 'enhanced']:
                impl_time = results[video_name][implementation][proc_key][batch_key]['time']
                
                # Skip timeout cases
                if impl_time == float('inf'):
                    results[video_name][implementation][proc_key][batch_key]['speedup'] = 0
                    continue
                
                speedup = original_time / impl_time if impl_time > 0 else 0
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
        original_times = []
        parallel_times = []
        optimized_times = []
        enhanced_times = []
        parallel_speedups = []
        optimized_speedups = []
        enhanced_speedups = []
        
        for proc_key, proc_value in video_results["original"].items():
            if proc_key.startswith("processes_"):
                process_count = int(proc_key.split("_")[1])
                
                for batch_key, batch_value in proc_value.items():
                    if batch_key.startswith("batch_"):
                        batch_size = int(batch_key.split("_")[1])
                        
                        # Get times for each implementation
                        original_time = batch_value["time"]
                        parallel_time = video_results["parallel"][proc_key][batch_key]["time"]
                        optimized_time = video_results["optimized"][proc_key][batch_key]["time"]
                        enhanced_time = video_results["enhanced"][proc_key][batch_key]["time"]
                        
                        # Skip configurations where any implementation timed out
                        if (original_time == float('inf') or parallel_time == float('inf') or
                            optimized_time == float('inf') or enhanced_time == float('inf')):
                            print(f"Skipping configuration P{process_count},B{batch_size} due to timeout")
                            continue
                        
                        # Get speedups
                        parallel_speedup = video_results["parallel"][proc_key][batch_key].get("speedup", 0)
                        optimized_speedup = video_results["optimized"][proc_key][batch_key].get("speedup", 0)
                        enhanced_speedup = video_results["enhanced"][proc_key][batch_key].get("speedup", 0)
                        
                        process_counts.append(process_count)
                        batch_sizes.append(batch_size)
                        original_times.append(original_time)
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
        plt.bar(x - width*1.5, original_times, width, label='Original Implementation')
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
        plt.ylabel('Speedup Factor (vs Original)')
        plt.title(f'Speedup Comparison for {video_name}')
        plt.xticks(x, config_labels, rotation=45)
        plt.axhline(y=1.0, color='r', linestyle='-', label='Baseline (Original)')
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
        
        # Create a bar chart comparing the best configuration for each implementation
        plt.figure(figsize=(12, 8))
        
        # Find best configuration for each implementation
        best_configs = {}
        for implementation in ['original', 'parallel', 'optimized', 'enhanced']:
            best_time = float('inf')
            best_config = None
            
            for i, (proc, batch) in enumerate(zip(process_counts, batch_sizes)):
                time_value = [original_times, parallel_times, optimized_times, enhanced_times][
                    ['original', 'parallel', 'optimized', 'enhanced'].index(implementation)
                ][i]
                
                if time_value < best_time:
                    best_time = time_value
                    best_config = (proc, batch)
            
            best_configs[implementation] = {
                'config': best_config,
                'time': best_time
            }
        
        # Create a comparison bar chart for the best configurations
        labels = ['Original', 'Parallel', 'Optimized', 'Enhanced']
        best_times = [best_configs[impl]['time'] for impl in ['original', 'parallel', 'optimized', 'enhanced']]
        best_configs_text = [f"P={best_configs[impl]['config'][0]}, B={best_configs[impl]['config'][1]}" 
                            for impl in ['original', 'parallel', 'optimized', 'enhanced']]
        
        bars = plt.bar(labels, best_times, color=['blue', 'green', 'orange', 'red'])
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
        for i, impl in enumerate(['parallel', 'optimized', 'enhanced']):
            if i > 0:  # Skip original
                best_time = best_configs[impl]['time']
                speedup = original_best_time / best_time if best_time > 0 else 0
                
                plt.annotate(f"{speedup:.2f}x faster", 
                            xy=(i+1, best_time), 
                            xytext=(i+0.5, (original_best_time + best_time)/2),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{video_name}_best_configs_comparison.png"))
        
        # Create a line chart showing execution time vs process count for each implementation
        plt.figure(figsize=(14, 8))
        
        # Group data by process count and batch size
        process_data = {}
        for p in set(process_counts):
            process_data[p] = {
                'original': [],
                'parallel': [],
                'optimized': [],
                'enhanced': []
            }
        
        for i, (p, b) in enumerate(zip(process_counts, batch_sizes)):
            process_data[p]['original'].append(original_times[i])
            process_data[p]['parallel'].append(parallel_times[i])
            process_data[p]['optimized'].append(optimized_times[i])
            process_data[p]['enhanced'].append(enhanced_times[i])
        
        # Calculate average time for each process count
        process_counts_unique = sorted(set(process_counts))
        original_avg = [np.mean(process_data[p]['original']) for p in process_counts_unique]
        parallel_avg = [np.mean(process_data[p]['parallel']) for p in process_counts_unique]
        optimized_avg = [np.mean(process_data[p]['optimized']) for p in process_counts_unique]
        enhanced_avg = [np.mean(process_data[p]['enhanced']) for p in process_counts_unique]
        
        # Plot average times
        plt.plot(process_counts_unique, original_avg, 'o-', label='Original', linewidth=2)
        plt.plot(process_counts_unique, parallel_avg, 's-', label='Parallel', linewidth=2)
        plt.plot(process_counts_unique, optimized_avg, '^-', label='Optimized', linewidth=2)
        plt.plot(process_counts_unique, enhanced_avg, 'D-', label='Enhanced', linewidth=2)
        
        plt.xlabel('Number of Processes')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title(f'Execution Time vs Process Count for {video_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, f"{video_name}_time_vs_processes.png"))
        
        # Create a line chart showing execution time vs batch size for each implementation
        plt.figure(figsize=(14, 8))
        
        # Group data by batch size
        batch_data = {}
        for b in set(batch_sizes):
            batch_data[b] = {
                'original': [],
                'parallel': [],
                'optimized': [],
                'enhanced': []
            }
        
        for i, (p, b) in enumerate(zip(process_counts, batch_sizes)):
            batch_data[b]['original'].append(original_times[i])
            batch_data[b]['parallel'].append(parallel_times[i])
            batch_data[b]['optimized'].append(optimized_times[i])
            batch_data[b]['enhanced'].append(enhanced_times[i])
        
        # Calculate average time for each batch size
        batch_sizes_unique = sorted(set(batch_sizes))
        original_avg = [np.mean(batch_data[b]['original']) for b in batch_sizes_unique]
        parallel_avg = [np.mean(batch_data[b]['parallel']) for b in batch_sizes_unique]
        optimized_avg = [np.mean(batch_data[b]['optimized']) for b in batch_sizes_unique]
        enhanced_avg = [np.mean(batch_data[b]['enhanced']) for b in batch_sizes_unique]
        
        # Plot average times
        plt.plot(batch_sizes_unique, original_avg, 'o-', label='Original', linewidth=2)
        plt.plot(batch_sizes_unique, parallel_avg, 's-', label='Parallel', linewidth=2)
        plt.plot(batch_sizes_unique, optimized_avg, '^-', label='Optimized', linewidth=2)
        plt.plot(batch_sizes_unique, enhanced_avg, 'D-', label='Enhanced', linewidth=2)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title(f'Execution Time vs Batch Size for {video_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, f"{video_name}_time_vs_batch_size.png"))

def print_summary(results, output_dir):
    """
    Print a summary of the results and save it to a file.
    
    Args:
        results (dict): Results of the comparison
        output_dir (str): Directory to save the summary
    """
    summary = []
    summary.append("# Implementation Comparison Summary")
    summary.append("")
    
    for video_name, video_results in results.items():
        summary.append(f"## {video_name}")
        summary.append("")
        
        # Find best configuration for each implementation
        for implementation in ['original', 'parallel', 'optimized', 'enhanced']:
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
            
            summary.append(f"### Best {implementation.capitalize()} Configuration")
            summary.append(f"- Processes: {best_config[0]}")
            summary.append(f"- Batch Size: {best_config[1]}")
            summary.append(f"- Execution Time: {best_time:.2f} seconds")
            
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
                summary.append(f"- Speedup over best original: {speedup:.2f}x")
                summary.append(f"- Time reduction: {best_original_time - best_time:.2f} seconds ({(best_original_time - best_time)/best_original_time*100:.1f}%)")
            
            summary.append("")
        
        # Compare enhanced vs optimized
        best_optimized_time = float('inf')
        best_enhanced_time = float('inf')
        
        for proc_key, proc_value in video_results['optimized'].items():
            for batch_key, batch_value in proc_value.items():
                time_value = batch_value["time"]
                if time_value < best_optimized_time:
                    best_optimized_time = time_value
        
        for proc_key, proc_value in video_results['enhanced'].items():
            for batch_key, batch_value in proc_value.items():
                time_value = batch_value["time"]
                if time_value < best_enhanced_time:
                    best_enhanced_time = time_value
        
        improvement = best_optimized_time / best_enhanced_time if best_enhanced_time > 0 else 0
        summary.append(f"### Enhanced vs Optimized")
        summary.append(f"- Enhanced vs Optimized improvement: {improvement:.2f}x")
        summary.append(f"- Time difference: {best_optimized_time - best_enhanced_time:.2f} seconds ({(best_optimized_time - best_enhanced_time)/best_optimized_time*100:.1f}%)")
        summary.append("")
    
    # Print summary
    for line in summary:
        print(line)
    
    # Save summary to file
    with open(os.path.join(output_dir, "comparison_summary.md"), "w") as f:
        f.write("\n".join(summary))

def main():
    # Input video
    input_video = "input_short.mp4"
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"custom_comparison_{timestamp}"
    
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} CPU cores")
    
    # Define process counts to test
    # Include the optimal 75% of cores that the enhanced implementation uses by default
    process_counts = [max(2, cpu_count // 2), max(2, int(cpu_count * 0.75)), cpu_count]
    process_counts = sorted(list(set(process_counts)))  # Remove duplicates
    
    # Define batch sizes to test - use smaller batch sizes for this test
    batch_sizes = [5, 10, 20]
    
    # Run comparison
    print(f"\nComparing all implementations on {input_video}...")
    results = run_comparison(input_video, output_dir, process_counts, batch_sizes)
    
    # Plot results
    print("\nPlotting comparison results...")
    plot_results(results, output_dir)
    
    # Print summary
    print("\nComparison Summary:")
    print_summary(results, output_dir)
    
    print(f"\nResults saved to {output_dir} directory")

if __name__ == "__main__":
    main()