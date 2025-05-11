import psutil
import time
import argparse
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def monitor_resources(command, interval=0.5, output_dir="test_output/resources"):
    """
    Monitor CPU and memory usage while executing a command.
    
    Args:
        command (list): Command to execute as a list of arguments
        interval (float): Sampling interval in seconds
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Resource usage data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start the process
    process = subprocess.Popen(command)
    
    # Initialize data collection
    timestamps = []
    cpu_percentages = []
    memory_usages = []
    
    # Monitor resources until process completes
    start_time = time.time()
    try:
        while process.poll() is None:
            # Get current timestamp
            current_time = time.time() - start_time
            timestamps.append(current_time)
            
            # Get CPU usage
            try:
                p = psutil.Process(process.pid)
                cpu_percent = p.cpu_percent(interval=0.1)
                cpu_percentages.append(cpu_percent)
                
                # Get memory usage (in MB)
                memory_info = p.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                memory_usages.append(memory_mb)
                
                print(f"\rRunning: {current_time:.1f}s | CPU: {cpu_percent:.1f}% | Memory: {memory_mb:.1f} MB", end="")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have terminated between poll and trying to get info
                break
            
            # Wait for next sample
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        process.terminate()
    
    # Process completed
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcess completed in {total_time:.2f} seconds")
    
    # Return collected data
    return {
        "timestamps": timestamps,
        "cpu_percentages": cpu_percentages,
        "memory_usages": memory_usages,
        "total_time": total_time
    }

def plot_resource_usage(data, title, output_path):
    """
    Plot resource usage data.
    
    Args:
        data (dict): Resource usage data
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    timestamps = data["timestamps"]
    cpu_percentages = data["cpu_percentages"]
    memory_usages = data["memory_usages"]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Ensure arrays have the same length
    min_length = min(len(timestamps), len(cpu_percentages), len(memory_usages))
    timestamps = timestamps[:min_length]
    cpu_percentages = cpu_percentages[:min_length]
    memory_usages = memory_usages[:min_length]
    
    # Plot CPU usage on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('CPU Usage (%)', color=color)
    ax1.plot(timestamps, cpu_percentages, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(100, max(cpu_percentages) * 1.1) if cpu_percentages else 100)
    
    # Create a second y-axis for memory usage
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Memory Usage (MB)', color=color)
    ax2.plot(timestamps, memory_usages, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(memory_usages) * 1.1 if memory_usages else 100)
    
    # Add title and adjust layout
    plt.title(title)
    fig.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Monitor resource usage during command execution")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds")
    args = parser.parse_args()
    
    # Prepare command
    command = [
        "python", "../main.py",
        args.input, args.output,
        "--batch-size", str(args.batch_size)
    ]
    
    if args.processes is not None:
        command.extend(["--processes", str(args.processes)])
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive name for output files
    processes_str = f"p{args.processes}" if args.processes is not None else "pAuto"
    batch_str = f"b{args.batch_size}"
    base_name = f"{timestamp}_{processes_str}_{batch_str}"
    
    # Monitor resources
    print(f"Monitoring resource usage for command: {' '.join(command)}")
    data = monitor_resources(
        command, 
        interval=args.interval,
        output_dir=f"../test_output/resources"
    )
    
    # Plot resource usage
    title = f"Resource Usage (Processes: {args.processes or 'Auto'}, Batch Size: {args.batch_size})"
    plot_path = f"../test_output/resources/{base_name}_resources.png"
    plot_resource_usage(data, title, plot_path)
    
    print(f"Resource usage plot saved to: {plot_path}")
    
    # Save raw data
    import json
    data_path = f"../test_output/resources/{base_name}_data.json"
    with open(data_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            "timestamps": list(data["timestamps"]),
            "cpu_percentages": list(data["cpu_percentages"]),
            "memory_usages": list(data["memory_usages"]),
            "total_time": data["total_time"],
            "command": command,
            "processes": args.processes,
            "batch_size": args.batch_size
        }
        json.dump(serializable_data, f, indent=2)
    
    print(f"Raw data saved to: {data_path}")

if __name__ == "__main__":
    main()