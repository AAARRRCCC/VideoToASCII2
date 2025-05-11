import unittest
import os
import argparse
import sys
import subprocess
from datetime import datetime
import shutil
import tempfile
import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from skimage.metrics import structural_similarity as ssim

class TestVideoToASCII(unittest.TestCase):

    def run_command(self, cmd):
        """Helper to run a command and return the result."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def test_comparison(self):
        """Test the comparison feature."""
        print(f"\n=== Running Comparison Test ===")
        test_video = "test_videos/small.mp4"
        if not os.path.exists(test_video):
            self.skipTest(f"Test video not found at {test_video}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"test_output/comparison_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        ascii_output = os.path.join(output_dir, "ascii_output.mp4")

        cmd = [
            "python", "main.py",
            test_video,
            ascii_output,
            "--width", "60",
            "--height", "30",
            "--compare"
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = self.run_command(cmd)

        self.assertEqual(result.returncode, 0, f"Comparison test failed: {result.stderr}")
        self.assertTrue(os.path.exists(ascii_output), f"ASCII output file not created: {ascii_output}")
        comparison_output = os.path.join(output_dir, "ascii_output_comparison.mp4")
        self.assertTrue(os.path.exists(comparison_output), f"Comparison video not created: {comparison_output}")
        print("Comparison Test completed successfully!")

    def corrupt_video(self, input_video, output_video):
        """Create a corrupted copy of a video file."""
        shutil.copy2(input_video, output_video)
        with open(output_video, "r+b") as f:
            f.seek(0, 2)
            size = f.tell()
            f.truncate(int(size * 0.8))
        return output_video

    def test_error_handling(self):
        """Test error handling in the parallel implementation."""
        print(f"\n=== Running Error Handling Tests ===")
        error_test_video = "test_videos/small.mp4"
        if not os.path.exists(error_test_video):
            self.skipTest(f"Test video not found at {error_test_video}")

        os.makedirs("test_output/error_tests", exist_ok=True)
        corrupted_video = "test_output/error_tests/corrupted.mp4"
        self.corrupt_video(error_test_video, corrupted_video)

        error_tests = [
            {
                "name": "non_existent_input",
                "input": "non_existent_file.mp4",
                "processes": 4,
                "batch_size": 10,
                "expected_error": "Error: Input video file not found"
            },
            {
                "name": "corrupted_input",
                "input": corrupted_video,
                "processes": 4,
                "batch_size": 10,
                "expected_error": "Error: Could not open video file"
            },
            {
                "name": "negative_processes",
                "input": error_test_video,
                "processes": -1,
                "batch_size": 10,
                "expected_error": "Error: max_workers must be greater than 0"
            },
            {
                "name": "negative_batch_size",
                "input": error_test_video,
                "processes": 4,
                "batch_size": -1,
                "expected_error": "Error: object of type 'NoneType' has no len()"
            },
            {
                "name": "zero_processes",
                "input": error_test_video,
                "processes": 0,
                "batch_size": 10,
                "expected_error": "Error: max_workers must be greater than 0"
            },
            {
                "name": "zero_batch_size",
                "input": error_test_video,
                "processes": 4,
                "batch_size": 0,
                "expected_error": "Error: range() arg 3 must not be zero"
            }
        ]

        for test in error_tests:
            with self.subTest(msg=test["name"]):
                output_dir = os.path.join("test_output", "error_tests", test["name"])
                os.makedirs(output_dir, exist_ok=True)
                output_video = os.path.join(output_dir, "output.mp4")

                cmd = [
                    "python", "main.py",
                    test["input"], output_video,
                    "--processes", str(test["processes"]),
                    "--batch-size", str(test["batch_size"])
                ]

                result = self.run_command(cmd)

                self.assertNotEqual(result.returncode, 0, f"Test {test['name']} failed: Command succeeded but was expected to fail.")
                # Check if the expected error is in either stdout or stderr
                full_output = result.stdout + result.stderr
                self.assertIn(test["expected_error"], full_output, f"Test {test['name']} failed: Expected error '{test['expected_error']}' not found in output.\nStderr: {result.stderr}\nStdout: {result.stdout}")
                print(f"✅ Test passed: {test['name']}")

    def run_conversion(self, input_video, output_video, processes, batch_size, width=120, height=60):
        """Run the video to ASCII conversion and measure time/resources."""
        cmd = [
            "python", "main.py",
            input_video, output_video,
            "--width", str(width),
            "--height", str(height),
            "--processes", str(processes),
            "--batch-size", str(batch_size)
        ]

        start_time = time.time()
        result = self.run_command(cmd)
        execution_time = time.time() - start_time

        resource_stats = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

        return execution_time, resource_stats

    def compare_images(self, image1_path, image2_path):
        """Compare two images and calculate similarity metrics."""
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        similarity = ssim(gray1, gray2)
        return similarity

    def extract_frames(self, video_path, output_dir, num_frames=5):
        """Extract frames from a video for quality comparison."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
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

    def test_performance(self):
        """Run performance tests with different configurations."""
        print(f"\n=== Running Performance Tests ===")
        test_videos = ["test_videos/small.mp4"] # Add other test videos here
        process_counts = [1, 2, 4, 8, 16, 22] # Add other process counts here
        batch_sizes = [1, 10, 100] # Add other batch sizes here

        existing_performance_videos = [v for v in test_videos if os.path.exists(v)]

        if not existing_performance_videos:
            self.skipTest("No test videos found for performance testing.")

        results = {}
        os.makedirs("test_output", exist_ok=True)
        os.makedirs("test_output/frames", exist_ok=True)

        for video_path in existing_performance_videos:
            video_name = os.path.basename(video_path).split('.')[0]
            results[video_name] = {}

            # Reference output (sequential processing)
            reference_output = f"test_output/{video_name}_sequential.mp4"
            print(f"\nProcessing {video_name} with sequential processing (processes=1, batch_size=1)...")
            seq_time, seq_stats = self.run_conversion(video_path, reference_output, 1, 1)
            self.assertEqual(seq_stats["exit_code"], 0, f"Sequential conversion failed: {seq_stats['stderr']}")
            results[video_name]["sequential"] = {
                "time": seq_time,
                "stats": seq_stats
            }
            print(f"Sequential processing time: {seq_time:.2f} seconds")

            # Extract reference frames
            reference_frames_dir = f"test_output/frames/{video_name}_sequential"
            reference_frames = self.extract_frames(reference_output, reference_frames_dir)

            # Test different process counts
            for processes in process_counts:
                if processes == 1:
                    continue

                results[video_name][f"processes_{processes}"] = {}

                for batch_size in batch_sizes:
                    test_name = f"p{processes}_b{batch_size}"
                    output_path = f"test_output/{video_name}_{test_name}.mp4"

                    print(f"Processing {video_name} with {processes} processes, batch size {batch_size}...")
                    exec_time, stats = self.run_conversion(video_path, output_path, processes, batch_size)
                    self.assertEqual(stats["exit_code"], 0, f"Parallel conversion ({test_name}) failed: {stats['stderr']}")

                    # Extract frames for quality comparison
                    test_frames_dir = f"test_output/frames/{video_name}_{test_name}"
                    test_frames = self.extract_frames(output_path, test_frames_dir)

                    # Compare frames with reference
                    similarities = []
                    for ref_frame, test_frame in zip(reference_frames, test_frames):
                        similarity = self.compare_images(ref_frame, test_frame)
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
                    # Add assertions for performance and quality if needed
                    # self.assertGreater(speedup, 1.0, "Expected speedup in parallel mode")
                    # self.assertGreater(np.mean(similarities), 0.9, "Expected high frame similarity")

        plot_results(results, batch_sizes) # Plotting might not be suitable in this environment

    def test_profiling(self):
        """Test that the --profile flag runs without errors."""
        print(f"\n=== Running Profiling Test ===")
        test_video = "test_videos/small.mp4"
        if not os.path.exists(test_video):
            self.skipTest(f"Test video not found at {test_video}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"test_output/profiling_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        ascii_output = os.path.join(output_dir, "ascii_output.mp4")

        cmd = [
            "python", "main.py",
            test_video,
            ascii_output,
            "--profile"
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = self.run_command(cmd)

        self.assertEqual(result.returncode, 0, f"Profiling test failed: {result.stderr}")
        print("Profiling Test completed successfully!")

def plot_results(results, batch_sizes):
    """Plots the performance test results."""
    plot_dir = "test_output/plots"
    os.makedirs(plot_dir, exist_ok=True)

    for video_name, video_results in results.items():
        # Plot Execution Time vs. Processes for each batch size
        plt.figure(figsize=(10, 6))
        for batch_size in sorted(batch_sizes):
            processes = []
            times = []
            # Include sequential results (processes=1, batch_size=1)
            if batch_size == 1 and "sequential" in video_results:
                 processes.append(1)
                 times.append(video_results["sequential"]["time"])

            for proc_key, proc_data in video_results.items():
                if proc_key.startswith("processes_"):
                    p = int(proc_key.split("_")[1])
                    if f"batch_{batch_size}" in proc_data:
                        processes.append(p)
                        times.append(proc_data[f"batch_{batch_size}"]["time"])

            # Sort by processes for correct plotting
            sorted_data = sorted(zip(processes, times))
            processes, times = zip(*sorted_data)

            plt.plot(processes, times, marker='o', label=f'Batch Size {batch_size}')

        plt.xlabel("Number of Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Execution Time vs. Processes for {video_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{video_name}_execution_time.png"))
        plt.close()

        # Plot Speedup vs. Processes for each batch size
        plt.figure(figsize=(10, 6))
        for batch_size in sorted(batch_sizes):
            processes = []
            speedups = []
             # Include sequential results (processes=1, speedup=1)
            if batch_size == 1 and "sequential" in video_results:
                 processes.append(1)
                 speedups.append(1.0) # Speedup of sequential is 1

            for proc_key, proc_data in video_results.items():
                if proc_key.startswith("processes_"):
                    p = int(proc_key.split("_")[1])
                    if f"batch_{batch_size}" in proc_data:
                        processes.append(p)
                        speedups.append(proc_data[f"batch_{batch_size}"]["speedup"])

            # Sort by processes for correct plotting
            sorted_data = sorted(zip(processes, speedups))
            processes, speedups = zip(*sorted_data)

            plt.plot(processes, speedups, marker='o', label=f'Speedup (Batch Size {batch_size})')

        plt.xlabel("Number of Processes")
        plt.ylabel("Speedup Factor")
        plt.title(f"Speedup vs. Processes for {video_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"{video_name}_speedup.png"))
        plt.close()

        # Plot Similarity vs. Processes for each batch size
        plt.figure(figsize=(10, 6))
        for batch_size in sorted(batch_sizes):
            processes = []
            similarities = []
            # Include sequential results (processes=1, similarity=1)
            if batch_size == 1 and "sequential" in video_results:
                 processes.append(1)
                 similarities.append(1.0) # Sequential compared to itself is 1

            for proc_key, proc_data in video_results.items():
                if proc_key.startswith("processes_"):
                    p = int(proc_key.split("_")[1])
                    if f"batch_{batch_size}" in proc_data:
                        processes.append(p)
                        similarities.append(proc_data[f"batch_{batch_size}"]["similarity"])

            # Sort by processes for correct plotting
            sorted_data = sorted(zip(processes, similarities))
            processes, similarities = zip(*sorted_data)

            plt.plot(processes, similarities, marker='o', label=f'Batch Size {batch_size}')

        plt.xlabel("Number of Processes")
        plt.ylabel("Average Frame Similarity (SSIM)")
        plt.title(f"Quality (SSIM) vs. Processes for {video_name}")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05) # SSIM is between 0 and 1
        plt.savefig(os.path.join(plot_dir, f"{video_name}_similarity.png"))
        plt.close()

if __name__ == '__main__':
    unittest.main()