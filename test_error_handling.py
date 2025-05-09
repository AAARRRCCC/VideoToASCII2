import os
import subprocess
import argparse
import shutil
import tempfile
import time

def test_error_handling(test_name, input_video, processes, batch_size, expected_error=None):
    """
    Test error handling in the parallel implementation.
    
    Args:
        test_name (str): Name of the test
        input_video (str): Path to input video
        processes (int): Number of processes to use
        batch_size (int): Batch size for processing
        expected_error (str, optional): Expected error message
        
    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"\n=== Running Error Test: {test_name} ===")
    print(f"Processes: {processes}, Batch Size: {batch_size}")
    
    # Create temporary output directory
    output_dir = os.path.join("test_output", "error_tests", test_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output path
    output_video = os.path.join(output_dir, "output.mp4")
    
    # Prepare command
    cmd = [
        "python", "main.py",
        input_video, output_video,
        "--processes", str(processes),
        "--batch-size", str(batch_size)
    ]
    
    # Run the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    execution_time = time.time() - start_time
    
    # Save output
    with open(os.path.join(output_dir, "stdout.txt"), "w") as f:
        f.write(result.stdout)
    
    with open(os.path.join(output_dir, "stderr.txt"), "w") as f:
        f.write(result.stderr)
    
    # Check result
    if expected_error is None:
        # Expect success
        if result.returncode == 0:
            print(f"✅ Test passed: Command completed successfully in {execution_time:.2f} seconds")
            return True
        else:
            print(f"❌ Test failed: Command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
    else:
        # Expect failure with specific error
        if result.returncode != 0 and expected_error in result.stderr:
            print(f"✅ Test passed: Command failed as expected with error: {expected_error}")
            return True
        elif result.returncode != 0:
            print(f"❓ Test partially passed: Command failed but with unexpected error")
            print(f"Expected: {expected_error}")
            print(f"Actual: {result.stderr}")
            return False
        else:
            print(f"❌ Test failed: Command succeeded but was expected to fail with: {expected_error}")
            return False

def corrupt_video(input_video, output_video):
    """
    Create a corrupted copy of a video file.
    
    Args:
        input_video (str): Path to input video
        output_video (str): Path to output corrupted video
        
    Returns:
        str: Path to corrupted video
    """
    # Copy the video
    shutil.copy2(input_video, output_video)
    
    # Corrupt the file by truncating it
    with open(output_video, "r+b") as f:
        # Get file size
        f.seek(0, 2)
        size = f.tell()
        
        # Truncate to 80% of original size
        f.truncate(int(size * 0.8))
    
    return output_video

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test error handling in parallel implementation")
    parser.add_argument("--input", default="test_videos/small.mp4", help="Input video path")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("test_output/error_tests", exist_ok=True)
    
    # Prepare corrupted video
    corrupted_video = "test_output/error_tests/corrupted.mp4"
    if os.path.exists(args.input):
        corrupt_video(args.input, corrupted_video)
    else:
        print(f"Error: Input video not found: {args.input}")
        print("Please run create_test_video.py first to generate test videos")
        return
    
    # Define tests
    tests = [
        # Test with non-existent input file
        {
            "name": "non_existent_input",
            "input": "non_existent_file.mp4",
            "processes": 4,
            "batch_size": 10,
            "expected_error": "Input video file not found"
        },
        # Test with corrupted input file
        {
            "name": "corrupted_input",
            "input": corrupted_video,
            "processes": 4,
            "batch_size": 10,
            "expected_error": "Error"  # Generic error expected
        },
        # Test with invalid process count (negative)
        {
            "name": "negative_processes",
            "input": args.input,
            "processes": -1,
            "batch_size": 10,
            "expected_error": "Error"  # Some error expected
        },
        # Test with invalid batch size (negative)
        {
            "name": "negative_batch_size",
            "input": args.input,
            "processes": 4,
            "batch_size": -1,
            "expected_error": "Error"  # Some error expected
        },
        # Test with zero processes
        {
            "name": "zero_processes",
            "input": args.input,
            "processes": 0,
            "batch_size": 10,
            "expected_error": "Error"  # Some error expected
        },
        # Test with zero batch size
        {
            "name": "zero_batch_size",
            "input": args.input,
            "processes": 4,
            "batch_size": 0,
            "expected_error": "Error"  # Some error expected
        },
        # Test with very large batch size
        {
            "name": "large_batch_size",
            "input": args.input,
            "processes": 4,
            "batch_size": 1000,
            "expected_error": None  # Should succeed
        },
        # Test with very large process count
        {
            "name": "large_process_count",
            "input": args.input,
            "processes": 100,
            "batch_size": 10,
            "expected_error": None  # Should succeed but might be inefficient
        }
    ]
    
    # Run tests
    results = {}
    for test in tests:
        result = test_error_handling(
            test["name"],
            test["input"],
            test["processes"],
            test["batch_size"],
            test["expected_error"]
        )
        results[test["name"]] = result
    
    # Print summary
    print("\n=== Error Handling Test Summary ===")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    print(f"Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed < total:
        print("\nFailed tests:")
        for name, result in results.items():
            if not result:
                print(f"- {name}")

if __name__ == "__main__":
    main()