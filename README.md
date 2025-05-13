# Video to Japanese ASCII Art Converter

A Python application that converts videos to ASCII art animations using Japanese characters, specifically designed for Windows environments.

## Overview

This application converts videos to ASCII art animations using Japanese characters, optimized for Windows environments. It utilizes parallel processing to significantly speed up the conversion.

The conversion process involves:

1. Downscaling the input video using ffmpeg while preserving the aspect ratio.
2. Converting each frame to black and white.
3. Mapping pixel brightness to Japanese characters (denser characters for brighter pixels, dots for dark pixels).
4. Rendering the output at the specified FPS (default: 30fps).
5. Utilizing parallel processing across multiple stages for faster conversion (achieving significant speedup on multi-core systems).

## System Requirements

- Windows 10 or 11
- Python 3.10 or higher
- ffmpeg (installed and accessible in PATH)
- Sufficient disk space for video processing (at least 2GB free)
- RAM: Minimum 4GB recommended for processing standard videos
- Multi-core CPU (for optimal parallel processing benefits)

## Installation

### Step 1: Install Python

1. Install Python 3.10 or higher from the Microsoft Store or [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Also check "Install pip"

### Step 2: Install ffmpeg

1. Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) (specifically the Windows builds)
2. Extract the ZIP file to a location like C:\ffmpeg
3. Add the bin folder (e.g., C:\ffmpeg\bin) to your PATH environment variable:
   1. Search for "Environment Variables" in the Start Menu
   2. Click "Edit the system environment variables"
   3. Click "Environment Variables" button
   4. Under "System variables", find and select "Path", then click "Edit"
   5. Click "New" and add the path to the bin folder (e.g., C:\ffmpeg\bin)
   6. Click "OK" on all dialogs to save changes

### Step 3: Install the Application

1. Clone or download this repository
2. Open Command Prompt in the project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python main.py input_video.mp4 output_video.mp4
```

### Advanced Options

```bash
python main.py input_video.mp4 output_video.mp4 --width 160 --height 90 --fps 30 --font-size 14 --temp-dir .\temp_folder --processes 4 --batch-size 20 --mode parallel --scale 2 --profile
```

### Parameters

- `input_path`: Path to the input video file (required)
- `output_path`: Path to save the output video (required)
- `--width`: Maximum width of the ASCII output in characters. The aspect ratio of the original video will be preserved. (default: 120)
- `--height`: Maximum height of the ASCII output in characters. The aspect ratio of the original video will be preserved. (default: 60)
- `--fps`: Frames per second of the output video. (default: 30)
- `--font-size`: Font size for ASCII characters in the rendered output video. (default: 12)
- `--temp-dir`: Directory for temporary files created during the conversion process. (default: `.\temp`)
- `--processes`: Number of processes to use for parallel processing. If not specified, it defaults to the number of CPU cores. Set to 1 for sequential processing (equivalent to `--mode sequential`).
- `--batch-size`: Number of frames to process in each batch during parallel processing. Adjusting this can impact memory usage and performance. (default: 10)
- `--compare`: Include this flag to create a side-by-side comparison video of the original and ASCII versions. The comparison video will be saved with `_comparison` appended to the output filename.
- `--mode`: Processing mode. Choose between `sequential` and `parallel`. `parallel` is the default and recommended for performance on multi-core systems. (default: `parallel`)
- `--scale`: Scaling factor for the resolution of the rendered ASCII output video. A scale of 2 will result in an output video with dimensions twice the calculated ASCII width and height. (default: 1)
- `--profile`: Include this flag to enable performance profiling during the conversion process. Profiling results will be printed to the console.

## Examples

### Convert a video with default settings

```bash
python main.py my_video.mp4 ascii_output.mp4
```

### Create a higher resolution ASCII video (adjusting width and height while preserving aspect ratio)

```bash
python main.py my_video.mp4 high_res_ascii.mp4 --width 200 --height 100 --font-size 8
```

### Create a slower playback ASCII video

```bash
python main.py my_video.mp4 slow_ascii.mp4 --fps 15
```

### Optimize for performance using parallel processing

```bash
python main.py my_video.mp4 optimized_ascii.mp4 --processes 4 --batch-size 10
```

### Process a large video with adjusted batch size for memory

```bash
python main.py large_video.mp4 large_ascii.mp4 --batch-size 50
```

### Force sequential processing mode

```bash
python main.py my_video.mp4 sequential_ascii.mp4 --mode sequential
```

### Render ASCII output at a higher resolution scale

```bash
python main.py my_video.mp4 scaled_ascii.mp4 --scale 2
```

### Create a side-by-side comparison video

```bash
python main.py my_video.mp4 ascii_output.mp4 --compare
```

This will create both the ASCII video and a side-by-side comparison video (named `*_comparison.mp4`) that shows the original video alongside the ASCII version.

### Enable performance profiling

```bash
python main.py my_video.mp4 profiled_ascii.mp4 --profile
```

## Troubleshooting

### Common Issues

1. **"'ffmpeg' is not recognized as an internal or external command"**:
   - ffmpeg is not properly added to your PATH
   - Solution: Follow the installation steps to add ffmpeg to PATH
   - Alternatively, restart your Command Prompt or computer after adding to PATH

2. **"Japanese characters display as squares or question marks"**:
   - Missing Japanese font support on Windows or the application could not load a suitable font.
   - Solution: Install a Japanese language pack in Windows Settings > Time & Language > Language & Region > Add a language. The application attempts to load several common Japanese fonts as detailed in the Technical Implementation Details.
3. **"Access is denied" errors**:
   - Windows permission issues
   - Solution: Run Command Prompt as Administrator or change to a directory where you have write permissions
4. **"The process cannot access the file because it is being used by another process"**:
   - Windows file locking issue
   - Solution: Close any applications that might be using the file, or restart the computer
5. **"Unicode characters not displaying in Command Prompt"**:
   - Windows Command Prompt Unicode configuration issue
   - Solution: Change Command Prompt font to a Unicode-compatible font, or use Windows Terminal instead

### Performance Tips

1. For better performance, close other applications while processing large videos
2. Consider adding exclusions for your temp folder in Windows Defender
3. Process videos on an SSD rather than an HDD if possible
4. For very large videos, consider reducing the input video resolution before processing
5. Adjust the number of processes (`--processes`) based on your CPU:
   - For best performance, set it to about half of your available CPU cores. The application defaults to using all available cores if `--processes` is not specified.
   - Using too many processes (more than available cores) can decrease performance due to context switching overhead.
   - For systems with limited RAM, use fewer processes to avoid memory issues.
6. Adjust the batch size (`--batch-size`) based on your available memory:
   - Larger batch sizes can improve performance but require more memory.
   - For systems with limited RAM, use smaller batch sizes (5-10).
   - For high-end systems, larger batch sizes (20-50) may improve performance.

### Parallelism Implementation

The application uses parallel processing (`--mode parallel`) to significantly speed up video conversion. This is the default mode.

1. **Frame-Level Parallelism**: Each stage of the pipeline processes multiple frames simultaneously:
   - ASCII conversion: Converting video frames to ASCII art
   - Frame rendering: Rendering ASCII frames to images
   - Frame extraction: Reading frames from video files

2. **Batch Processing**: Frames are processed in batches (`--batch-size`) to efficiently manage memory usage while maintaining performance.

3. **Performance Benefits**:
   - Achieves significant speedup on multi-core systems compared to sequential processing (`--mode sequential`).
   - Performance scales with the number of CPU cores (up to a point).
   - No quality compromise - parallel processing produces identical output to sequential processing.

### Recommended Configurations

For optimal performance, it is generally recommended to use the default parallel processing mode and experiment with the `--processes` and `--batch-size` parameters based on your system's CPU and RAM.

*   **`--processes`**: A good starting point is half the number of available CPU cores.
*   **`--batch-size`**: Start with the default (10) and increase for systems with more RAM, or decrease for systems with limited memory.

## Technical Implementation Details

For developers interested in understanding or extending the code, here's how parallelism is implemented in the VideoToASCII converter:

### Parallel Processing Architecture

The application uses Python's `multiprocessing` library with `ProcessPoolExecutor` to implement parallelism across multiple components:

1. **Video Processor (`video_processor.py`)**:
   - Implements parallel frame extraction using process pools.
   - Divides frame extraction tasks into batches for efficient memory management.
   - Maintains frame order using pre-allocated result lists.
   - Includes logic for preserving aspect ratio during downscaling.
   - Provides a method (`create_comparison_video`) for generating side-by-side comparison videos using ffmpeg.

2. **ASCII Converter (`ascii_converter.py`)**:
   - Converts video frames to ASCII art in parallel.
   - Uses batch processing to manage memory efficiently.
   - Implements NumPy vectorization for pixel processing.
   - Preserves frame order using indexed results.
   - Includes logic for preserving aspect ratio during ASCII conversion.

3. **Character Mapper (`character_mapper.py`)**:
   - Maps pixel brightness to Japanese characters based on defined thresholds.

4. **Renderer (`renderer.py`)**:
   - Renders ASCII frames to images in parallel.
   - Uses a static rendering method (`render_ascii_frame_to_image_static`) and a worker initializer (`_worker_init`) for compatibility with `ProcessPoolExecutor`.
   - Processes frames in batches to manage memory usage.
   - Combines rendered frames into the final video using OpenCV and the `mp4v` codec for Windows compatibility.
   - Includes a font loading mechanism that attempts several common Japanese fonts on Windows.

### Utility Functions (`utils.py`)

The `utils.py` file contains several helper functions:

- `check_ffmpeg_installed()`: Checks if ffmpeg is installed and accessible in the system's PATH.
- `create_directory_if_not_exists(directory_path)`: Creates a directory if it does not already exist, handling potential permission errors.
- `get_video_info(video_path)`: Retrieves basic information about a video file using ffprobe.
- `is_admin()`: Checks if the script is currently running with administrator privileges on Windows.

### Test Suite (`test_suite.py`)

The `test_suite.py` file contains a comprehensive test suite using the `unittest` framework to verify the application's functionality and performance. Key tests include:

- `test_comparison()`: Verifies the functionality of the side-by-side comparison video creation.
- `test_error_handling()`: Tests how the application handles various error conditions, such as non-existent or corrupted input files and invalid argument values.
- `test_performance()`: Runs performance tests with different configurations (number of processes, batch size) and measures execution time, speedup, and output quality (using SSIM).
- `test_profiling()`: Verifies that the `--profile` flag runs without errors and produces profiling output.

Helper functions within `test_suite.py` include:

- `run_command(cmd)`: Executes a given command and captures its output.
- `corrupt_video(input_video, output_video)`: Creates a corrupted copy of a video file for testing error handling.
- `run_conversion(input_video, output_video, processes, batch_size, width=120, height=60)`: Runs the main conversion process with specified parameters and returns execution time and resource statistics.
- `compare_images(image1_path, image2_path)`: Compares two images using the Structural Similarity Index (SSIM).
- `extract_frames(video_path, output_dir, num_frames=5)`: Extracts a specified number of frames from a video for quality comparison.

### Test Video Creation (`create_test_video.py`)

The `create_test_video.py` script is used to generate synthetic test video files with moving shapes. This is useful for consistent testing of the ASCII conversion and rendering process.

- `create_test_video(output_path, width=640, height=480, duration=5, fps=30)`: Creates a test video with specified dimensions, duration, and frame rate.

### Resource Monitoring (`monitor_resources.py`)

The `monitor_resources.py` script is a utility for monitoring the CPU and memory usage of a command while it is executing. This is used in conjunction with the performance tests to gather resource usage data.

- `monitor_resources(command, interval=0.5, output_dir="test_output/resources")`: Executes a command and monitors its resource usage at a specified interval, saving the data to a file.
- `plot_resource_usage(data, title, output_path)`: Generates a plot of the collected CPU and memory usage data over time.

## License

[MIT License](LICENSE)