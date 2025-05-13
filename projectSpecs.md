# Video to Japanese ASCII Art Converter
## Design Document for Windows Environment

## 1. Project Overview

This project creates a Python 3.10 application specifically designed for Windows that converts input videos to ASCII art animations using Japanese characters. The application is optimized for performance on multi-core systems through parallel processing and preserves the aspect ratio of the input video during conversion.

The conversion process involves:
1. Downscaling the input video using ffmpeg while preserving the aspect ratio.
2. Converting each frame to black and white.
3. Mapping pixel brightness to Japanese characters (denser characters for brighter pixels, dots for dark pixels).
4. Rendering the output at the specified FPS (default: 30fps).
5. Utilizing parallel processing across multiple stages for faster conversion.

## 2. System Requirements

### 2.1 Software Dependencies
- Windows 10 or 11
- Python 3.10 or higher
- ffmpeg (installed and accessible in PATH)
- Required Python packages (listed in `requirements.txt`):
  - `opencv-python` (for video processing)
  - `numpy` (for array operations)
  - `pillow` (for image processing)
  - `tqdm` (for progress bars)
- Standard Python libraries used: `subprocess`, `argparse`, `os`, `sys`, `time`, `multiprocessing`, `concurrent.futures`, `datetime`, `shutil`, `tempfile`, `psutil`, `matplotlib`, `json`, `ctypes`, `skimage.metrics`.

### 2.2 Hardware Requirements
- Sufficient disk space for video processing (at least 2GB free)
- RAM: Minimum 4GB recommended for processing standard videos
- Multi-core CPU (for optimal parallel processing benefits)

## 3. Project Structure

```
VideoToASCII/
│
├── ascii_converter.py       # ASCII conversion logic
├── character_mapper.py      # Maps pixel values to Japanese characters
├── create_test_video.py     # Script to create test video files
├── main.py                  # Entry point for the application
├── monitor_resources.py     # Script for monitoring resource usage
├── projectSpecs.md          # Project design document
├── README.md                # Project documentation
├── renderer.py              # Renders and saves the final output
├── requirements.txt         # Project dependencies
├── test_requirements.txt    # Test dependencies
├── test_suite.py            # Comprehensive test suite
├── utils.py                 # Utility functions
├── video_processor.py       # Video processing and ffmpeg operations
│
├── custom_comparison_*/     # Directory for custom comparison outputs
├── debug_output/            # Directory for debug outputs
├── test_output/             # Directory for test outputs (including frames, plots, resources)
└── test_videos/             # Directory for test video files
```

## 4. Detailed Component Design

### 4.1 Command-line Interface (`main.py`)

The main entry point parses command-line arguments using `argparse` and orchestrates the video conversion process by initializing and calling methods on instances of `VideoProcessor`, `ASCIIConverter`, and `Renderer`. It handles input/output paths, temporary directory creation, and basic error checking (like ffmpeg installation).

Command-line arguments:
- `input_path` (required): Path to the input video file.
- `output_path` (required): Path to save the output video file.
- `--width` (int, default: 120): Maximum width of the ASCII output in characters. The aspect ratio of the original video is preserved.
- `--height` (int, default: 60): Maximum height of the ASCII output in characters. The aspect ratio of the original video is preserved.
- `--fps` (int, default: 30): Frames per second of the output video.
- `--font-size` (int, default: 12): Font size for ASCII characters in the rendered output video.
- `--temp-dir` (str, default: `.\temp`): Directory for temporary files created during the conversion process.
- `--processes` (int, default: number of CPU cores): Number of processes to use for parallel processing. If not specified, it defaults to the number of CPU cores. Set to 1 for sequential processing (equivalent to `--mode sequential`).
- `--batch-size` (int, default: 10): Number of frames to process in each batch during parallel processing. Adjusting this can impact memory usage and performance.
- `--compare` (flag): If included, creates a side-by-side comparison video of the original and ASCII versions. The comparison video is saved with `_comparison` appended to the output filename.
- `--mode` (str, choices: `sequential`, `parallel`, default: `parallel`): Processing mode. `parallel` is the default and recommended for performance on multi-core systems. `sequential` forces single-process execution.
- `--scale` (int, default: 1): Scaling factor for the resolution of the rendered ASCII output video. A scale of 2 will result in an output video with dimensions twice the calculated ASCII width and height.
- `--profile` (flag): If included, enables performance profiling during the conversion process using `cProfile` and `pstats`. Profiling results are printed to the console.

The script includes logic to select between sequential and parallel processing modes based on the `--mode` argument and initializes the core components (`VideoProcessor`, `ASCIIConverter`, `Renderer`) with the appropriate number of processes. It also handles the creation of the output directory and cleanup of temporary files.

### 4.2 Video Processor (`video_processor.py`)

This component handles video operations using ffmpeg and OpenCV (`cv2`). It includes functionality for downscaling videos, extracting frames, and creating side-by-side comparison videos. Parallel processing for frame extraction is implemented using `concurrent.futures.ProcessPoolExecutor`.

Key methods:
- `__init__(self, num_processes=None)`: Initializes the processor, validates ffmpeg installation, and sets the number of processes for parallel operations (defaults to CPU count if not specified).
- `_validate_ffmpeg(self)`: Internal method to check if ffmpeg is installed and accessible in the PATH.
- `downscale_video(self, input_path, output_path, width, height)`: Downscales the input video to the specified maximum width and height while preserving the aspect ratio using ffmpeg. Returns the path to the downscaled video and its actual dimensions.
- `extract_frames(self, video_path, batch_size=10)`: Extracts frames from the video as NumPy arrays using parallel processing with batching for memory management. Returns a list of frames and the video's FPS.
- `create_comparison_video(self, input_path, ascii_path, output_path, scale_factor)`: Creates a side-by-side comparison video by resizing the original and ASCII videos to a target resolution based on the original video and scale factor, and stacking them horizontally using ffmpeg. Audio from the original video is preserved.

### 4.3 ASCII Converter (`ascii_converter.py`)

This component is responsible for converting video frames (as NumPy arrays) into ASCII art representations. It utilizes parallel processing with batching for efficient conversion and memory management.

Key methods:
- `__init__(self, num_processes=None)`: Initializes the converter and the `CharacterMapper`, and sets the number of processes for parallel operations (defaults to CPU count if not specified).
- `convert_video_to_ascii(self, video_path, width, height, batch_size=10)`: Converts all frames of a video to ASCII art using parallel processing and batching. It also calculates the actual dimensions used for the ASCII art based on the video's aspect ratio and the target width/height. Returns a list of 2D lists representing the ASCII frames and the actual dimensions used.
- `convert_frame_to_ascii(self, frame, width, height)`: Converts a single video frame (NumPy array) to a 2D list of ASCII characters. It resizes the frame, converts it to grayscale, and maps pixel brightness to characters using the `CharacterMapper`. NumPy vectorization is used for pixel processing.

### 4.4 Character Mapper (`character_mapper.py`)

This component provides the logic for mapping grayscale pixel values to Japanese characters based on their perceived density or brightness.

Key methods:
- `__init__(self)`: Initializes the mapper with a predefined list of Japanese characters ordered from dense to sparse, a character for black pixels, and calculates brightness thresholds and steps.
- `map_pixel_to_character(self, pixel_value)`: Maps a given grayscale pixel value (0-255) to a corresponding Japanese character based on the defined thresholds and character list.

### 4.5 Renderer (`renderer.py`)

This component handles the rendering of ASCII art frames into images and then compiling these images into the final output video. Parallel processing is used for rendering individual frames.

Key methods:
- `__init__(self, font_size=12, fps=30, num_processes=None)`: Initializes the renderer with font size, FPS, and the number of processes for parallel rendering (defaults to CPU count if not specified). It also loads a suitable Japanese font.
- `_load_font(self)`: Internal method to load a Japanese font on Windows. It attempts to load several common Japanese fonts and falls back to a default if none are found.
- `render_ascii_frames(self, ascii_frames, output_path, actual_dimensions=None, batch_size=10)`: Renders a list of 2D ASCII frames into image files in a temporary directory using parallel processing with batching. It then compiles these images into the final output video using OpenCV.
- `_render_ascii_frame_to_image(self, ascii_frame, output_path, img_width, img_height)`: Internal method to render a single ASCII frame to an image file using the Pillow library (`PIL`).
- `_create_video_from_frames(self, frame_paths, output_path, width, height)`: Internal method to create a video file from a list of image file paths using OpenCV. It uses the `mp4v` codec for Windows compatibility.
- `render_ascii_frame_to_image_static(ascii_frame, output_path, img_width, img_height)`: A static method used by the `ProcessPoolExecutor` workers to render a single ASCII frame. It accesses the font loaded in the worker initializer.

### 4.6 Utility Functions (`utils.py`)

This file contains various helper functions used across the project.

Functions:
- `check_ffmpeg_installed()`: Checks if the `ffmpeg` command is available in the system's PATH by attempting to run `ffmpeg -version`. Returns `True` if successful, `False` otherwise.
- `create_directory_if_not_exists(directory_path)`: Creates the specified directory if it does not already exist. Includes basic error handling for `PermissionError`.
- `get_video_info(video_path)`: Uses `ffprobe` to extract basic information about a video file, such as width, height, duration, and frame rate. Returns a dictionary containing the video stream information.
- `is_admin()`: Checks if the current script is running with administrator privileges on Windows using `ctypes`. Returns `True` if running as admin, `False` otherwise.

### 4.7 Test Suite (`test_suite.py`)

This file contains a comprehensive test suite implemented using Python's built-in `unittest` framework. The tests cover various aspects of the application, including core functionality, error handling, and performance.

Test Cases (`TestVideoToASCII` class):
- `test_comparison()`: Verifies that the `--compare` flag correctly generates a side-by-side comparison video.
- `test_error_handling()`: Tests the application's robustness by providing invalid inputs (e.g., non-existent files, corrupted videos, invalid argument values) and asserting that appropriate errors are raised or handled.
- `test_performance()`: Evaluates the performance of the video conversion process under different configurations (varying number of processes and batch sizes). It measures execution time, calculates speedup compared to sequential processing, and assesses output quality using the Structural Similarity Index (SSIM).
- `test_profiling()`: Checks that the `--profile` command-line flag in `main.py` executes without errors and enables performance profiling.

Helper methods within the test class:
- `run_command(cmd)`: Executes a given command using `subprocess.run` and returns the result, including stdout and stderr.
- `corrupt_video(input_video, output_video)`: Creates a corrupted copy of a video file by truncating its content.
- `run_conversion(input_video, output_video, processes, batch_size, width=120, height=60)`: A helper to run the main application script with specified parameters and return execution time and basic resource statistics (exit code, stdout, stderr).
- `compare_images(image1_path, image2_path)`: Calculates the SSIM between two image files.
- `extract_frames(video_path, output_dir, num_frames=5)`: Extracts a specified number of frames from a video and saves them as image files.

Additional functions:
- `plot_results(results, batch_sizes)`: (Note: This function is present but may not execute correctly in all environments due to reliance on a graphical backend for `matplotlib`. It is intended to generate plots of performance test results.)

### 4.8 Test Video Creation (`create_test_video.py`)

This script provides a simple way to generate synthetic video files for testing purposes.

Functions:
- `create_test_video(output_path, width=640, height=480, duration=5, fps=30)`: Creates an MP4 video file at the specified `output_path` with moving colored shapes and a frame counter. This is useful for generating consistent input for testing the ASCII conversion and rendering logic.

### 4.9 Resource Monitoring (`monitor_resources.py`)

This script is a standalone utility for monitoring the CPU and memory usage of another process. It is used by the performance tests in `test_suite.py` to gather resource usage data.

Functions:
- `monitor_resources(command, interval=0.5, output_dir="test_output/resources")`: Executes the specified `command` as a subprocess and periodically samples its CPU and memory usage using the `psutil` library. The collected data is returned as a dictionary.
- `plot_resource_usage(data, title, output_path)`: Generates and saves a plot of the collected resource usage data (CPU percentage and memory usage over time) using `matplotlib`.

## License

[MIT License](LICENSE)