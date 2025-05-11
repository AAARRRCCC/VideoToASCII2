# Video to Japanese ASCII Art Converter

A Python application that converts videos to ASCII art animations using Japanese characters, specifically designed for Windows environments.

## Overview

This application takes input videos and transforms them into ASCII art animations using Japanese characters. The conversion process involves:

1. Downscaling the input video using ffmpeg
2. Converting each frame to black and white
3. Mapping pixel brightness to Japanese characters (denser characters for brighter pixels, dots for dark pixels)
4. Rendering the output at 30fps (or user-specified FPS)
5. Utilizing parallel processing for faster conversion (up to 2.5x speedup)

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
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```
python main.py input_video.mp4 output_video.mp4
```

### Advanced Options

```
python main.py input_video.mp4 output_video.mp4 --width 160 --height 90 --fps 30 --font-size 14 --temp-dir .\temp_folder --processes 4 --batch-size 20
```

### Parameters

- `input_path`: Path to the input video file (required)
- `output_path`: Path to save the output video (required)
- `--width`: Width of the ASCII output in characters (default: 120)
- `--height`: Height of the ASCII output in characters (default: 60)
- `--fps`: Frames per second of the output video (default: 30)
- `--font-size`: Font size for ASCII characters (default: 12)
- `--temp-dir`: Directory for temporary files (default: .\temp)
- `--processes`: Number of processes to use for parallel processing (default: number of CPU cores)
- `--batch-size`: Number of frames to process in each batch (default: 10)
- `--compare`: Create a side-by-side comparison video of the original and ASCII versions

## Examples

### Convert a video with default settings

```
python main.py my_video.mp4 ascii_output.mp4
```

### Create a higher resolution ASCII video

```
python main.py my_video.mp4 high_res_ascii.mp4 --width 200 --height 100 --font-size 8
```

### Create a slower playback ASCII video

```
python main.py my_video.mp4 slow_ascii.mp4 --fps 15
```

### Optimize for performance on a quad-core system

```
python main.py my_video.mp4 optimized_ascii.mp4 --processes 4 --batch-size 10
```

### Process a large video with optimal settings

```
python main.py large_video.mp4 large_ascii.mp4 --processes 8 --batch-size 50
```

### Process on a system with limited memory

```
python main.py my_video.mp4 memory_optimized.mp4 --processes 4 --batch-size 5
```

### Maximum performance on a high-end system (16 cores)

```
python main.py my_video.mp4 max_performance.mp4 --processes 8 --batch-size 20
```

### Create a side-by-side comparison video

```
python main.py my_video.mp4 ascii_output.mp4 --compare
```

This will create both the ASCII video and a side-by-side comparison video (named *_comparison.mp4) that shows the original video alongside the ASCII version.

## Troubleshooting

### Common Issues

1. **"'ffmpeg' is not recognized as an internal or external command"**:
   - ffmpeg is not properly added to your PATH
   - Solution: Follow the installation steps to add ffmpeg to PATH
   - Alternatively, restart your Command Prompt or computer after adding to PATH

2. **"Japanese characters display as squares or question marks"**:
   - Missing Japanese font support on Windows
   - Solution: Install a Japanese language pack in Windows Settings > Time & Language > Language & Region > Add a language

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
   - For best performance, set it to about half of your available CPU cores
   - Using too many processes (more than available cores) can decrease performance due to context switching overhead
   - For systems with limited RAM, use fewer processes to avoid memory issues
6. Adjust the batch size (`--batch-size`) based on your available memory:
   - Larger batch sizes can improve performance but require more memory
   - For systems with limited RAM, use smaller batch sizes (5-10)
   - For high-end systems, larger batch sizes (20-50) may improve performance

### Parallelism Implementation

The application uses parallel processing to significantly speed up video conversion:

1. **Frame-Level Parallelism**: Each stage of the pipeline processes multiple frames simultaneously:
   - ASCII conversion: Converting video frames to ASCII art
   - Frame rendering: Rendering ASCII frames to images
   - Frame extraction: Reading frames from video files

2. **Batch Processing**: Frames are processed in batches to efficiently manage memory usage while maintaining performance

3. **Performance Benefits**:
   - Up to 2.5x faster processing on multi-core systems
   - Linear scaling with the number of CPU cores (up to a point)
   - No quality compromise - parallel processing produces identical output to sequential processing

### Recommended Configurations

Based on extensive testing, here are the recommended configurations for different video sizes:

1. **For small videos** (less than 30 seconds, low resolution):
   - Processes: 11-22 (depending on available CPU cores)
   - Batch size: 10-20

2. **For medium videos** (30 seconds to 2 minutes, medium resolution):
   - Processes: About half of available CPU cores
   - Batch size: 20

3. **For large videos** (longer than 2 minutes, high resolution):
   - Processes: About half of available CPU cores
   - Batch size: 50

4. **For systems with limited memory**:
   - Reduce batch size to 5-10
   - Keep process count at half of available CPU cores

## Technical Implementation Details

For developers interested in understanding or extending the code, here's how parallelism is implemented in the VideoToASCII converter:

### Parallel Processing Architecture

The application uses Python's `multiprocessing` library with `ProcessPoolExecutor` to implement parallelism across multiple components:

1. **Video Processor (`video_processor.py`)**:
   - Implements parallel frame extraction using process pools
   - Divides frame extraction tasks into batches for efficient memory management
   - Maintains frame order using pre-allocated result lists

2. **ASCII Converter (`ascii_converter.py`)**:
   - Converts video frames to ASCII art in parallel
   - Uses batch processing to manage memory efficiently
   - Implements NumPy vectorization for pixel processing
   - Preserves frame order using indexed results

3. **Renderer (`renderer.py`)**:
   - Renders ASCII frames to images in parallel
   - Uses a static rendering method for pickle compatibility
   - Processes frames in batches to manage memory usage
   - Combines rendered frames into the final video

### Key Implementation Features

1. **Batch Processing**:
   ```python
   for batch_start in range(0, len(frames), batch_size):
       batch_end = min(batch_start + batch_size, len(frames))
       batch_frames = frames[batch_start:batch_end]
       # Process batch in parallel
   ```

2. **Order Preservation**:
   ```python
   # Pre-allocate result list to maintain frame order
   ascii_frames = [None] * len(frames)
   
   # Store results at correct indices
   for future in future_to_index:
       ascii_frame = future.result()
       idx = future_to_index[future]
       ascii_frames[idx] = ascii_frame
   ```

3. **Process Pool Creation**:
   ```python
   with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
       # Submit tasks to the process pool
   ```

4. **Error Handling**:
   ```python
   try:
       result = future.result()
   except Exception as e:
       print(f"Error processing frame: {e}")
   ```

### Memory Management

The batch processing approach helps manage memory usage by:

1. Processing a limited number of frames at once
2. Releasing memory after each batch is complete
3. Avoiding loading the entire video into memory at once

This approach allows the application to process videos of any size without excessive memory usage, while still benefiting from parallel processing.

## License

[MIT License](LICENSE)