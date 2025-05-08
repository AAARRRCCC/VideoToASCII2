# Video to Japanese ASCII Art Converter

A Python application that converts videos to ASCII art animations using Japanese characters, specifically designed for Windows environments.

## Overview

This application takes input videos and transforms them into ASCII art animations using Japanese characters. The conversion process involves:

1. Downscaling the input video using ffmpeg
2. Converting each frame to black and white
3. Mapping pixel brightness to Japanese characters (denser characters for brighter pixels, dots for dark pixels)
4. Rendering the output at 30fps (or user-specified FPS)

## System Requirements

- Windows 10 or 11
- Python 3.10 or higher
- ffmpeg (installed and accessible in PATH)
- Sufficient disk space for video processing (at least 2GB free)
- RAM: Minimum 4GB recommended for processing standard videos

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
python main.py input_video.mp4 output_video.mp4 --width 160 --height 90 --fps 30 --font-size 14 --temp-dir .\temp_folder
```

### Parameters

- `input_path`: Path to the input video file (required)
- `output_path`: Path to save the output video (required)
- `--width`: Width of the ASCII output in characters (default: 120)
- `--height`: Height of the ASCII output in characters (default: 60)
- `--fps`: Frames per second of the output video (default: 30)
- `--font-size`: Font size for ASCII characters (default: 12)
- `--temp-dir`: Directory for temporary files (default: .\temp)

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

## License

[MIT License](LICENSE)