# Video to Japanese ASCII Art Converter
## Design Document for Windows Environment

## 1. Project Overview

This project creates a Python 3.10 application specifically designed for Windows that converts input videos to ASCII art animations using Japanese characters. The conversion process involves:
1. Downscaling the input video using ffmpeg
2. Converting each frame to black and white
3. Mapping pixel brightness to Japanese characters (denser characters for brighter pixels, dots for dark pixels)
4. Rendering the output at 30fps

## 2. System Requirements

### 2.1 Software Dependencies
- Windows 10 or 11
- Python 3.10 or higher
- ffmpeg (installed and accessible in PATH)
- Required Python packages:
  - `opencv-python` (for video processing)
  - `numpy` (for array operations)
  - `pillow` (for image processing)
  - `subprocess` (for calling ffmpeg)
  - `argparse` (for command-line argument parsing)
  - `tqdm` (for progress bars)

### 2.2 Hardware Requirements
- Sufficient disk space for video processing (at least 2GB free)
- RAM: Minimum 4GB recommended for processing standard videos
- Windows-compatible CPU (Intel Core i3 or higher recommended)

## 3. Project Structure

```
ascii_video_converter/
│
├── main.py                  # Entry point for the application
├── video_processor.py       # Video processing and ffmpeg operations
├── ascii_converter.py       # ASCII conversion logic
├── character_mapper.py      # Maps pixel values to Japanese characters
├── renderer.py              # Renders and saves the final output
├── utils.py                 # Utility functions
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 4. Detailed Component Design

### 4.1 Command-line Interface (main.py)

The main entry point will parse command-line arguments and orchestrate the conversion process.

```python
import argparse
import os
import sys
from video_processor import VideoProcessor
from ascii_converter import ASCIIConverter
from renderer import Renderer
from utils import check_ffmpeg_installed, create_directory_if_not_exists

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert video to Japanese ASCII art')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to output video file')
    parser.add_argument('--width', type=int, default=120, help='Width of ASCII output in characters')
    parser.add_argument('--height', type=int, default=60, help='Height of ASCII output in characters')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second of output video')
    parser.add_argument('--font-size', type=int, default=12, help='Font size for ASCII characters')
    parser.add_argument('--temp-dir', type=str, default='.\\temp', help='Directory for temporary files')
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
    args.input_path = args.input_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')
    args.temp_dir = args.temp_dir.replace('\\', '/')
    
    # Create temporary directory if it doesn't exist
    create_directory_if_not_exists(args.temp_dir)
    
    try:
        # Initialize components
        video_processor = VideoProcessor()
        ascii_converter = ASCIIConverter()
        renderer = Renderer(font_size=args.font_size, fps=args.fps)
        
        # Process video
        print(f"Processing video: {args.input_path}")
        downscaled_video = video_processor.downscale_video(
            args.input_path, 
            os.path.join(args.temp_dir, "downscaled.mp4"),
            args.width,
            args.height
        )
        
        # Convert to ASCII frames
        print("Converting to ASCII art...")
        ascii_frames = ascii_converter.convert_video_to_ascii(
            downscaled_video,
            args.width,
            args.height
        )
        
        # Render and save output
        print(f"Rendering output to: {args.output_path}")
        renderer.render_ascii_frames(ascii_frames, args.output_path)
        
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
```

### 4.2 Video Processor (video_processor.py)

This component handles video operations using ffmpeg, with Windows-specific path handling.

```python
import subprocess
import os
import cv2
import sys

class VideoProcessor:
    def __init__(self):
        self._validate_ffmpeg()
    
    def _validate_ffmpeg(self):
        """Validate that ffmpeg is installed and accessible."""
        try:
            # Use shell=True for Windows command execution
            subprocess.run('ffmpeg -version', check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("ffmpeg is not installed or not accessible in PATH")
    
    def downscale_video(self, input_path, output_path, width, height):
        """
        Downscale the input video using ffmpeg.
        
        Args:
            input_path (str): Path to input video file
            output_path (str): Path to save downscaled video
            width (int): Target width
            height (int): Target height
            
        Returns:
            str: Path to downscaled video
        """
        # Ensure input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Create ffmpeg command for downscaling with Windows-safe quoting
        cmd = (
            f'ffmpeg -i "{input_path}" '
            f'-vf "scale={width}:{height}" '
            f'-c:v libx264 -crf 23 -preset fast -y "{output_path}"'
        )
        
        # Execute ffmpeg command
        try:
            # Use shell=True for Windows command execution
            process = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, shell=True)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Error downscaling video: {error_message}")
        
        return output_path
    
    def extract_frames(self, video_path):
        """
        Extract frames from the video as numpy arrays.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            list: List of frames as numpy arrays
            float: FPS of the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, fps
```

### 4.3 ASCII Converter (ascii_converter.py)

This component handles the conversion of video frames to ASCII art.

```python
import numpy as np
import cv2
from character_mapper import CharacterMapper
from tqdm import tqdm

class ASCIIConverter:
    def __init__(self):
        self.character_mapper = CharacterMapper()
    
    def convert_video_to_ascii(self, video_path, width, height):
        """
        Convert video frames to ASCII art.
        
        Args:
            video_path (str): Path to video file
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            
        Returns:
            list: List of ASCII art frames
        """
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ascii_frames = []
        
        for _ in tqdm(range(total_frames), desc="Converting frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to ASCII
            ascii_frame = self.convert_frame_to_ascii(frame, width, height)
            ascii_frames.append(ascii_frame)
        
        cap.release()
        return ascii_frames
    
    def convert_frame_to_ascii(self, frame, width, height):
        """
        Convert a single video frame to ASCII art.
        
        Args:
            frame (numpy.ndarray): Input video frame
            width (int): Width of ASCII output in characters
            height (int): Height of ASCII output in characters
            
        Returns:
            list: 2D list of ASCII characters
        """
        # Resize frame to match target dimensions
        resized_frame = cv2.resize(frame, (width, height))
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Create ASCII frame
        ascii_frame = []
        for y in range(height):
            ascii_row = []
            for x in range(width):
                pixel_value = gray_frame[y, x]
                character = self.character_mapper.map_pixel_to_character(pixel_value)
                ascii_row.append(character)
            ascii_frame.append(ascii_row)
        
        return ascii_frame
```

### 4.4 Character Mapper (character_mapper.py)

This component maps pixel brightness to Japanese characters.

```python
class CharacterMapper:
    def __init__(self):
        # Define a list of Japanese characters from dense to sparse
        # The order should be from characters that appear densest/brightest to least dense/darkest
        self.japanese_chars = [
            'あ', 'い', 'う', 'え', 'お', 
            'か', 'き', 'く', 'け', 'こ', 
            'さ', 'し', 'す', 'せ', 'そ', 
            'た', 'ち', 'つ', 'て', 'と', 
            'な', 'に', 'ぬ', 'ね', 'の', 
            'は', 'ひ', 'ふ', 'へ', 'ほ', 
            'ま', 'み', 'む', 'め', 'も', 
            'や', 'ゆ', 'よ', 
            'ら', 'り', 'る', 'れ', 'ろ', 
            'わ', 'を', 'ん', 
            'ア', 'イ', 'ウ', 'エ', 'オ', 
            'カ', 'キ', 'ク', 'ケ', 'コ', 
            'サ', 'シ', 'ス', 'セ', 'ソ', 
            'タ', 'チ', 'ツ', 'テ', 'ト', 
            'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 
            'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 
            'マ', 'ミ', 'ム', 'メ', 'モ', 
            'ヤ', 'ユ', 'ヨ', 
            'ラ', 'リ', 'ル', 'レ', 'ロ', 
            'ワ', 'ヲ', 'ン', 
            '゛', '゜', 'ー', '・', '「', '」', '（', '）'
        ]
        
        # Use a black dot for the darkest pixels
        self.black_char = '・'
        
        # Calculate brightness thresholds
        self.black_threshold = 30  # Pixels below this value will be represented by a black dot
        self.brightness_levels = len(self.japanese_chars)
        self.brightness_step = (255 - self.black_threshold) / self.brightness_levels
    
    def map_pixel_to_character(self, pixel_value):
        """
        Map a pixel value (0-255) to a Japanese character.
        
        Args:
            pixel_value (int): Grayscale pixel value (0-255)
            
        Returns:
            str: Japanese character representing the pixel brightness
        """
        # Black pixels get a dot
        if pixel_value <= self.black_threshold:
            return self.black_char
        
        # Map other pixel values to Japanese characters
        adjusted_value = pixel_value - self.black_threshold
        char_index = min(
            int(adjusted_value / self.brightness_step),
            len(self.japanese_chars) - 1
        )
        
        return self.japanese_chars[char_index]
```

### 4.5 Renderer (renderer.py)

This component renders ASCII frames and creates the output video, with Windows-specific font handling.

```python
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

class Renderer:
    def __init__(self, font_size=12, fps=30):
        self.font_size = font_size
        self.fps = fps
        self._load_font()
    
    def _load_font(self):
        """
        Load a Japanese font for rendering on Windows.
        Tries several common Japanese fonts and falls back to a default if none are available.
        """
        # List of Japanese fonts to try on Windows, in order of preference
        font_paths = [
            "C:\\Windows\\Fonts\\msgothic.ttc",   # MS Gothic (common Japanese font on Windows)
            "C:\\Windows\\Fonts\\YuGothR.ttc",    # Yu Gothic Regular
            "C:\\Windows\\Fonts\\meiryo.ttc",     # Meiryo
            "C:\\Windows\\Fonts\\meiryob.ttc",    # Meiryo Bold
            "C:\\Windows\\Fonts\\msmincho.ttc",   # MS Mincho
            "C:\\Windows\\Fonts\\yugothib.ttf",   # Yu Gothic Bold
            "C:\\Windows\\Fonts\\malgun.ttf"      # Malgun Gothic (Korean, but has some Japanese support)
        ]
        
        # Try to load each font until one succeeds
        self.font = None
        loaded_font_path = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.font = ImageFont.truetype(font_path, self.font_size)
                    loaded_font_path = font_path
                    break
                except IOError:
                    continue
        
        # If no font was loaded, try to use a default font
        if self.font is None:
            try:
                # Try to use Arial as a last resort (though it won't display Japanese properly)
                self.font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", self.font_size)
                print("Warning: Could not load a Japanese font. Using Arial instead.")
                print("Japanese characters may not display properly.")
            except IOError:
                # Last resort: use default
                self.font = ImageFont.load_default()
                print("Warning: Could not load any font. Using default font.")
                print("Japanese characters may not display properly.")
        else:
            print(f"Using font: {os.path.basename(loaded_font_path)}")
    
    def render_ascii_frames(self, ascii_frames, output_path):
        """
        Render ASCII frames to an output video.
        
        Args:
            ascii_frames (list): List of 2D ASCII frames
            output_path (str): Path to save the output video
        """
        if not ascii_frames:
            raise ValueError("No ASCII frames to render")
        
        # Determine dimensions
        height = len(ascii_frames[0])
        width = len(ascii_frames[0][0])
        
        # Calculate output image dimensions
        char_width = self.font_size
        char_height = self.font_size
        img_width = width * char_width
        img_height = height * char_height
        
        # Set up temporary directory for frames
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Render each ASCII frame as an image
            frame_paths = []
            for i, ascii_frame in enumerate(tqdm(ascii_frames, desc="Rendering frames")):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                self._render_ascii_frame_to_image(ascii_frame, frame_path, img_width, img_height)
                frame_paths.append(frame_path)
            
            # Use OpenCV to create the video
            self._create_video_from_frames(frame_paths, output_path, img_width, img_height)
            
        finally:
            # Clean up temporary frames
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {temp_dir}")
                print(f"Error: {str(e)}")
                print("You may want to delete it manually.")
    
    def _render_ascii_frame_to_image(self, ascii_frame, output_path, img_width, img_height):
        """
        Render a single ASCII frame to an image.
        
        Args:
            ascii_frame (list): 2D list of ASCII characters
            output_path (str): Path to save the rendered image
            img_width (int): Width of the output image
            img_height (int): Height of the output image
        """
        # Create a black image
        image = Image.new("RGB", (img_width, img_height), color="black")
        draw = ImageDraw.Draw(image)
        
        # Draw each character
        for y, row in enumerate(ascii_frame):
            for x, char in enumerate(row):
                # Draw white character on black background
                draw.text(
                    (x * self.font_size, y * self.font_size),
                    char,
                    font=self.font,
                    fill="white"
                )
        
        # Save the image
        image.save(output_path)
    
    def _create_video_from_frames(self, frame_paths, output_path, width, height):
        """
        Create a video from a list of frame image paths.
        
        Args:
            frame_paths (list): List of paths to frame images
            output_path (str): Path to save the output video
            width (int): Width of the video
            height (int): Height of the video
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # For Windows compatibility, use mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create video writer for {output_path}")
        
        for frame_path in tqdm(frame_paths, desc="Creating video"):
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
            video_writer.write(frame)
        
        video_writer.release()
        
        print(f"Video saved to {output_path}")
```

### 4.6 Utilities (utils.py)

Utility functions for the project with Windows-specific paths.

```python
import os
import subprocess
import sys

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible on Windows."""
    try:
        # Use shell=True for Windows command execution
        subprocess.run('ffmpeg -version', check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def create_directory_if_not_exists(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except PermissionError:
            print(f"Error: Permission denied when creating directory: {directory_path}")
            print("Try running the script with administrator privileges or choose a different directory.")
            sys.exit(1)

def get_video_info(video_path):
    """Get basic information about a video file using ffprobe on Windows."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Windows-safe command with proper quoting
    cmd = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=width,height,duration,r_frame_rate '
        f'-of json "{video_path}"'
    )
    
    try:
        # Use shell=True for Windows command execution
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        import json
        info = json.loads(result.stdout)
        return info['streams'][0]
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        error_message = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)
        raise RuntimeError(f"Error getting video info: {error_message}")

def is_admin():
    """Check if the script is running with administrator privileges on Windows."""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False
```

## 5. Installation and Setup Instructions for Windows

### 5.1 Requirements.txt

```
opencv-python==4.7.0.72
numpy==1.24.3
Pillow==9.5.0
tqdm==4.65.0
```

### 5.2 Installation Steps

1. Install Python 3.10 from the Microsoft Store or https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"
   - Also check "Install pip"

2. Install ffmpeg on Windows:
   - Download ffmpeg from https://ffmpeg.org/download.html (specifically the Windows builds)
   - Extract the ZIP file to a location like C:\ffmpeg
   - Add the bin folder (e.g., C:\ffmpeg\bin) to your PATH environment variable:
     1. Search for "Environment Variables" in the Start Menu
     2. Click "Edit the system environment variables"
     3. Click "Environment Variables" button
     4. Under "System variables", find and select "Path", then click "Edit"
     5. Click "New" and add the path to the bin folder (e.g., C:\ffmpeg\bin)
     6. Click "OK" on all dialogs to save changes

3. Open Command Prompt and create a project directory:
   ```
   mkdir ascii_video_converter
   cd ascii_video_converter
   ```

4. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

5. Create requirements.txt file with the contents from section 5.1, then install dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Create all Python files as specified in the design document sections 4.1 through 4.6.

## 6. Usage Instructions for Windows

### 6.1 Basic Usage

Open Command Prompt in the project directory and run:

```
python main.py input_video.mp4 output_video.mp4
```

### 6.2 Advanced Options

```
python main.py input_video.mp4 output_video.mp4 --width 160 --height 90 --fps 30 --font-size 14 --temp-dir .\temp_folder
```

### 6.3 Parameters

- `input_path`: Path to the input video file (required)
- `output_path`: Path to save the output video (required)
- `--width`: Width of the ASCII output in characters (default: 120)
- `--height`: Height of the ASCII output in characters (default: 60)
- `--fps`: Frames per second of the output video (default: 30)
- `--font-size`: Font size for ASCII characters (default: 12)
- `--temp-dir`: Directory for temporary files (default: .\temp)

## 7. Windows-Specific Error Handling

The application includes comprehensive error handling for Windows-specific issues:

1. **Path Issues**: Windows paths with backslashes are properly handled
2. **ffmpeg Installation**: Clear instructions for installing ffmpeg on Windows
3. **File Permission Problems**: Warnings for common Windows permission issues
4. **Font Availability**: Checks for Windows Japanese fonts
5. **Command Prompt Limitations**: All commands use shell=True for Windows compatibility

## 8. Performance Considerations for Windows

1. **Antivirus Impact**: Windows antivirus scanning may slow down file operations. Consider adding exclusions for your temp folder.
2. **Disk Fragmentation**: On Windows, disk fragmentation can impact video processing performance.
3. **Windows Defender**: Real-time protection can slow down processing of multiple files.
4. **Windows File Locking**: Windows tends to lock files more aggressively, which can cause issues when cleaning up temp files.

## 9. Testing Instructions for Windows

1. **Basic Functionality Test**:
   ```
   python main.py test_video.mp4 output.mp4 --width 80 --height 40
   ```

2. **Font Test** (to verify Japanese characters display correctly):
   ```
   python main.py test_video.mp4 font_test.mp4 --font-size 18
   ```

3. **Performance Test**:
   ```
   python main.py large_test_video.mp4 performance_test.mp4 --width 200 --height 100
   ```

4. **Windows Path Test** (testing paths with spaces and backslashes):
   ```
   python main.py "C:\My Videos\input video.mp4" "C:\My Videos\output video.mp4"
   ```

## 10. Windows-Specific Troubleshooting Guide

### 10.1 Common Windows Issues

1. **"'ffmpeg' is not recognized as an internal or external command"**:
   - ffmpeg is not properly added to your PATH
   - Solution: Follow the installation steps in section 5.2 to add ffmpeg to PATH
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

### 10.2 Support Resources

- ffmpeg Windows builds: https://ffmpeg.org/download.html#build-windows
- Windows Japanese fonts: https://www.microsoft.com/en-us/download/details.aspx?id=10020
- Python on Windows documentation: https://docs.python.org/3/using/windows.html

## 11. Conclusion

This document provides a comprehensive blueprint for creating a Japanese ASCII art video converter using Python 3.10 specifically for Windows environments. By following these specifications, developers can create a robust application that downscales videos, converts them to black and white, and maps pixel brightness to Japanese characters.

The design addresses Windows-specific concerns including path handling, font availability, command execution, and common Windows errors. All components are clearly described with proper error handling and user-friendly operation.