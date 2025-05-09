import os
import cv2
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

class Renderer:
    def __init__(self, font_size=12, fps=30, num_processes=None):
        self.font_size = font_size
        self.fps = fps
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
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
    
    def render_ascii_frames(self, ascii_frames, output_path, actual_dimensions=None, batch_size=10):
        """
        Render ASCII frames to an output video using parallel processing.
        
        Args:
            ascii_frames (list): List of 2D ASCII frames
            output_path (str): Path to save the output video
            actual_dimensions (tuple, optional): Actual width and height of ASCII frames
                                               (for information purposes only)
            batch_size (int): Number of frames to process in each batch
        """
        if not ascii_frames:
            raise ValueError("No ASCII frames to render")
        
        # Determine dimensions from the actual ASCII frames
        # (actual_dimensions parameter is for information only)
        height = len(ascii_frames[0])
        width = len(ascii_frames[0][0])
        
        # Log the dimensions being used
        print(f"Rendering ASCII frames with dimensions: {width}x{height}")
        print(f"Using {self.num_processes} processes for parallel rendering")
        
        # Calculate output image dimensions
        char_width = self.font_size
        char_height = self.font_size
        img_width = width * char_width
        img_height = height * char_height
        
        # Set up temporary directory for frames
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Prepare frame rendering tasks
            frame_paths = []
            render_tasks = []
            
            for i, ascii_frame in enumerate(ascii_frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                frame_paths.append(frame_path)
                render_tasks.append((ascii_frame, frame_path, img_width, img_height, self.font_size))
            
            # Render frames in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Process frames in batches to manage memory
                for batch_start in tqdm(range(0, len(render_tasks), batch_size), desc="Rendering frame batches"):
                    batch_end = min(batch_start + batch_size, len(render_tasks))
                    batch_tasks = render_tasks[batch_start:batch_end]
                    
                    # Submit batch for parallel processing
                    futures = [
                        executor.submit(
                            render_ascii_frame_to_image_static,
                            task[0],  # ascii_frame
                            task[1],  # frame_path
                            task[2],  # img_width
                            task[3],  # img_height
                            task[4]   # font_size
                        ) for task in batch_tasks
                    ]
                    
                    # Wait for all futures to complete
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error rendering frame: {e}")
            
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


# Static method for parallel processing
def render_ascii_frame_to_image_static(ascii_frame, output_path, img_width, img_height, font_size):
    """
    Static method to render a single ASCII frame to an image.
    This is used for parallel processing since instance methods can't be pickled.
    
    Args:
        ascii_frame (list): 2D list of ASCII characters
        output_path (str): Path to save the rendered image
        img_width (int): Width of the output image
        img_height (int): Height of the output image
        font_size (int): Font size for ASCII characters
    """
    # Create a black image
    image = Image.new("RGB", (img_width, img_height), color="black")
    draw = ImageDraw.Draw(image)
    
    # Load font (need to load it here since we can't pickle the font object)
    font_paths = [
        "C:\\Windows\\Fonts\\msgothic.ttc",
        "C:\\Windows\\Fonts\\YuGothR.ttc",
        "C:\\Windows\\Fonts\\meiryo.ttc",
        "C:\\Windows\\Fonts\\meiryob.ttc",
        "C:\\Windows\\Fonts\\msmincho.ttc",
        "C:\\Windows\\Fonts\\yugothib.ttf",
        "C:\\Windows\\Fonts\\malgun.ttf"
    ]
    
    # Try to load each font until one succeeds
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except IOError:
                continue
    
    # If no font was loaded, try to use a default font
    if font is None:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # Draw each character
    for y, row in enumerate(ascii_frame):
        for x, char in enumerate(row):
            # Draw white character on black background
            draw.text(
                (x * font_size, y * font_size),
                char,
                font=font,
                fill="white"
            )
    
    # Save the image
    image.save(output_path)