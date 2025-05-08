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