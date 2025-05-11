import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import os
import sys
import threading
import subprocess
import multiprocessing
import cv2
from ..utils.utils import check_ffmpeg_installed
import re

class VideoToASCIIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VideoToASCII GUI")
        self.root.geometry("800x800")
        self.root.resizable(True, True)
        
        # Set default values
        self.input_path = tk.StringVar(value="input.mp4")
        self.output_path = tk.StringVar(value="output.mp4")
        self.scale_factor = tk.IntVar(value=100)  # Scale factor as percentage
        self.fps = tk.IntVar(value=30)
        self.font_size = tk.IntVar(value=12)
        self.temp_dir = tk.StringVar(value=".\\temp")
        self.processes = tk.IntVar(value=multiprocessing.cpu_count())
        self.batch_size = tk.IntVar(value=10)
        
        # Video info
        self.video_info = None
        self.width = tk.IntVar(value=120)  # Hidden variables for actual width/height
        self.height = tk.IntVar(value=60)  # These will be calculated based on scale factor
        
        # Create the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the form
        self.create_form()
        
        # Create the status area
        self.create_status_area()
        
        # Create the button area
        self.create_button_area()
        
        # Check for ffmpeg installation
        self.check_ffmpeg()

    def create_form(self):
        # Create a frame for the form
        form_frame = ttk.LabelFrame(self.main_frame, text="Settings", padding="10")
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for file selection
        file_frame = ttk.Frame(form_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        # Input file
        ttk.Label(file_frame, text="Input Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file
        ttk.Label(file_frame, text="Output Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Temp directory
        ttk.Label(file_frame, text="Temp Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.temp_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_temp_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # Create a frame for sliders
        slider_frame = ttk.Frame(form_frame)
        slider_frame.pack(fill=tk.X, pady=10)
        
        # Create a grid layout for sliders
        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=3)
        slider_frame.columnconfigure(2, weight=1)
        
        # Scale factor slider
        ttk.Label(slider_frame, text="Scale:").grid(row=0, column=0, sticky=tk.W, pady=5)
        scale_slider = ttk.Scale(slider_frame, from_=25, to=200, variable=self.scale_factor, orient=tk.HORIZONTAL,
                                command=self.update_dimensions)
        scale_slider.grid(row=0, column=1, sticky=tk.EW, pady=5)
        scale_entry = ttk.Entry(slider_frame, textvariable=self.scale_factor, width=5)
        scale_entry.grid(row=0, column=2, padx=5, pady=5)
        self.create_tooltip(scale_slider, "Scale factor for ASCII output (higher values = more detail)")
        
        # Output resolution label
        self.resolution_label = ttk.Label(slider_frame, text="Output Resolution: 120x60 characters")
        self.resolution_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Estimated pixel resolution label
        self.pixel_resolution_label = ttk.Label(slider_frame, text="Estimated Pixel Resolution: 1440x720 pixels")
        self.pixel_resolution_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # FPS slider
        ttk.Label(slider_frame, text="FPS:").grid(row=3, column=0, sticky=tk.W, pady=5)
        fps_slider = ttk.Scale(slider_frame, from_=10, to=60, variable=self.fps, orient=tk.HORIZONTAL)
        fps_slider.grid(row=3, column=1, sticky=tk.EW, pady=5)
        fps_entry = ttk.Entry(slider_frame, textvariable=self.fps, width=5)
        fps_entry.grid(row=3, column=2, padx=5, pady=5)
        self.create_tooltip(fps_slider, "Frames per second of output video")
        
        # Font size slider
        ttk.Label(slider_frame, text="Font Size:").grid(row=4, column=0, sticky=tk.W, pady=5)
        font_size_slider = ttk.Scale(slider_frame, from_=8, to=24, variable=self.font_size, orient=tk.HORIZONTAL,
                                    command=self.update_pixel_resolution)
        font_size_slider.grid(row=4, column=1, sticky=tk.EW, pady=5)
        font_size_entry = ttk.Entry(slider_frame, textvariable=self.font_size, width=5)
        font_size_entry.grid(row=4, column=2, padx=5, pady=5)
        self.create_tooltip(font_size_slider, "Font size for ASCII characters")
        
        # Processes slider
        ttk.Label(slider_frame, text="Processes:").grid(row=5, column=0, sticky=tk.W, pady=5)
        processes_slider = ttk.Scale(slider_frame, from_=1, to=multiprocessing.cpu_count(), variable=self.processes, orient=tk.HORIZONTAL)
        processes_slider.grid(row=5, column=1, sticky=tk.EW, pady=5)
        processes_entry = ttk.Entry(slider_frame, textvariable=self.processes, width=5)
        processes_entry.grid(row=5, column=2, padx=5, pady=5)
        self.create_tooltip(processes_slider, "Number of processes to use for parallel processing (default: number of CPU cores)")
        
        # Batch size slider
        ttk.Label(slider_frame, text="Batch Size:").grid(row=6, column=0, sticky=tk.W, pady=5)
        batch_size_slider = ttk.Scale(slider_frame, from_=1, to=50, variable=self.batch_size, orient=tk.HORIZONTAL)
        batch_size_slider.grid(row=6, column=1, sticky=tk.EW, pady=5)
        batch_size_entry = ttk.Entry(slider_frame, textvariable=self.batch_size, width=5)
        batch_size_entry.grid(row=6, column=2, padx=5, pady=5)
        self.create_tooltip(batch_size_slider, "Number of frames to process in each batch (default: 10)")

    def update_dimensions(self, *args):
        """Update width and height based on scale factor and video aspect ratio"""
        # Base dimensions for 100% scale
        base_width = 120
        base_height = 60
        
        # If we have video info, use its aspect ratio
        if self.video_info:
            aspect_ratio = self.video_info["width"] / self.video_info["height"]
            # Calculate base dimensions that maintain this aspect ratio
            if aspect_ratio > 2:  # Wide video
                base_width = 120
                base_height = int(base_width / aspect_ratio)
            else:  # Taller video
                base_height = 60
                base_width = int(base_height * aspect_ratio)
        
        # Apply scale factor
        scale = self.scale_factor.get() / 100.0
        new_width = int(base_width * scale)
        new_height = int(base_height * scale)
        
        # Update internal width and height variables
        self.width.set(new_width)
        self.height.set(new_height)
        
        # Update resolution label
        self.resolution_label.config(text=f"Output Resolution: {new_width}x{new_height} characters")
        
        # Update pixel resolution
        self.update_pixel_resolution()
    
    def update_pixel_resolution(self, *args):
        """Update the estimated pixel resolution based on character dimensions and font size"""
        char_width = self.width.get()
        char_height = self.height.get()
        font_size = self.font_size.get()
        
        pixel_width = char_width * font_size
        pixel_height = char_height * font_size
        
        self.pixel_resolution_label.config(text=f"Estimated Pixel Resolution: {pixel_width}x{pixel_height} pixels")

    def create_status_area(self):
        # Create a frame for the status area
        status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a frame for video info
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        # Video info label
        self.video_info_label = ttk.Label(info_frame, text="Video Info: No video selected")
        self.video_info_label.pack(anchor=tk.W, pady=2)
        
        # Create a scrolled text widget for status messages
        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)

    def create_button_area(self):
        # Create a frame for the buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Create the buttons
        self.run_button = ttk.Button(button_frame, text="Run", command=self.run_conversion)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.preview_button = ttk.Button(button_frame, text="Preview", command=self.show_preview, state=tk.DISABLED)
        self.preview_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Reset", command=self.reset_form).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if filename:
            self.input_path.set(filename)
            # Auto-generate output path
            if not self.output_path.get() or self.output_path.get() == "output.mp4":
                input_dir = os.path.dirname(filename)
                input_name = os.path.basename(filename)
                name, ext = os.path.splitext(input_name)
                output_name = f"{name}_ascii{ext}"
                output_path = os.path.join(input_dir, output_name)
                self.output_path.set(output_path)
            
            # Get video info and update dimensions
            self.get_video_info()
            
            # Enable preview button if video exists
            if os.path.exists(self.input_path.get()):
                self.preview_button.config(state=tk.NORMAL)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output Video",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
            defaultextension=".mp4"
        )
        if filename:
            self.output_path.set(filename)

    def browse_temp_dir(self):
        dirname = filedialog.askdirectory(title="Select Temporary Directory")
        if dirname:
            self.temp_dir.set(dirname)

    def check_ffmpeg(self):
        if not check_ffmpeg_installed():
            self.log_message("Error: ffmpeg is not installed or not accessible in PATH.")
            self.log_message("Please install ffmpeg from https://ffmpeg.org/download.html")
            self.log_message("Make sure to add it to your PATH environment variable.")
            messagebox.showerror("FFmpeg Not Found", 
                                "FFmpeg is not installed or not accessible in PATH.\n"
                                "Please install ffmpeg from https://ffmpeg.org/download.html\n"
                                "Make sure to add it to your PATH environment variable.")
        else:
            self.log_message("FFmpeg is installed and accessible.")
            
    def get_video_info(self):
        """Get information about the input video file"""
        if not self.input_path.get() or not os.path.exists(self.input_path.get()):
            return
        
        try:
            # Get video info using OpenCV
            cap = cv2.VideoCapture(self.input_path.get())
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.input_path.get()}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            self.video_info = {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration
            }
            
            # Update video info label
            self.video_info_label.config(
                text=f"Video Info: {width}x{height}, {fps:.2f} FPS, {frame_count} frames, {duration:.2f} seconds"
            )
            
            # Update dimensions based on new video info
            self.update_dimensions()
            
            # Update FPS to match video if it's valid
            if fps > 0:
                self.fps.set(int(fps))
            
        except Exception as e:
            self.log_message(f"Error getting video info: {str(e)}")
            self.video_info = None
            self.video_info_label.config(text="Video Info: Error getting video information")

    def log_message(self, message):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def validate_inputs(self):
        # Check if input file exists
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input video file.")
            return False
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", f"Input file does not exist: {self.input_path.get()}")
            return False
        
        # Check if output path is specified
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output video file.")
            return False
        
        # Check if output directory exists
        output_dir = os.path.dirname(self.output_path.get())
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {str(e)}")
                return False
        
        # Validate numeric inputs
        try:
            width = self.width.get()
            height = self.height.get()
            fps = self.fps.get()
            font_size = self.font_size.get()
            processes = self.processes.get()
            batch_size = self.batch_size.get()
            
            if width <= 0 or height <= 0 or fps <= 0 or font_size <= 0 or processes <= 0 or batch_size <= 0:
                messagebox.showerror("Error", "All numeric values must be greater than zero.")
                return False
        except:
            messagebox.showerror("Error", "Invalid numeric input.")
            return False
        
        return True

    def run_conversion(self):
        if not self.validate_inputs():
            return
        
        # Disable the Run button during conversion
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button) and widget["text"] == "Run":
                widget.config(state=tk.DISABLED)
        
        # Clear the status area
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
        
        # Build the command
        # Note: We use the calculated width and height values from the scale factor
        # rather than passing the scale factor directly
        cmd = [
            sys.executable,
            "../main.py",
            self.input_path.get(),
            self.output_path.get(),
            "--width", str(self.width.get()),
            "--height", str(self.height.get()),
            "--fps", str(self.fps.get()),
            "--font-size", str(self.font_size.get()),
            "--temp-dir", self.temp_dir.get(),
            "--processes", str(self.processes.get()),
            "--batch-size", str(self.batch_size.get())
        ]
        
        # Log the command
        self.log_message("Running command:")
        self.log_message(" ".join(cmd))
        self.log_message("\nProcessing... (this may take a while)")
        
        # Run the command in a separate thread
        threading.Thread(target=self.run_command, args=(cmd,), daemon=True).start()

    def run_command(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read and display output in real-time
            for line in iter(process.stdout.readline, ''):
                self.log_message(line.strip())
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                self.log_message("\nConversion completed successfully!")
                messagebox.showinfo("Success", "Video conversion completed successfully!")
            else:
                self.log_message(f"\nConversion failed with return code {return_code}")
                messagebox.showerror("Error", f"Conversion failed with return code {return_code}")
        
        except Exception as e:
            self.log_message(f"\nError: {str(e)}")
            messagebox.showerror("Error", str(e))
        
        finally:
            # Re-enable the Run button
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for button in child.winfo_children():
                                if isinstance(button, ttk.Button) and button["text"] == "Run":
                                    button.config(state=tk.NORMAL)

    def reset_form(self):
        # Reset all values to defaults
        self.input_path.set("input.mp4")
        self.output_path.set("output.mp4")
        self.scale_factor.set(100)
        self.fps.set(30)
        self.font_size.set(12)
        self.temp_dir.set(".\\temp")
        self.processes.set(multiprocessing.cpu_count())
        self.batch_size.set(10)
        
        # Reset video info
        self.video_info = None
        self.video_info_label.config(text="Video Info: No video selected")
        
        # Disable preview button
        self.preview_button.config(state=tk.DISABLED)
        
        # Reset width and height
        self.width.set(120)
        self.height.set(60)
        
        # Update resolution labels
        self.resolution_label.config(text="Output Resolution: 120x60 characters")
        self.update_pixel_resolution()
        
        # Clear the status area
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
        
        self.log_message("Form reset to default values.")

    def create_tooltip(self, widget, text):
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, justify=tk.LEFT,
                             background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                             font=("tahoma", "8", "normal"))
            label.pack(ipadx=1)
        
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def show_preview(self):
        """Show a preview of the ASCII conversion for a single frame"""
        if not self.validate_inputs():
            return
        
        try:
            # Extract a single frame from the video
            cap = cv2.VideoCapture(self.input_path.get())
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.input_path.get()}")
            
            # Try to get a frame from the middle of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError("Could not read frame from video")
            
            # Resize frame to match ASCII dimensions
            width = self.width.get()
            height = self.height.get()
            resized_frame = cv2.resize(frame, (width, height))
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Create ASCII frame
            ascii_frame = []
            
            # Simple mapping of grayscale values to ASCII characters
            # This is a simplified version of what the actual program does
            ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
            
            for y in range(height):
                row = []
                for x in range(width):
                    pixel_value = gray_frame[y, x]
                    # Map pixel value (0-255) to ASCII character
                    char_index = min(int(pixel_value / 25.5), 9)
                    row.append(ascii_chars[char_index])
                ascii_frame.append(''.join(row))
            
            # Create a preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("ASCII Preview")
            
            # Use a monospace font for proper alignment
            text = tk.Text(preview_window, font=("Courier New", 8), wrap=tk.NONE)
            text.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars
            h_scrollbar = ttk.Scrollbar(preview_window, orient=tk.HORIZONTAL, command=text.xview)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            v_scrollbar = ttk.Scrollbar(preview_window, orient=tk.VERTICAL, command=text.yview)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            text.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            
            # Insert ASCII frame
            for line in ascii_frame:
                text.insert(tk.END, line + '\n')
            
            # Make text read-only
            text.config(state=tk.DISABLED)
            
            # Add a note about the preview
            note_frame = ttk.Frame(preview_window, padding=5)
            note_frame.pack(fill=tk.X)
            
            ttk.Label(note_frame, text="Note: This is a simplified preview. The actual output will use Japanese characters and may look different.").pack()
            
            # Add a close button
            ttk.Button(note_frame, text="Close", command=preview_window.destroy).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Error creating preview: {str(e)}")


def main():
    root = tk.Tk()
    app = VideoToASCIIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()