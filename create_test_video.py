import cv2
import numpy as np
import os

def create_test_video(output_path, width=640, height=480, duration=5, fps=30):
    """
    Create a test video with moving shapes for testing the ASCII converter.
    
    Args:
        output_path (str): Path to save the output video
        width (int): Width of the video
        height (int): Height of the video
        duration (int): Duration of the video in seconds
        fps (int): Frames per second
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Create frames with moving shapes
    for i in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a moving circle
        circle_x = int(width / 2 + (width / 4) * np.sin(i * 0.05))
        circle_y = int(height / 2 + (height / 4) * np.cos(i * 0.05))
        cv2.circle(frame, (circle_x, circle_y), 50, (0, 0, 255), -1)
        
        # Draw a moving rectangle
        rect_x = int(width / 2 + (width / 4) * np.cos(i * 0.03))
        rect_y = int(height / 2 + (height / 4) * np.sin(i * 0.03))
        cv2.rectangle(frame, (rect_x - 40, rect_y - 40), (rect_x + 40, rect_y + 40), (0, 255, 0), -1)
        
        # Draw a moving triangle
        triangle_x = int(width / 2 + (width / 4) * np.sin(i * 0.07))
        triangle_y = int(height / 2 + (height / 4) * np.cos(i * 0.07))
        points = np.array([
            [triangle_x, triangle_y - 50],
            [triangle_x - 40, triangle_y + 30],
            [triangle_x + 40, triangle_y + 30]
        ], np.int32)
        cv2.fillPoly(frame, [points], (255, 0, 0))
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write the frame
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    print(f"Test video created at: {output_path}")

if __name__ == "__main__":
    # Create test videos of different sizes
    create_test_video("test_videos/small.mp4", width=320, height=240, duration=3, fps=30)
    create_test_video("test_videos/medium.mp4", width=640, height=480, duration=3, fps=30)
    create_test_video("test_videos/large.mp4", width=1280, height=720, duration=3, fps=30)