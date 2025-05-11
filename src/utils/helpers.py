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