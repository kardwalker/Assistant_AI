import subprocess
import os
from moviepy import VideoFileClip





file_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"

def process_video(video_file_path, output_audio_path=None, output_video_path=None):
    """
    Extract audio and video separately from a video file using moviepy library
    
    Args:
        video_file_path (str): Path to the input video file
        output_audio_path (str, optional): Path for the output audio file. 
                                         If None, will create an audio file with same name as video
        output_video_path (str, optional): Path for the output video file (video without audio).
                                         If None, will create a video file with same name + "_video_only"
    
    Returns:
        dict: Dictionary containing paths to extracted files
              {'audio': audio_file_path, 'video': video_file_path, 'success': True/False}
    """
    
    # Check if input file exists
    if not os.path.exists(video_file_path):
        print(f"Error: Input video file does not exist: {video_file_path}")
        return {'audio': None, 'video': None, 'success': False}
    
    # If no output paths specified, create them based on the video filename
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]
    video_dir = os.path.dirname(video_file_path)
    
    if output_audio_path is None:
        output_audio_path = os.path.join(video_dir, f"{video_name}_audio.mp3")
    
    if output_video_path is None:
        output_video_path = os.path.join(video_dir, f"{video_name}_video_only.mp4")
    
    try:
        # Load video file
        video = VideoFileClip(video_file_path)
        
        # Extract audio
        audio = video.audio
        print("Extracting audio...")
        audio.write_audiofile(output_audio_path)
        
        # Extract video without audio
        print("Extracting video without audio...")
        video_without_audio = video.without_audio()
        video_without_audio.write_videofile(output_video_path)
        
        # Close the clips to free up memory
        audio.close()
        video_without_audio.close()
        video.close()
        
        print(f"Audio extracted successfully: {output_audio_path}")
        print(f"Video extracted successfully: {output_video_path}")
        
        return {
            'audio': output_audio_path,
            'video': output_video_path,
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return {'audio': None, 'video': None, 'success': False}

def extract_audio_only(video_file_path, output_audio_path=None):
    """
    Extract only audio from a video file
    
    Args:
        video_file_path (str): Path to the input video file
        output_audio_path (str, optional): Path for the output audio file
    
    Returns:
        str: Path to the extracted audio file or None if failed
    """
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]
    video_dir = os.path.dirname(video_file_path)
    
    if output_audio_path is None:
        output_audio_path = os.path.join(video_dir, f"{video_name}_audio.mp3")
    
    try:
        video = VideoFileClip(video_file_path)
        audio = video.audio
        print("Extracting audio only...")
        audio.write_audiofile(output_audio_path)
        audio.close()
        video.close()
        print(f"Audio extracted successfully: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_video_only(video_file_path, output_video_path=None):
    """
    Extract only video (without audio) from a video file
    
    Args:
        video_file_path (str): Path to the input video file
        output_video_path (str, optional): Path for the output video file
    
    Returns:
        str: Path to the extracted video file or None if failed
    """
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]
    video_dir = os.path.dirname(video_file_path)
    
    if output_video_path is None:
        output_video_path = os.path.join(video_dir, f"{video_name}_video_only.mp4")
    
    try:
        video = VideoFileClip(video_file_path)
        print("Extracting video only...")
        video_without_audio = video.without_audio()
        video_without_audio.write_videofile(output_video_path)
        video_without_audio.close()
        video.close()
        print(f"Video extracted successfully: {output_video_path}")
        return output_video_path
    except Exception as e:
        print(f"Error extracting video: {e}")
        return None

# Example usage
print("=" * 50)
print("🎬 VIDEO SEPARATOR - Processing your video...")
print("=" * 50)

result = process_video(file_path)

if result['success']:
    print(f"✅ Processing completed successfully!")
    print(f"📄 Audio file: {result['audio']}")
    print(f"🎬 Video file (without audio): {result['video']}")
else:
    print("❌ Processing failed!")

print("\n" + "=" * 50)
print("📋 Available Functions:")
print("=" * 50)
print("1. process_video(video_path) - Extracts both audio and video separately")
print("2. extract_audio_only(video_path) - Extracts only audio")
print("3. extract_video_only(video_path) - Extracts only video without audio")
print("=" * 50)

