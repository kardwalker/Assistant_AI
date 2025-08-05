"""
VLM Diagnostic Script
Diagnoses why video analysis is producing poor results
"""

import os
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check environment setup"""
    print("🔍 Environment Diagnostic")
    print("=" * 50)
    
    # Check Python environment
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check API keys
    print(f"OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    print(f"Azure API Key: {'✅ Set' if os.getenv('AZURE_API_KEY') else '❌ Missing'}")
    
    # Check required packages
    packages = ['transformers', 'torch', 'cv2', 'PIL', 'openai']
    for package in packages:
        try:
            __import__(package)
            print(f"{package}: ✅ Available")
        except ImportError:
            print(f"{package}: ❌ Missing")

def test_vlm_models():
    """Test VLM model initialization"""
    print("\n🤖 VLM Models Diagnostic")
    print("=" * 50)
    
    try:
        from enhanced_vlm_analyzer import EnhancedVLMVideoAnalyzer
        
        # Test different model types
        models_to_test = ["qwen", "blip", "llava"]
        
        for model_type in models_to_test:
            print(f"\nTesting {model_type}...")
            try:
                analyzer = EnhancedVLMVideoAnalyzer(model_type=model_type)
                if analyzer.model is not None:
                    print(f"  {model_type}: ✅ Initialized successfully")
                else:
                    print(f"  {model_type}: ❌ Failed to initialize")
            except Exception as e:
                print(f"  {model_type}: ❌ Error - {str(e)}")
                
        # Test OpenAI separately (requires API key)
        if os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_API_KEY'):
            print(f"\nTesting OpenAI...")
            try:
                analyzer = EnhancedVLMVideoAnalyzer(model_type="openai")
                print(f"  OpenAI: ✅ API configured")
            except Exception as e:
                print(f"  OpenAI: ❌ Error - {str(e)}")
        else:
            print(f"\nOpenAI: ❌ No API key found")
                
    except ImportError as e:
        print(f"❌ Cannot import EnhancedVLMVideoAnalyzer: {e}")

def test_video_processing():
    """Test video file processing"""
    print("\n🎥 Video Processing Diagnostic")
    print("=" * 50)
    
    import cv2
    
    # Look for test videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    current_dir = Path('.')
    videos = []
    
    for ext in video_extensions:
        videos.extend(list(current_dir.glob(f'*{ext}')))
    
    if videos:
        test_video = videos[0]
        print(f"Testing with: {test_video}")
        
        cap = cv2.VideoCapture(str(test_video))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"  Frames: {frame_count}")
            print(f"  FPS: {fps}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Video readable: ✅")
            
            # Test frame extraction
            ret, frame = cap.read()
            if ret:
                print(f"  Frame shape: {frame.shape}")
                print(f"  Frame extraction: ✅")
            else:
                print(f"  Frame extraction: ❌")
                
        else:
            print(f"  Cannot open video: ❌")
            
        cap.release()
    else:
        print("No video files found in current directory")

def suggest_fixes():
    """Suggest fixes based on diagnostic results"""
    print("\n💡 Suggested Fixes")
    print("=" * 50)
    
    fixes = [
        "1. Install missing packages:",
        "   pip install transformers torch torchvision opencv-python pillow openai",
        "",
        "2. Set up API keys (choose one):",
        "   - OpenAI: Set OPENAI_API_KEY in .env file",
        "   - Azure: Set AZURE_API_KEY and AZURE_ENDPOINT in .env file",
        "",
        "3. For local models, ensure sufficient GPU memory:",
        "   - Qwen/LLaVA require 8GB+ VRAM",
        "   - Use BLIP for lower memory requirements",
        "",
        "4. Use CPU fallback if GPU unavailable:",
        "   - Models will run slower but should work",
        "",
        "5. Test with a simple video first:",
        "   - Short duration (< 1 minute)",
        "   - Clear content with people/objects",
        "",
        "6. Check video codec compatibility:",
        "   - H.264/MP4 works best",
        "   - Avoid exotic codecs"
    ]
    
    for fix in fixes:
        print(fix)

if __name__ == "__main__":
    check_environment()
    test_vlm_models()
    test_video_processing()
    suggest_fixes()
