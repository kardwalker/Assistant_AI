"""
Test script for video processing and audio summarization integration
"""
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def test_video_to_summary():
    """Test the complete pipeline from video to summary"""
    
    # Video file path
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    print("🎬 Testing Video Processing and Summarization Pipeline")
    print("=" * 60)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
        return
    
    try:
        # Step 1: Test separator.py
        print("📁 Step 1: Testing audio extraction...")
        from separator import extract_audio_only
        
        audio_file = extract_audio_only(video_path)
        if audio_file and os.path.exists(audio_file):
            print(f"✅ Audio extracted: {os.path.basename(audio_file)}")
        else:
            print("❌ Audio extraction failed")
            return
        
        # Step 2: Test summazier.py
        print("\n🎵 Step 2: Testing audio summarization...")
        from summazier import AudioSummarizer
        
        summarizer = AudioSummarizer(use_azure=False)  # Use local model
        result = summarizer.process_audio_file(audio_file)
        
        if result['success']:
            print("✅ Audio processing successful!")
            print(f"\n📝 Transcription Preview:")
            print("-" * 30)
            preview = result['transcription'][:200] + "..." if len(result['transcription']) > 200 else result['transcription']
            print(preview)
            print(f"\n📄 Summary:")
            print("-" * 30)
            print(result['summary'])
        else:
            print("❌ Audio processing failed")
        
        # Step 3: Test complete pipeline
        print(f"\n🔄 Step 3: Testing complete pipeline...")
        from summazier import process_video_and_summarize
        
        complete_result = process_video_and_summarize(video_path, use_azure=False)
        
        if complete_result['success']:
            print("✅ Complete pipeline successful!")
            print(f"\n📊 Final Results:")
            print("-" * 30)
            print(f"Video: {os.path.basename(complete_result['video_file'])}")
            print(f"Audio: {os.path.basename(complete_result['audio_file'])}")
            print(f"Transcription length: {len(complete_result['transcription'])} characters")
            print(f"Summary: {complete_result['summary']}")
        else:
            print(f"❌ Complete pipeline failed: {complete_result['summary']}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("pip install transformers torch librosa openai")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")

def test_with_azure():
    """Test with Azure APIs (if configured)"""
    
    # Check for Azure configuration
    azure_key = os.getenv("AZURE_API_KEY")
    if not azure_key:
        print("❌ Azure API key not found in environment variables")
        print("Set AZURE_API_KEY and AZURE_ENDPOINT to test Azure integration")
        return
    
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    try:
        from summazier import process_video_and_summarize
        
        print("☁️ Testing with Azure APIs...")
        result = process_video_and_summarize(video_path, use_azure=True)
        
        if result['success']:
            print("✅ Azure processing successful!")
            print(f"Summary: {result['summary']}")
        else:
            print(f"❌ Azure processing failed: {result['summary']}")
            
    except Exception as e:
        print(f"❌ Azure test error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Integration Tests...")
    print("Choose test option:")
    print("1. Test with local models (Hugging Face)")
    print("2. Test with Azure APIs")
    print("3. Run both tests")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_video_to_summary()
    elif choice == "2":
        test_with_azure()
    elif choice == "3":
        test_video_to_summary()
        print("\n" + "="*60)
        test_with_azure()
    else:
        print("Invalid choice. Running default test...")
        test_video_to_summary()
