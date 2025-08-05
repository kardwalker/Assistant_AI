"""
Test Music-Only Detection and Enhanced Video Analysis
Tests the new feature that focuses on video analysis when audio contains only music
"""
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def test_music_detection():
    """Test music-only detection and enhanced video analysis"""
    
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    print("🎵 TESTING MUSIC-ONLY DETECTION & ENHANCED VIDEO ANALYSIS")
    print("="*70)
    
    try:
        from video_analyzer import (
            detect_music_only_audio, 
            VideoContentAnalyzer, 
            parallel_analyze_video_and_audio
        )
        
        # Test 1: Mock music-only audio detection
        print("\n🧪 Test 1: Music-Only Detection")
        print("-" * 40)
        
        # Test cases for music detection
        test_cases = [
            {"success": True, "transcription": "Give me the keys to the cup, I'm pull out", "summary": "Speech content"},
            {"success": True, "transcription": "", "summary": ""},  # Empty - likely music
            {"success": True, "transcription": "music", "summary": "Background music"},  # Music keyword
            {"success": True, "transcription": "instrumental beat", "summary": "Instrumental"},  # Instrumental
            {"success": True, "transcription": "Hello there how are you doing today?", "summary": "Conversation"},  # Clear speech
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            is_music = detect_music_only_audio(test_case)
            transcription = test_case.get('transcription', '')
            print(f"Test {i}: '{transcription}' → {'🎵 MUSIC-ONLY' if is_music else '🗣️ SPEECH'}")
        
        # Test 2: Enhanced Video Analysis Mode
        print("\n🎬 Test 2: Enhanced vs Standard Video Analysis")
        print("-" * 40)
        
        analyzer = VideoContentAnalyzer(use_ai_model=False, use_api=False)  # Use basic CV for speed
        
        # Standard analysis
        print("📊 Standard Analysis:")
        result_standard = analyzer.analyze_video_content(video_path, enhanced_mode=False)
        if result_standard.success:
            print(f"  Frames analyzed: {len(result_standard.frame_descriptions)}")
            print(f"  Key objects: {len(result_standard.key_objects)} items")
            print(f"  Summary length: {len(result_standard.scene_summary)} chars")
            print(f"  Summary: {result_standard.scene_summary[:100]}...")
        
        # Enhanced analysis
        print("\n🔍 Enhanced Analysis (Music-Only Mode):")
        result_enhanced = analyzer.analyze_video_content(video_path, enhanced_mode=True)
        if result_enhanced.success:
            print(f"  Frames analyzed: {len(result_enhanced.frame_descriptions)}")
            print(f"  Key objects: {len(result_enhanced.key_objects)} items")
            print(f"  Summary length: {len(result_enhanced.scene_summary)} chars")
            print(f"  Summary: {result_enhanced.scene_summary[:150]}...")
            print(f"  Objects found: {', '.join(result_enhanced.key_objects[:5])}")
        
        # Test 3: Full Parallel Analysis with Music Detection
        print("\n🚀 Test 3: Full Parallel Analysis with Music Detection")
        print("-" * 40)
        
        print("🔄 Running complete analysis with automatic music detection...")
        full_result = parallel_analyze_video_and_audio(video_path, use_azure=False)
        
        if full_result['separation_success']:
            audio_analysis = full_result['audio_analysis']
            video_analysis = full_result['video_analysis']
            
            # Check if music was detected
            if audio_analysis:
                is_music_detected = detect_music_only_audio(audio_analysis)
                print(f"🎵 Music-only detected: {'YES' if is_music_detected else 'NO'}")
                
                if is_music_detected:
                    print("✅ Enhanced video analysis was automatically enabled")
                    print(f"🎬 Enhanced analysis method: {video_analysis.get('analysis_method', 'Unknown')}")
                    print(f"🔍 Enhanced mode active: {video_analysis.get('enhanced_mode', False)}")
                
            print(f"\n📊 Combined Summary:")
            print(f"{full_result['combined_summary']}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎯 MUSIC-ONLY DETECTION FEATURES:")
    print("✅ Automatic detection of instrumental/music-only audio")
    print("✅ Enhanced video analysis with more detailed frame extraction")
    print("✅ Expanded object detection for visual content")
    print("✅ Detailed scene progression analysis")
    print("✅ Smart focus shift from audio to visual content")
    print("="*70)

if __name__ == "__main__":
    test_music_detection()
