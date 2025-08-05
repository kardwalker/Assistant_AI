"""
Test Script for Video Analysis: VLM vs Basic Computer Vision
Compares Vision Language Models (VLM) with traditional computer vision approaches
"""
import os
import sys
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def test_vlm_vs_basic_cv():
    """Compare VLM (Vision Language Models) vs Basic Computer Vision"""
    
    # Video file path
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    print("🔬 HYBRID ANALYSIS: VLM + BASIC CV PERFORMANCE COMPARISON")
    print("="*70)
    
    try:
        from video_analyzer import VideoContentAnalyzer, analyze_video_only, HybridAnalysisResult
        
        # Test 1: Hybrid Analysis (RECOMMENDED APPROACH)
        print("\n🚀 HYBRID ANALYSIS (Smart Basic CV + VLM)")
        print("-" * 60)
        print("📊 Features:")
        print("   • Fast Basic CV analysis first (3-5 seconds)")
        print("   • Intelligent VLM enhancement when beneficial")
        print("   • Merged results for best accuracy")
        print("   • Automatic cost optimization")
        
        import time
        start_time = time.time()
        
        # Create hybrid analyzer (supports both local and API VLM)
        hybrid_analyzer = VideoContentAnalyzer(use_ai_model=True, use_api=False)  # Use local VLM
        hybrid_result = hybrid_analyzer.analyze_video_content_hybrid(video_path)
        
        total_time = time.time() - start_time
        
        print(f"\n📈 HYBRID RESULTS:")
        print(f"⏱️ Total Processing Time: {total_time:.2f} seconds")
        print(f"🎯 Final Analysis Method: {hybrid_result.merged_result.analysis_method}")
        print(f"📊 Confidence Score: {hybrid_result.merged_result.confidence_score:.1f}/1.0")
        print(f"✅ Success: {hybrid_result.merged_result.success}")
        
        print(f"\n🔍 MERGED ANALYSIS:")
        print(f"📹 Duration: {hybrid_result.merged_result.duration:.1f}s")
        print(f"🖼️ Frames Analyzed: {len(hybrid_result.merged_result.frame_descriptions)}")
        print(f"🎬 Scene: {hybrid_result.merged_result.scene_summary}")
        print(f"🔍 Objects: {', '.join(hybrid_result.merged_result.key_objects)}")
        
        print(f"\n⚡ PERFORMANCE BREAKDOWN:")
        metrics = hybrid_result.performance_metrics
        print(f"Basic CV Time: {metrics['basic_cv_time']:.2f}s")
        print(f"VLM Time: {metrics['vlm_time']:.2f}s")
        print(f"Used VLM Enhancement: {'✅ Yes' if metrics['used_vlm'] else '❌ No'}")
        print(f"Confidence Improvement: {metrics['confidence_improvement']:.1%}")
        print(f"Analysis Depth Factor: {metrics['speedup_factor']:.1f}x")
        
        # Compare individual results
        print(f"\n📋 DETAILED COMPARISON:")
        print("-" * 60)
        print(f"🔧 BASIC CV ALONE:")
        print(f"   Time: {hybrid_result.basic_result.processing_time:.2f}s")
        print(f"   Objects: {len(hybrid_result.basic_result.key_objects)} found")
        print(f"   Summary: {hybrid_result.basic_result.scene_summary[:80]}...")
        
        if hybrid_result.vlm_result:
            print(f"\n🤖 VLM ENHANCEMENT:")
            print(f"   Time: {hybrid_result.vlm_result.processing_time:.2f}s")
            print(f"   Objects: {len(hybrid_result.vlm_result.key_objects)} found")
            print(f"   Summary: {hybrid_result.vlm_result.scene_summary[:80]}...")
        else:
            print(f"\n🤖 VLM ENHANCEMENT: Skipped (Basic CV was sufficient)")
        
        # Test 2: Compare with pure methods for reference
        print(f"\n📊 REFERENCE COMPARISON (Individual Methods)")
        print("-" * 60)
        
        # Basic CV only
        print("🔧 Pure Basic CV:")
        start_time = time.time()
        basic_only = VideoContentAnalyzer(use_ai_model=False, use_api=False)
        basic_result = basic_only.analyze_video_content(video_path)
        basic_time = time.time() - start_time
        print(f"   ⏱️ Time: {basic_time:.2f}s | Objects: {len(basic_result.key_objects)} | Success: {basic_result.success}")
        
        # VLM only (if available)
        if hybrid_analyzer.use_ai_model or hybrid_analyzer.use_api:
            print("🤖 Pure VLM:")
            start_time = time.time()
            vlm_only = VideoContentAnalyzer(use_ai_model=True, use_api=False)
            vlm_result = vlm_only.analyze_video_content(video_path)
            vlm_time = time.time() - start_time
            print(f"   ⏱️ Time: {vlm_time:.2f}s | Objects: {len(vlm_result.key_objects)} | Success: {vlm_result.success}")
        
        print(f"\n🎯 HYBRID ADVANTAGE:")
        print("-" * 60)
        print("✅ Gets speed of Basic CV + accuracy of VLM")
        print("✅ Automatically chooses best approach per video")
        print("✅ Optimizes cost by avoiding unnecessary VLM calls")
        print("✅ Merges results for comprehensive analysis")
        print("✅ Maintains high confidence with smart fallbacks")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure video_analyzer.py is in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🏁 Hybrid Analysis Comparison Completed!")
    
    print("\n💡 HYBRID RECOMMENDATION:")
    print("   🚀 Use hybrid analysis for production systems")
    print("   ⚡ Fast initial results + enhanced quality when needed")
    print("   💰 Cost-effective with intelligent VLM usage")
    print("   🎯 Best balance of speed, accuracy, and cost")
    
    try:
        from video_analyzer import VideoContentAnalyzer, analyze_video_only
        
        # Test 1: Basic Computer Vision (Traditional CV)
        print("\n🔧 BASIC COMPUTER VISION (Traditional CV)")
        print("-" * 50)
        print("📊 Capabilities:")
        print("   • Edge detection, color analysis, brightness")
        print("   • Statistical analysis of pixel patterns")
        print("   • Motion detection, complexity analysis")
        print("   • Fast, deterministic, no internet required")
        
        start_time = time.time()
        analyzer_basic = VideoContentAnalyzer(use_ai_model=False, use_api=False)
        result_basic = analyzer_basic.analyze_video_content(video_path)
        basic_time = time.time() - start_time
        
        print(f"\n⏱️ Processing Time: {basic_time:.2f} seconds")
        if result_basic.success:
            print(f"✅ Analysis successful!")
            print(f"📹 Duration: {result_basic.duration:.1f}s")
            print(f"🖼️ Frames: {len(result_basic.frame_descriptions)}")
            print(f"🎬 Scene: {result_basic.scene_summary[:100]}...")
            print(f"🔍 Objects: {', '.join(result_basic.key_objects) if result_basic.key_objects else 'Basic patterns detected'}")
            print(f"💰 Cost: FREE")
        
        # Test 2: VLM Analysis (Local AI Models)
        print("\n🤖 VLM - LOCAL AI MODELS (BLIP, etc.)")
        print("-" * 50)
        print("📊 Capabilities:")
        print("   • Natural language scene descriptions")
        print("   • Object recognition and naming")
        print("   • Context understanding, relationships")
        print("   • Semantic understanding of scenes")
        
        start_time = time.time()
        analyzer_local_ai = VideoContentAnalyzer(use_ai_model=True, use_api=False)
        if analyzer_local_ai.use_ai_model:
            result_local_ai = analyzer_local_ai.analyze_video_content(video_path)
            local_ai_time = time.time() - start_time
            
            print(f"\n⏱️ Processing Time: {local_ai_time:.2f} seconds")
            if result_local_ai.success:
                print(f"✅ VLM analysis successful!")
                print(f"🎬 Scene: {result_local_ai.scene_summary[:120]}...")
                print(f"🔍 Objects: {', '.join(result_local_ai.key_objects)}")
                print(f"💰 Cost: FREE (after 990MB download)")
                print(f"📶 Internet: Not required after download")
            else:
                print(f"❌ VLM analysis failed")
        else:
            print("⚠️ Local VLM models not available")
            print("💡 Models would need to be downloaded first")
        
        # Test 3: VLM via API (GPT-4 Vision, etc.)
        print("\n🌐 VLM - API SERVICES (GPT-4 Vision, etc.)")
        print("-" * 50)
        print("📊 Capabilities:")
        print("   • State-of-the-art vision understanding")
        print("   • Complex reasoning about visual content")
        print("   • Multi-modal analysis (vision + language)")
        print("   • Constantly improving models")
        
        analyzer_api = VideoContentAnalyzer(use_ai_model=True, use_api=True)
        
        if analyzer_api.use_api:
            print("✅ API keys found - VLM API available")
            start_time = time.time()
            result_api = analyzer_api.analyze_video_content(video_path)
            api_time = time.time() - start_time
            
            print(f"⏱️ Processing Time: {api_time:.2f} seconds")
            if result_api.success:
                print(f"🎬 Scene: {result_api.scene_summary[:120]}...")
                print(f"💰 Cost: ~$0.01-0.10 per video (varies by frames)")
                print(f"📶 Internet: Required")
            else:
                print(f"❌ API analysis failed")
        else:
            print("⚠️ No API keys found - VLM API analysis skipped")
            print("💡 To enable VLM API analysis:")
            print("   1. Set OPENAI_API_KEY for GPT-4 Vision")
            print("   2. Or set AZURE_API_KEY + AZURE_ENDPOINT")
        
        # Comparison Summary
        print("\n📊 COMPREHENSIVE COMPARISON")
        print("="*60)
        print("| Method            | Speed    | Accuracy | Cost      | Internet |")
        print("|-------------------|----------|----------|-----------|----------|")
        print("| Basic CV          | Fast     | Basic    | FREE      | No       |")
        print("| Local VLM         | Medium   | High     | FREE*     | No       |")
        print("| API VLM           | Fast     | Highest  | Pay/Use   | Yes      |")
        print("*After initial model download")
        
        print("\n🎯 WHEN TO USE EACH APPROACH:")
        print("-" * 60)
        print("🔧 BASIC COMPUTER VISION:")
        print("   ✅ Quick prototyping and testing")
        print("   ✅ Real-time processing requirements")
        print("   ✅ Limited budget or offline scenarios")
        print("   ✅ Simple pattern detection tasks")
        print("   ❌ Limited semantic understanding")
        
        print("\n🤖 LOCAL VLM MODELS:")
        print("   ✅ Good balance of accuracy and cost")
        print("   ✅ Privacy-sensitive applications")
        print("   ✅ Offline deployment requirements")
        print("   ✅ Batch processing scenarios")
        print("   ❌ Initial setup complexity")
        print("   ❌ Large storage requirements")
        
        print("\n🌐 API VLM SERVICES:")
        print("   ✅ Highest accuracy and capabilities")
        print("   ✅ Latest model improvements")
        print("   ✅ No local storage requirements")
        print("   ✅ Complex reasoning tasks")
        print("   ❌ Ongoing costs")
        print("   ❌ Internet dependency")
        
        print("\n🚀 FUTURE TRENDS:")
        print("-" * 60)
        print("📈 VLMs are rapidly improving but:")
        print("   • Basic CV still valuable for speed/efficiency")
        print("   • Hybrid approaches often work best")
        print("   • Edge computing favors lightweight CV")
        print("   • Cost considerations matter at scale")
        print("   • Privacy requirements favor local processing")
        
        print("\n💡 RECOMMENDATION FOR YOUR PROJECT:")
        print("-" * 60)
        print("🎯 Use a HYBRID APPROACH:")
        print("   1. Basic CV for real-time preview/filtering")
        print("   2. Local VLM for detailed analysis")
        print("   3. API VLM for critical/complex content")
        print("   4. Let users choose based on their needs")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure video_analyzer.py is in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🏁 VLM vs Basic CV Comparison Completed!")

if __name__ == "__main__":
    test_vlm_vs_basic_cv()
        analyzer_basic = VideoContentAnalyzer(use_ai_model=False, use_api=False)
        result_basic = analyzer_basic.analyze_video_content(video_path)
        
        if result_basic.success:
            print(f"✅ Basic analysis successful!")
            print(f"Duration: {result_basic.duration:.1f}s")
            print(f"Frames analyzed: {len(result_basic.frame_descriptions)}")
            print(f"Scene summary: {result_basic.scene_summary}")
            print(f"Key objects: {', '.join(result_basic.key_objects)}")
        else:
            print(f"❌ Basic analysis failed: {result_basic.scene_summary}")
        
        # Test 2: API-based analysis (if keys available)
        print("\n🚀 Test 2: API-based Analysis")
        print("-" * 40)
        analyzer_api = VideoContentAnalyzer(use_ai_model=True, use_api=True)
        
        if analyzer_api.use_api:
            print("✅ API keys found - using fast API analysis")
            result_api = analyzer_api.analyze_video_content(video_path)
            if result_api.success:
                print(f"Duration: {result_api.duration:.1f}s")
                print(f"Scene summary: {result_api.scene_summary}")
            else:
                print(f"❌ API analysis failed: {result_api.scene_summary}")
        else:
            print("⚠️ No API keys found - API analysis skipped")
            print("💡 To enable API analysis:")
            print("   1. Set OPENAI_API_KEY environment variable")
            print("   2. Or set AZURE_API_KEY + AZURE_ENDPOINT")
        
        # Test 3: Standalone video analysis function
        print("\n📹 Test 3: Standalone Video Analysis Function")
        print("-" * 40)
        standalone_result = analyze_video_only(video_path)
        
        if standalone_result['success']:
            print(f"✅ Standalone analysis successful!")
            print(f"Method: {standalone_result.get('analysis_method', 'Unknown')}")
            print(f"Duration: {standalone_result['duration']:.1f}s")
            print(f"Scene summary: {standalone_result['scene_summary']}")
        else:
            print(f"❌ Standalone analysis failed: {standalone_result.get('error', 'Unknown error')}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure video_analyzer.py is in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("🏁 Testing completed!")
    
    # Performance recommendations
    print("\n💡 Performance Recommendations:")
    print("   🚀 FASTEST: Use API keys (OpenAI/Azure) - instant results")
    print("   🔧 MEDIUM: Basic computer vision - no downloads, good results")
    print("   ⏳ SLOWEST: Local AI models - 990MB download, high accuracy")

if __name__ == "__main__":
    test_video_analysis()
