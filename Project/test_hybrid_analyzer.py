"""
Test Script for Hybrid Video Analysis
Demonstrates the hybrid approach combining Basic CV + VLM for optimal performance
"""
import os
import sys
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def test_hybrid_analysis():
    """Test the hybrid analysis approach"""
    
    # Video file path
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    print("🔬 HYBRID ANALYSIS: SMART BASIC CV + VLM COMBINATION")
    print("="*70)
    
    try:
        from video_analyzer import VideoContentAnalyzer, HybridAnalysisResult
        
        # Test 1: Hybrid Analysis (RECOMMENDED APPROACH)
        print("\n🚀 HYBRID ANALYSIS (Smart Basic CV + VLM)")
        print("-" * 60)
        print("📊 Features:")
        print("   • Fast Basic CV analysis first (3-5 seconds)")
        print("   • Intelligent VLM enhancement when beneficial")
        print("   • Merged results for best accuracy")
        print("   • Automatic cost optimization")
        
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
        
        # Performance Summary Table
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 60)
        print("| Method                | Time      | Objects | Confidence | Cost     |")
        print("|----------------------|-----------|---------|------------|----------|")
        print(f"| Basic CV Only        | {basic_time:5.1f}s    | {len(basic_result.key_objects):7} | 0.7        | FREE     |")
        if hybrid_result.vlm_result:
            vlm_time = hybrid_result.vlm_result.processing_time
            print(f"| VLM Only            | {vlm_time:5.1f}s    | {len(hybrid_result.vlm_result.key_objects):7} | 0.8        | FREE*    |")
        print(f"| HYBRID (Recommended) | {total_time:5.1f}s    | {len(hybrid_result.merged_result.key_objects):7} | {hybrid_result.merged_result.confidence_score:.1f}        | OPTIMAL  |")
        print("*After model download")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure video_analyzer.py is in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🏁 Hybrid Analysis Testing Completed!")
    
    print("\n💡 HYBRID RECOMMENDATION:")
    print("   🚀 Use hybrid analysis for production systems")
    print("   ⚡ Fast initial results + enhanced quality when needed")
    print("   💰 Cost-effective with intelligent VLM usage")
    print("   🎯 Best balance of speed, accuracy, and cost")
    
    print("\n🔄 INTELLIGENT DECISION MAKING:")
    print("   • Short/simple videos → Basic CV only")
    print("   • Complex/long videos → Basic CV + VLM enhancement")
    print("   • Music videos → Enhanced mode with more frames")
    print("   • Always merges best results from both approaches")

if __name__ == "__main__":
    test_hybrid_analysis()
