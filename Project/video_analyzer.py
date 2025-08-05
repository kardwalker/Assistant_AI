"""
Video Content Analyzer
Processes video files to extract visual information, detect objects, scenes, and actions


Demerits : The current approach analyzes the frames independently, without understanding the sequence
and relationships between them 
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Optional
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: BLIP model not available. Install with: pip install transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

@dataclass
class VideoFrame:
    """Represents a video frame with metadata"""
    frame_number: int
    timestamp: float
    description: str
    confidence: float

@dataclass
class VideoAnalysisResult:
    """Results from video content analysis"""
    video_file: str
    total_frames: int
    duration: float
    frame_descriptions: List[VideoFrame]
    scene_summary: str
    key_objects: List[str]
    success: bool
    analysis_method: str = "basic"
    confidence_score: float = 0.0
    processing_time: float = 0.0

@dataclass
class HybridAnalysisResult:
    """Results from hybrid analysis combining multiple methods"""
    basic_result: VideoAnalysisResult
    vlm_result: Optional[VideoAnalysisResult]
    merged_result: VideoAnalysisResult
    performance_metrics: Dict[str, float]

class VideoContentAnalyzer:
    def __init__(self, use_ai_model=True, use_api=True):
        """
        Initialize video content analyzer
        
        Args:
            use_ai_model (bool): Whether to use AI models for image captioning
            use_api (bool): Whether to use API services (faster than local models)
        """
        self.use_api = use_api and OPENAI_AVAILABLE and (OPENAI_API_KEY or AZURE_API_KEY)
        self.use_ai_model = use_ai_model and BLIP_AVAILABLE and TORCH_AVAILABLE
        
        if self.use_api:
            print("🚀 Using OpenAI API for fast image analysis (no model download needed)")
            if AZURE_API_KEY:
                openai.api_type = "azure"
                openai.api_key = AZURE_API_KEY
                openai.api_base = AZURE_ENDPOINT
                openai.api_version = "2024-02-01"
            else:
                openai.api_key = OPENAI_API_KEY
            self.processor = None
            self.model = None
        elif self.use_ai_model:
            print("⏳ Loading BLIP model for image captioning (this may take a while)...")
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("✅ BLIP model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load BLIP model: {e}")
                self.use_ai_model = False
                self.processor = None
                self.model = None
        else:
            print("🔧 Using basic computer vision (no AI models)")
            self.processor = None
            self.model = None
    
    def extract_frames(self, video_path: str, frame_interval: int = 30, enhanced_mode: bool = False) -> List[np.ndarray]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path (str): Path to video file
            frame_interval (int): Extract every N frames
            enhanced_mode (bool): Extract more frames for detailed analysis
            
        Returns:
            List[np.ndarray]: List of extracted frames
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adjust frame extraction for enhanced mode
        if enhanced_mode:
            frame_interval = max(15, frame_interval // 2)  # More frequent sampling
            max_frames = 30  # More frames for detailed analysis
        else:
            max_frames = 20  # Standard frame limit
        
        print(f"📹 Extracting frames from video (Total: {total_frames}, FPS: {fps:.2f})")
        if enhanced_mode:
            print("🔍 Enhanced mode: Extracting more frames for detailed visual analysis")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append({
                    'frame': frame,
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0
                })
            
            frame_count += 1
            
            # Limit to prevent too many frames
            if len(frames) >= max_frames:
                break
        
        cap.release()
        print(f"✅ Extracted {len(frames)} frames for analysis")
        return frames
    
    def describe_frame_ai(self, frame: np.ndarray) -> str:
        """
        Generate description for a frame using AI model or API
        
        Args:
            frame (np.ndarray): Video frame
            
        Returns:
            str: Frame description
        """
        if self.use_api:
            return self.describe_frame_api(frame)
        elif self.use_ai_model:
            return self.describe_frame_local_ai(frame)
        else:
            return self.describe_frame_basic(frame)
    
    def describe_frame_api(self, frame: np.ndarray) -> str:
        """
        Generate description using OpenAI API (much faster)
        
        Args:
            frame (np.ndarray): Video frame
            
        Returns:
            str: Frame description
        """
        try:
            import base64
            import io
            from PIL import Image
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Call OpenAI API
            if AZURE_API_KEY:
                response = openai.ChatCompletion.create(
                    engine="gpt-4-vision-preview",  # or your vision model deployment
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in one concise sentence."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                            ]
                        }
                    ],
                    max_tokens=50
                )
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in one concise sentence."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                            ]
                        }
                    ],
                    max_tokens=50
                )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"API description error: {e}")
            return self.describe_frame_basic(frame)
    
    def describe_frame_local_ai(self, frame: np.ndarray) -> str:
        """
        Generate description for a frame using local AI model
        
        Args:
            frame (np.ndarray): Video frame
            
        Returns:
            str: Frame description
        """
        if not self.use_ai_model or not self.processor or not self.model:
            return self.describe_frame_basic(frame)
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with BLIP
            inputs = self.processor(frame_rgb, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            return description
        except Exception as e:
            print(f"Local AI description error: {e}")
            return self.describe_frame_basic(frame)
    
    def describe_frame_basic(self, frame: np.ndarray) -> str:
        """
        Basic frame analysis using computer vision
        
        Args:
            frame (np.ndarray): Video frame
            
        Returns:
            str: Basic frame description
        """
        try:
            height, width = frame.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic statistics
            brightness = np.mean(gray)
            
            # Detect edges for complexity
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.sum(edges > 0) / (height * width)
            
            # Simple color analysis
            b, g, r = cv2.split(frame)
            avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
            
            # Determine dominant color
            if avg_b > max(avg_g, avg_r):
                dominant_color = "blue"
            elif avg_g > avg_r:
                dominant_color = "green"
            else:
                dominant_color = "red"
            
            # Generate description based on analysis
            if brightness > 200:
                lighting = "bright"
            elif brightness > 100:
                lighting = "normal"
            else:
                lighting = "dark"
            
            if complexity > 0.1:
                scene_type = "complex scene with many details"
            elif complexity > 0.05:
                scene_type = "moderate detail scene"
            else:
                scene_type = "simple scene"
            
            # Detect motion/movement indicators
            motion_indicator = ""
            if complexity > 0.15:
                motion_indicator = " with possible movement"
            
            return f"A {lighting} {scene_type} with {dominant_color} tones{motion_indicator}"
            
        except Exception as e:
            print(f"Basic frame analysis error: {e}")
            return "Unable to analyze frame content"
    
    def analyze_video_content(self, video_path: str, enhanced_mode: bool = False) -> VideoAnalysisResult:
        """
        Analyze video content and generate descriptions
        
        Args:
            video_path (str): Path to video file
            enhanced_mode (bool): Use enhanced analysis for music-only videos
            
        Returns:
            VideoAnalysisResult: Analysis results
        """
        if not os.path.exists(video_path):
            return VideoAnalysisResult(
                video_file=video_path,
                total_frames=0,
                duration=0,
                frame_descriptions=[],
                scene_summary="Video file not found",
                key_objects=[],
                success=False
            )
        
        print(f"🎬 Analyzing video content: {os.path.basename(video_path)}")
        if enhanced_mode:
            print("🎵 Enhanced video analysis mode (music-only audio detected)")
        
        # Extract frames with enhanced mode if needed
        frames_data = self.extract_frames(video_path, frame_interval=60, enhanced_mode=enhanced_mode)
        
        if not frames_data:
            return VideoAnalysisResult(
                video_file=video_path,
                total_frames=0,
                duration=0,
                frame_descriptions=[],
                scene_summary="Failed to extract frames from video",
                key_objects=[],
                success=False
            )
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Analyze frames in parallel
        frame_descriptions = []
        print("🔍 Analyzing frames...")
        
        # Reduce parallelism to avoid overwhelming the API/models
        max_workers = 2 if self.use_api else 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(self.describe_frame_ai, frame_data['frame']): frame_data 
                for frame_data in frames_data
            }
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_data = future_to_frame[future]
                try:
                    description = future.result(timeout=30)  # 30 second timeout per frame
                    frame_descriptions.append(VideoFrame(
                        frame_number=frame_data['frame_number'],
                        timestamp=frame_data['timestamp'],
                        description=description,
                        confidence=0.8  # Default confidence
                    ))
                    completed_count += 1
                    print(f"✅ Analyzed frame {completed_count}/{len(frames_data)}")
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Frame analysis timeout: frame {frame_data['frame_number']}")
                    # Add basic description as fallback
                    frame_descriptions.append(VideoFrame(
                        frame_number=frame_data['frame_number'],
                        timestamp=frame_data['timestamp'],
                        description=self.describe_frame_basic(frame_data['frame']),
                        confidence=0.3
                    ))
                except Exception as e:
                    print(f"❌ Frame analysis error: {e}")
                    # Add basic description as fallback
                    frame_descriptions.append(VideoFrame(
                        frame_number=frame_data['frame_number'],
                        timestamp=frame_data['timestamp'],
                        description=self.describe_frame_basic(frame_data['frame']),
                        confidence=0.3
                    ))
        
        # Sort by timestamp
        frame_descriptions.sort(key=lambda x: x.timestamp)
        
        # Generate scene summary with enhanced detail for music-only videos
        scene_summary = self.generate_scene_summary(frame_descriptions, enhanced_mode)
        
        # Extract key objects with enhanced detection
        key_objects = self.extract_key_objects(frame_descriptions, enhanced_mode)
        
        return VideoAnalysisResult(
            video_file=video_path,
            total_frames=total_frames,
            duration=duration,
            frame_descriptions=frame_descriptions,
            scene_summary=scene_summary,
            key_objects=key_objects,
            success=True
        )
    
    def generate_scene_summary(self, frame_descriptions: List[VideoFrame], enhanced_mode: bool = False) -> str:
        """Generate overall scene summary from frame descriptions"""
        if not frame_descriptions:
            return "No frame descriptions available"
        
        # Combine all descriptions
        all_descriptions = [f.description for f in frame_descriptions]
        
        if enhanced_mode:
            # Enhanced summary for music-only videos - more detailed
            if len(all_descriptions) <= 5:
                # Show all descriptions for short videos
                return " → ".join(all_descriptions)
            else:
                # Detailed progression for longer videos
                start_desc = all_descriptions[0]
                quarter_desc = all_descriptions[len(all_descriptions)//4]
                middle_desc = all_descriptions[len(all_descriptions)//2]
                three_quarter_desc = all_descriptions[3*len(all_descriptions)//4]
                end_desc = all_descriptions[-1]
                
                return f"Opening: {start_desc} | Early: {quarter_desc} | Middle: {middle_desc} | Late: {three_quarter_desc} | Ending: {end_desc}"
        else:
            # Standard summary generation
            if len(all_descriptions) <= 3:
                return " -> ".join(all_descriptions)
            else:
                start_desc = all_descriptions[0]
                middle_desc = all_descriptions[len(all_descriptions)//2]
                end_desc = all_descriptions[-1]
                return f"Video starts with: {start_desc}. Middle shows: {middle_desc}. Ends with: {end_desc}"
    
    def extract_key_objects(self, frame_descriptions: List[VideoFrame], enhanced_mode: bool = False) -> List[str]:
        """Extract key objects/themes from frame descriptions"""
        # Simple keyword extraction
        all_text = " ".join([f.description for f in frame_descriptions])
        
        if enhanced_mode:
            # Enhanced object detection for music-only videos
            keywords = [
                # People and actions
                'person', 'people', 'man', 'woman', 'child', 'dancing', 'walking', 'running',
                'sitting', 'standing', 'hand', 'face', 'gesture', 'movement',
                # Environment and objects
                'car', 'building', 'tree', 'sky', 'water', 'room', 'outdoor', 'indoor', 
                'street', 'stage', 'performance', 'light', 'lighting', 'crowd',
                # Visual elements
                'color', 'bright', 'dark', 'red', 'blue', 'green', 'scene', 'background',
                'foreground', 'text', 'sign', 'logo', 'banner',
                # Music video specific
                'concert', 'band', 'instrument', 'microphone', 'speaker', 'audience',
                'performer', 'stage', 'spotlight', 'costume', 'outfit'
            ]
        else:
            # Standard keywords
            keywords = ['person', 'people', 'car', 'building', 'tree', 'sky', 'water', 
                       'hand', 'face', 'room', 'outdoor', 'indoor', 'street', 'food']
        
        found_objects = []
        for keyword in keywords:
            if keyword in all_text.lower():
                found_objects.append(keyword)
        
        # Return more objects for enhanced mode
        max_objects = 10 if enhanced_mode else 5
        return found_objects[:max_objects]

    def analyze_video_content_hybrid(self, video_path: str) -> HybridAnalysisResult:
        """
        Hybrid analysis combining Basic CV and VLM for optimal performance
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            HybridAnalysisResult: Combined analysis results
        """
        import time
        print("🔬 STARTING HYBRID ANALYSIS (Basic CV + VLM)")
        print("=" * 60)
        
        # Step 1: Fast Basic CV Analysis
        print("⚡ Phase 1: Basic Computer Vision Analysis")
        start_time = time.time()
        basic_analyzer = VideoContentAnalyzer(use_ai_model=False, use_api=False)
        basic_result = basic_analyzer.analyze_video_content(video_path)
        basic_time = time.time() - start_time
        basic_result.processing_time = basic_time
        basic_result.analysis_method = "basic_cv"
        print(f"✅ Basic CV completed in {basic_time:.2f}s")
        
        # Step 2: Detect if VLM analysis is beneficial
        should_use_vlm = self._should_use_vlm_analysis(basic_result)
        vlm_result = None
        vlm_time = 0
        
        if should_use_vlm:
            print("🤖 Phase 2: VLM Enhancement Analysis")
            start_time = time.time()
            vlm_analyzer = VideoContentAnalyzer(use_ai_model=True, use_api=self.use_api)
            vlm_result = vlm_analyzer.analyze_video_content(video_path)
            vlm_time = time.time() - start_time
            vlm_result.processing_time = vlm_time
            vlm_result.analysis_method = "vlm_enhanced"
            print(f"✅ VLM analysis completed in {vlm_time:.2f}s")
        else:
            print("⏭️ Phase 2: Skipping VLM (Basic CV sufficient)")
        
        # Step 3: Merge Results
        print("🔀 Phase 3: Merging Results")
        merged_result = self._merge_analysis_results(basic_result, vlm_result)
        total_time = basic_time + vlm_time
        merged_result.processing_time = total_time
        merged_result.analysis_method = "hybrid"
        
        # Performance metrics
        performance_metrics = {
            'basic_cv_time': basic_time,
            'vlm_time': vlm_time,
            'total_time': total_time,
            'speedup_factor': vlm_time / basic_time if vlm_time > 0 else 1.0,
            'used_vlm': vlm_result is not None,
            'confidence_improvement': self._calculate_confidence_improvement(basic_result, vlm_result)
        }
        
        print(f"🏁 Hybrid analysis completed in {total_time:.2f}s")
        print(f"💡 Performance: {performance_metrics['speedup_factor']:.1f}x analysis depth")
        
        return HybridAnalysisResult(
            basic_result=basic_result,
            vlm_result=vlm_result,
            merged_result=merged_result,
            performance_metrics=performance_metrics
        )

    def _should_use_vlm_analysis(self, basic_result: VideoAnalysisResult) -> bool:
        """Determine if VLM analysis would add significant value"""
        if not self.use_ai_model and not self.use_api:
            return False
        
        # Use VLM for complex or long videos
        if basic_result.duration > 20:  # Longer videos benefit from detailed analysis
            return True
        
        # Use VLM if basic analysis seems limited
        if len(basic_result.key_objects) < 3:  # Basic CV found few objects
            return True
        
        # Use VLM if scene descriptions are generic
        generic_terms = ['scene', 'content', 'dark', 'bright', 'moderate']
        if any(term in basic_result.scene_summary.lower() for term in generic_terms):
            return True
        
        return False

    def _merge_analysis_results(self, basic_result: VideoAnalysisResult, vlm_result: Optional[VideoAnalysisResult]) -> VideoAnalysisResult:
        """Intelligently merge basic CV and VLM results"""
        if vlm_result is None:
            # No VLM result, return enhanced basic result
            basic_result.confidence_score = 0.7
            return basic_result
        
        # Merge scene summaries
        if vlm_result.scene_summary and len(vlm_result.scene_summary) > len(basic_result.scene_summary):
            merged_summary = vlm_result.scene_summary
        else:
            merged_summary = f"{basic_result.scene_summary} | Enhanced: {vlm_result.scene_summary}"
        
        # Merge key objects (combine unique objects from both)
        merged_objects = list(set(basic_result.key_objects + vlm_result.key_objects))
        
        # Use VLM frame descriptions if available, otherwise basic
        merged_frames = vlm_result.frame_descriptions if vlm_result.frame_descriptions else basic_result.frame_descriptions
        
        # Create merged result
        merged_result = VideoAnalysisResult(
            video_file=basic_result.video_file,
            total_frames=max(basic_result.total_frames, vlm_result.total_frames),
            duration=basic_result.duration,  # Use basic result for consistency
            frame_descriptions=merged_frames,
            scene_summary=merged_summary,
            key_objects=merged_objects,
            success=basic_result.success and vlm_result.success,
            analysis_method="hybrid_merged",
            confidence_score=0.9,  # High confidence from combined analysis
            processing_time=0  # Will be set by calling function
        )
        
        return merged_result

    def _calculate_confidence_improvement(self, basic_result: VideoAnalysisResult, vlm_result: Optional[VideoAnalysisResult]) -> float:
        """Calculate how much VLM improved the analysis confidence"""
        if vlm_result is None:
            return 0.0
        
        # Simple heuristic: compare information richness
        basic_info = len(basic_result.scene_summary) + len(basic_result.key_objects) * 10
        vlm_info = len(vlm_result.scene_summary) + len(vlm_result.key_objects) * 10
        
        if basic_info == 0:
            return 1.0
        
        improvement = (vlm_info - basic_info) / basic_info
        return max(0.0, min(1.0, improvement))  # Clamp between 0 and 1

def parallel_analyze_video_and_audio(video_path: str, use_azure: bool = False) -> Dict:
    """
    Analyze both video content and audio content in parallel
    
    Args:
        video_path (str): Path to video file
        use_azure (bool): Whether to use Azure APIs for audio
        
    Returns:
        Dict: Combined analysis results
    """
    print("🚀 Starting parallel video and audio analysis...")
    
    # Import audio processing
    try:
        from summazier import process_video_and_summarize
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(__file__))
        try:
            from summazier import process_video_and_summarize
        except ImportError:
            print("❌ Could not import audio processing module")
            return {
                'video_file': video_path,
                'separation_success': False,
                'error': 'Audio processing module not available',
                'audio_analysis': None,
                'video_analysis': None,
                'combined_summary': 'Failed to load audio processing'
            }
    
    # Import video processing
    try:
        from separator import process_video
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(__file__))
        try:
            from separator import process_video
        except ImportError:
            print("❌ Could not import video processing module")
            return {
                'video_file': video_path,
                'separation_success': False,
                'error': 'Video processing module not available',
                'audio_analysis': None,
                'video_analysis': None,
                'combined_summary': 'Failed to load video processing'
            }
    
    # Step 1: Separate video into audio and video files
    print("📁 Separating video into audio and video streams...")
    separation_result = process_video(video_path)
    
    if not separation_result['success']:
        return {
            'video_file': video_path,
            'separation_success': False,
            'error': 'Failed to separate video streams',
            'audio_analysis': None,
            'video_analysis': None,
            'combined_summary': 'Failed to process video'
        }
    
    video_only_path = separation_result['video']
    audio_file_path = separation_result['audio']
    
    # Step 2: Parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        print("🔄 Starting parallel processing...")
        
        # Submit audio analysis first to detect music-only content
        audio_future = executor.submit(process_video_and_summarize, video_path, use_azure)
        
        # Wait for audio analysis to complete first (to detect music-only)
        try:
            audio_result = audio_future.result(timeout=300)  # 5 minute timeout
            print("✅ Audio analysis completed")
            
            # Detect if audio is music-only
            is_music_only = detect_music_only_audio(audio_result)
            if is_music_only:
                print("🎵 Music-only audio detected - enabling enhanced video analysis")
            
        except Exception as e:
            print(f"❌ Audio analysis failed: {e}")
            audio_result = {'success': False, 'summary': str(e)}
            is_music_only = False
        
        # Now submit video analysis with enhanced mode if needed
        video_future = executor.submit(analyze_video_only, video_only_path, is_music_only)
        
        # Get video results
        try:
            video_result = video_future.result(timeout=300)  # 5 minute timeout
            print("✅ Video analysis completed")
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            video_result = {'success': False, 'scene_summary': str(e)}
    
    # Step 3: Combine results
    combined_summary = generate_combined_summary(audio_result, video_result)
    
    return {
        'video_file': video_path,
        'separation_success': True,
        'audio_file': audio_file_path,
        'video_only_file': video_only_path,
        'audio_analysis': audio_result,
        'video_analysis': video_result,
        'combined_summary': combined_summary,
        'processing_timestamp': datetime.now().isoformat()
    }

def analyze_video_only(video_only_path: str, enhanced_mode: bool = False) -> Dict:
    """Analyze video-only file content with optional enhanced mode"""
    try:
        # Try API first, then local AI, then basic CV
        analyzer = VideoContentAnalyzer(use_ai_model=True, use_api=True)
        result = analyzer.analyze_video_content(video_only_path, enhanced_mode=enhanced_mode)
        
        return {
            'success': result.success,
            'duration': result.duration,
            'total_frames': result.total_frames,
            'scene_summary': result.scene_summary,
            'key_objects': result.key_objects,
            'frame_count': len(result.frame_descriptions),
            'analysis_method': 'API' if analyzer.use_api else 'Local AI' if analyzer.use_ai_model else 'Basic CV',
            'enhanced_mode': enhanced_mode
        }
    except Exception as e:
        print(f"❌ Video analysis error: {e}")
        return {
            'success': False,
            'duration': 0,
            'total_frames': 0,
            'scene_summary': f'Video analysis failed: {str(e)}',
            'key_objects': [],
            'frame_count': 0,
            'analysis_method': 'Failed',
            'enhanced_mode': enhanced_mode,
            'error': str(e)
        }

def detect_music_only_audio(audio_result: Dict) -> bool:
    """
    Detect if audio contains only music (no speech/vocals)
    
    Args:
        audio_result (Dict): Audio analysis results
        
    Returns:
        bool: True if audio appears to be music-only
    """
    if not audio_result.get('success'):
        return False
    
    transcription = audio_result.get('transcription', '').strip().lower()
    
    # Indicators of music-only content
    music_indicators = [
        len(transcription) < 10,  # Very short or empty transcription
        transcription in ['', 'music', 'instrumental', 'background music'],
        'instrumental' in transcription,
        'background music' in transcription,
        transcription.count(' ') < 3,  # Very few words
    ]
    
    # Keywords that suggest music rather than speech
    music_keywords = ['beat', 'rhythm', 'melody', 'instrumental', 'music', 'song']
    speech_keywords = ['said', 'speak', 'talk', 'voice', 'conversation', 'dialogue']
    
    music_score = sum(keyword in transcription for keyword in music_keywords)
    speech_score = sum(keyword in transcription for keyword in speech_keywords)
    
    # If multiple music indicators or high music score with low speech score
    return (sum(music_indicators) >= 2) or (music_score > speech_score and speech_score == 0)

def generate_combined_summary(audio_result: Dict, video_result: Dict) -> str:
    """Generate combined summary from audio and video analysis with music detection"""
    summary_parts = []
    is_music_only = detect_music_only_audio(audio_result)
    
    # Video summary (prioritized if music-only audio)
    if video_result.get('success'):
        duration = video_result.get('duration', 0)
        scene_summary = video_result.get('scene_summary', '')
        key_objects = video_result.get('key_objects', [])
        
        if is_music_only:
            # Enhanced video focus for music-only content
            video_summary = f"Visual Content ({duration:.1f}s): {scene_summary}"
            if key_objects:
                video_summary += f" | Key visual elements: {', '.join(key_objects)}"
            video_summary += " | Audio: Background music/instrumental"
        else:
            # Standard video summary
            video_summary = f"Video Content ({duration:.1f}s): {scene_summary}"
            if key_objects:
                video_summary += f" Key visual elements: {', '.join(key_objects)}"
        
        summary_parts.append(video_summary)
    
    # Audio summary (de-emphasized if music-only)
    if audio_result.get('success') and not is_music_only:
        audio_summary = audio_result.get('summary', '')
        if audio_summary:
            summary_parts.append(f"Audio Content: {audio_summary}")
    
    # Combined insights
    if is_music_only and video_result.get('success'):
        # Focus on visual content when audio is just music
        combined = summary_parts[0] if summary_parts else "Video with background music"
        combined += " | This video primarily conveys information through visual content with musical accompaniment."
    elif len(summary_parts) >= 2:
        combined = " | ".join(summary_parts)
        combined += " | This video combines visual and audio elements for a complete multimedia experience."
    elif len(summary_parts) == 1:
        combined = summary_parts[0]
    else:
        combined = "Unable to analyze video content."
    
    return combined

# Example usage and testing
if __name__ == "__main__":
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    
    # Test parallel analysis
    result = parallel_analyze_video_and_audio(video_path, use_azure=False)
    
    print("\n" + "="*70)
    print("📊 PARALLEL ANALYSIS RESULTS")
    print("="*70)
    
    if result['separation_success']:
        print(f"🎬 Original Video: {os.path.basename(result['video_file'])}")
        print(f"🔊 Audio File: {os.path.basename(result['audio_file'])}")
        print(f"📹 Video Only: {os.path.basename(result['video_only_file'])}")
        
        print(f"\n📝 Audio Analysis:")
        print("-" * 40)
        audio_analysis = result['audio_analysis']
        if audio_analysis and audio_analysis.get('success'):
            print(f"Transcription: {audio_analysis.get('transcription', 'N/A')}")
            print(f"Audio Summary: {audio_analysis.get('summary', 'N/A')}")
        else:
            print("Audio analysis failed")
        
        print(f"\n🎭 Video Analysis:")
        print("-" * 40)
        video_analysis = result['video_analysis']
        if video_analysis and video_analysis.get('success'):
            print(f"Duration: {video_analysis.get('duration', 0):.1f}s")
            print(f"Scene Summary: {video_analysis.get('scene_summary', 'N/A')}")
            print(f"Key Objects: {', '.join(video_analysis.get('key_objects', []))}")
        else:
            print("Video analysis failed")
        
        print(f"\n🎯 Combined Summary:")
        print("-" * 40)
        print(result['combined_summary'])
    else:
        print(f"❌ Failed to process video: {result.get('error', 'Unknown error')}")
    
    print("="*70)


