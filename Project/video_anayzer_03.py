"""
Enhanced Video Content Analyzer with Event Recognition and Temporal Analysis
Processes video files to extract visual information, detect objects, scenes, actions, and events
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import json
from dotenv import load_dotenv
load_dotenv()
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
    """Represents a video frame with metadata and temporal context"""
    frame_number: int
    timestamp: float
    description: str
    confidence: float
    objects_detected: List[str] = field(default_factory=list)
    motion_vector: Tuple[float, float] = (0.0, 0.0)
    scene_change_score: float = 0.0
    brightness: float = 0.0
    visual_features: Dict = field(default_factory=dict)

@dataclass
class VideoEvent:
    """Represents a detected event in the video"""
    event_type: str
    start_time: float
    end_time: float
    confidence: float
    description: str
    key_frames: List[int] = field(default_factory=list)
    objects_involved: List[str] = field(default_factory=list)
    motion_intensity: float = 0.0

@dataclass
class VideoScene:
    """Represents a coherent scene in the video"""
    scene_id: int
    start_time: float
    end_time: float
    description: str
    dominant_objects: List[str] = field(default_factory=list)
    avg_brightness: float = 0.0
    motion_pattern: str = "static"
    key_events: List[VideoEvent] = field(default_factory=list)

@dataclass
class VideoAnalysisResult:
    """Enhanced results from video content analysis"""
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
    # Enhanced fields
    detected_events: List[VideoEvent] = field(default_factory=list)
    scene_segments: List[VideoScene] = field(default_factory=list)
    temporal_patterns: Dict = field(default_factory=dict)
    object_timeline: Dict = field(default_factory=dict)
    motion_analysis: Dict = field(default_factory=dict)
    detailed_summary: str = ""  # Add this line

class EnhancedVideoContentAnalyzer:
    def __init__(self, use_ai_model=True, use_api=True, event_detection_threshold=0.7):
        """
        Initialize enhanced video content analyzer with event recognition
        
        Args:
            use_ai_model (bool): Whether to use AI models for image captioning
            use_api (bool): Whether to use API services (faster than local models)
            event_detection_threshold (float): Confidence threshold for event detection
        """
        self.use_api = use_api and OPENAI_AVAILABLE and (OPENAI_API_KEY or AZURE_API_KEY)
        self.use_ai_model = use_ai_model and BLIP_AVAILABLE and TORCH_AVAILABLE
        self.event_threshold = event_detection_threshold
        
        # Initialize motion tracking
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        
        # Event detection patterns
        self.event_patterns = {
            'scene_change': {'motion_threshold': 0.3, 'visual_threshold': 0.4},
            'object_appearance': {'confidence_threshold': 0.8},
            'motion_event': {'motion_threshold': 0.5, 'duration_threshold': 1.0},
            'lighting_change': {'brightness_threshold': 30.0},
            'activity_peak': {'motion_threshold': 0.7, 'object_threshold': 3}
        }
        
        if self.use_api:
            print("🚀 Using OpenAI API for enhanced video analysis with event recognition")
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
            print("⏳ Loading BLIP model for enhanced image analysis...")
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("✅ BLIP model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load BLIP model: {e}")
                self.use_ai_model = False
                self.processor = None
                self.model = None
        else:
            print("🔧 Using enhanced computer vision with event detection")
            self.processor = None
            self.model = None
    
    def extract_frames_with_temporal_info(self, video_path: str, frame_interval: int = 30, 
                                        enhanced_mode: bool = False) -> List[Dict]:
        """
        Extract frames with temporal and motion information for event detection
        """
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames_data
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adaptive frame extraction based on content
        if enhanced_mode:
            frame_interval = max(10, frame_interval // 3)  # More frequent for event detection
            max_frames = 50
        else:
            max_frames = 30
        
        print(f"📹 Extracting frames with temporal analysis (Total: {total_frames}, FPS: {fps:.2f})")
        
        prev_gray = None
        motion_accumulator = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps if fps > 0 else 0
            
            if frame_count % frame_interval == 0:
                # Calculate motion and visual features
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                motion_magnitude = 0.0
                scene_change_score = 0.0
                
                if prev_gray is not None:
                    # Optical flow for motion detection
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, 
                        np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                        None
                    )[0]
                    if flow is not None and len(flow) > 0:
                        motion_magnitude = np.linalg.norm(flow[0][0] - [100, 100])
                    
                    # Scene change detection using histogram comparison
                    hist1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    scene_change_score = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                # Extract visual features
                brightness = np.mean(gray)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
                
                # Color analysis
                b, g, r = cv2.split(frame)
                color_variance = np.var([np.mean(b), np.mean(g), np.mean(r)])
                
                frames_data.append({
                    'frame': frame,
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'motion_magnitude': motion_magnitude,
                    'scene_change_score': scene_change_score,
                    'brightness': brightness,
                    'edge_density': edge_density,
                    'color_variance': color_variance,
                    'visual_features': {
                        'brightness': brightness,
                        'edge_density': edge_density,
                        'color_variance': color_variance
                    }
                })
                
                motion_accumulator.append(motion_magnitude)
                prev_gray = gray.copy()
            
            frame_count += 1
            
            if len(frames_data) >= max_frames:
                break
        
        cap.release()
        
        # Post-process motion data for better event detection
        if motion_accumulator:
            motion_mean = np.mean(motion_accumulator)
            motion_std = np.std(motion_accumulator)
            
            for i, frame_data in enumerate(frames_data):
                if i < len(motion_accumulator):
                    # Normalize motion relative to video's motion profile
                    normalized_motion = (motion_accumulator[i] - motion_mean) / (motion_std + 1e-6)
                    frame_data['normalized_motion'] = normalized_motion
        
        print(f"✅ Extracted {len(frames_data)} frames with temporal features")
        return frames_data
    
    def describe_frame_with_context(self, frame: np.ndarray, context: Dict) -> Tuple[str, List[str]]:
        """
        Generate description with temporal context for better event recognition
        """
        if self.use_api:
            return self.describe_frame_api_enhanced(frame, context)
        elif self.use_ai_model:
            return self.describe_frame_local_ai_enhanced(frame, context)
        else:
            return self.describe_frame_enhanced_cv(frame, context)
    
    def describe_frame_api_enhanced(self, frame: np.ndarray, context: Dict) -> Tuple[str, List[str]]:
        """Enhanced API-based frame description with object detection focus"""
        try:
            import base64
            import io
            from PIL import Image
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Enhanced prompt for event detection
            motion_context = "high motion" if context.get('normalized_motion', 0) > 0.5 else "low motion"
            brightness_context = "bright" if context.get('brightness', 100) > 150 else "dark"
            
            prompt = f"""Analyze this video frame in detail. Context: {motion_context}, {brightness_context} scene.
            
            Provide:
            1. A detailed description of what's happening
            2. List all visible objects, people, and activities
            3. Identify any actions or movements
            4. Note the setting/environment
            
            Format: [Description] | Objects: [object1, object2, ...]"""
            
            if AZURE_API_KEY:
                response = openai.ChatCompletion.create(
                    engine="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    max_tokens=150
                )
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    max_tokens=150
                )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response to extract description and objects
            if " | Objects: " in response_text:
                description, objects_str = response_text.split(" | Objects: ", 1)
                objects = [obj.strip() for obj in objects_str.split(",") if obj.strip()]
            else:
                description = response_text
                objects = self._extract_objects_from_text(response_text)
            
            return description, objects
            
        except Exception as e:
            print(f"Enhanced API description error: {e}")
            return self.describe_frame_enhanced_cv(frame, context)
    
    def describe_frame_enhanced_cv(self, frame: np.ndarray, context: Dict) -> Tuple[str, List[str]]:
        """Enhanced computer vision analysis with object and activity detection"""
        try:
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced visual analysis
            brightness = context.get('brightness', np.mean(gray))
            edge_density = context.get('edge_density', 0)
            motion_magnitude = context.get('motion_magnitude', 0)
            normalized_motion = context.get('normalized_motion', 0)
            
            # Object detection using basic CV
            detected_objects = []
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                detected_objects.extend(['person', 'face'] * min(len(faces), 3))
            
            # Edge-based object approximation
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for object shapes
            large_objects = [c for c in contours if cv2.contourArea(c) > width * height * 0.01]
            
            if len(large_objects) > 0:
                detected_objects.append('object')
            if len(large_objects) > 3:
                detected_objects.extend(['multiple_objects', 'busy_scene'])
            
            # Color analysis for object inference
            b, g, r = cv2.split(frame)
            color_dominance = {
                'blue': np.mean(b),
                'green': np.mean(g), 
                'red': np.mean(r)
            }
            dominant_color = max(color_dominance, key=color_dominance.get)
            
            # Infer objects from color patterns
            if color_dominance['green'] > 100 and color_dominance['green'] > color_dominance['blue']:
                detected_objects.extend(['vegetation', 'outdoor'])
            if color_dominance['blue'] > 120:
                detected_objects.extend(['sky', 'water'])
            
            # Activity and motion analysis
            activity_level = "static"
            if normalized_motion > 1.0:
                activity_level = "high_activity"
                detected_objects.extend(['movement', 'action'])
            elif normalized_motion > 0.3:
                activity_level = "moderate_activity"
                detected_objects.append('movement')
            
            # Lighting analysis
            lighting = "normal"
            if brightness > 200:
                lighting = "bright"
                detected_objects.append('bright_lighting')
            elif brightness < 80:
                lighting = "dark"
                detected_objects.append('low_lighting')
            
            # Scene complexity
            complexity = "simple"
            if edge_density > 0.15:
                complexity = "complex"
                detected_objects.extend(['detailed_scene', 'complex_environment'])
            elif edge_density > 0.08:
                complexity = "moderate"
                detected_objects.append('moderate_detail')
            
            # Generate enhanced description
            description = f"A {lighting} {complexity} scene with {dominant_color} tones showing {activity_level}"
            
            if len(faces) > 0:
                description += f" featuring {len(faces)} person(s)"
            
            if motion_magnitude > 0.5:
                description += " with visible movement"
            
            if len(large_objects) > 5:
                description += " containing multiple distinct objects"
            
            # Remove duplicates from detected objects
            detected_objects = list(set(detected_objects))
            
            return description, detected_objects[:10]  # Limit to top 10 objects
            
        except Exception as e:
            print(f"Enhanced CV analysis error: {e}")
            return "Unable to analyze frame content", []
    
    def _extract_objects_from_text(self, text: str) -> List[str]:
        """Extract object names from description text"""
        # Common objects to look for in descriptions
        common_objects = [
            'person', 'people', 'man', 'woman', 'child', 'face', 'hand',
            'car', 'vehicle', 'building', 'house', 'tree', 'sky', 'water',
            'chair', 'table', 'book', 'phone', 'computer', 'screen',
            'food', 'drink', 'animal', 'dog', 'cat', 'bird',
            'road', 'street', 'outdoor', 'indoor', 'room', 'kitchen'
        ]
        
        text_lower = text.lower()
        found_objects = []
        
        for obj in common_objects:
            if obj in text_lower:
                found_objects.append(obj)
        
        return found_objects[:8]  # Limit to 8 objects
    
    def detect_events_from_frames(self, frames_data: List[Dict]) -> List[VideoEvent]:
        """
        Detect events by analyzing temporal patterns in frame data
        """
        events = []
        
        if len(frames_data) < 2:
            return events
        
        # Event detection algorithms
        events.extend(self._detect_scene_changes(frames_data))
        events.extend(self._detect_motion_events(frames_data))
        events.extend(self._detect_lighting_changes(frames_data))
        events.extend(self._detect_object_events(frames_data))
        
        # Sort events by start time
        events.sort(key=lambda x: x.start_time)
        
        # Merge overlapping events
        events = self._merge_overlapping_events(events)
        
        print(f"🎯 Detected {len(events)} events in video")
        return events
    
    def _detect_scene_changes(self, frames_data: List[Dict]) -> List[VideoEvent]:
        """Detect scene change events"""
        events = []
        threshold = self.event_patterns['scene_change']['visual_threshold']
        
        for i in range(1, len(frames_data)):
            current_frame = frames_data[i]
            scene_change_score = current_frame.get('scene_change_score', 0)
            
            if scene_change_score > threshold:
                events.append(VideoEvent(
                    event_type="scene_change",
                    start_time=current_frame['timestamp'] - 0.5,
                    end_time=current_frame['timestamp'] + 0.5,
                    confidence=min(scene_change_score * 2, 1.0),
                    description=f"Scene transition detected at {current_frame['timestamp']:.1f}s",
                    key_frames=[current_frame['frame_number']]
                ))
        
        return events
    
    def _detect_motion_events(self, frames_data: List[Dict]) -> List[VideoEvent]:
        """Detect motion-based events"""
        events = []
        threshold = self.event_patterns['motion_event']['motion_threshold']
        min_duration = self.event_patterns['motion_event']['duration_threshold']
        
        # Find periods of high motion
        high_motion_periods = []
        current_period = None
        
        for frame_data in frames_data:
            normalized_motion = frame_data.get('normalized_motion', 0)
            
            if normalized_motion > threshold:
                if current_period is None:
                    current_period = {
                        'start_time': frame_data['timestamp'],
                        'end_time': frame_data['timestamp'],
                        'max_motion': normalized_motion,
                        'frames': [frame_data['frame_number']]
                    }
                else:
                    current_period['end_time'] = frame_data['timestamp']
                    current_period['max_motion'] = max(current_period['max_motion'], normalized_motion)
                    current_period['frames'].append(frame_data['frame_number'])
            else:
                if current_period is not None:
                    if current_period['end_time'] - current_period['start_time'] >= min_duration:
                        high_motion_periods.append(current_period)
                    current_period = None
        
        # Add final period if exists
        if current_period is not None:
            if current_period['end_time'] - current_period['start_time'] >= min_duration:
                high_motion_periods.append(current_period)
        
        # Create motion events
        for period in high_motion_periods:
            events.append(VideoEvent(
                event_type="high_motion",
                start_time=period['start_time'],
                end_time=period['end_time'],
                confidence=min(period['max_motion'], 1.0),
                description=f"High motion activity from {period['start_time']:.1f}s to {period['end_time']:.1f}s",
                key_frames=period['frames'],
                motion_intensity=period['max_motion']
            ))
        
        return events
    
    def _detect_lighting_changes(self, frames_data: List[Dict]) -> List[VideoEvent]:
        """Detect significant lighting changes"""
        events = []
        threshold = self.event_patterns['lighting_change']['brightness_threshold']
        
        for i in range(1, len(frames_data)):
            prev_brightness = frames_data[i-1].get('brightness', 0)
            curr_brightness = frames_data[i].get('brightness', 0)
            
            brightness_change = abs(curr_brightness - prev_brightness)
            
            if brightness_change > threshold:
                change_type = "brightness_increase" if curr_brightness > prev_brightness else "brightness_decrease"
                
                events.append(VideoEvent(
                    event_type="lighting_change",
                    start_time=frames_data[i]['timestamp'] - 0.3,
                    end_time=frames_data[i]['timestamp'] + 0.3,
                    confidence=min(brightness_change / 100.0, 1.0),
                    description=f"Lighting change ({change_type}) at {frames_data[i]['timestamp']:.1f}s",
                    key_frames=[frames_data[i]['frame_number']]
                ))
        
        return events
    
    def _detect_object_events(self, frames_data: List[Dict]) -> List[VideoEvent]:
        """Detect object appearance/disappearance events based on descriptions"""
        events = []
        
        # This would be enhanced with actual object tracking
        # For now, we'll detect based on description changes
        for i in range(1, len(frames_data)):
            prev_frame = frames_data[i-1]
            curr_frame = frames_data[i]
            
            # Placeholder for object tracking logic
            # In a full implementation, you'd track specific objects frame by frame
            edge_change = abs(curr_frame.get('edge_density', 0) - prev_frame.get('edge_density', 0))
            
            if edge_change > 0.05:  # Significant change in visual complexity
                events.append(VideoEvent(
                    event_type="visual_change",
                    start_time=curr_frame['timestamp'] - 0.2,
                    end_time=curr_frame['timestamp'] + 0.2,
                    confidence=min(edge_change * 10, 1.0),
                    description=f"Visual content change at {curr_frame['timestamp']:.1f}s",
                    key_frames=[curr_frame['frame_number']]
                ))
        
        return events
    
    def _merge_overlapping_events(self, events: List[VideoEvent]) -> List[VideoEvent]:
        """Merge events that overlap in time"""
        if not events:
            return events
        
        merged = []
        current_event = events[0]
        
        for next_event in events[1:]:
            # Check if events overlap
            if (next_event.start_time <= current_event.end_time and 
                next_event.event_type == current_event.event_type):
                # Merge events
                current_event.end_time = max(current_event.end_time, next_event.end_time)
                current_event.confidence = max(current_event.confidence, next_event.confidence)
                current_event.key_frames.extend(next_event.key_frames)
                current_event.description += f" + {next_event.description}"
            else:
                merged.append(current_event)
                current_event = next_event
        
        merged.append(current_event)
        return merged
    
    def segment_video_into_scenes(self, frames_data: List[Dict], events: List[VideoEvent]) -> List[VideoScene]:
        """
        Segment video into coherent scenes based on events and frame analysis
        """
        if not frames_data:
            return []
        
        scenes = []
        scene_boundaries = [0]  # Start with first frame
        
        # Find scene boundaries based on scene change events
        scene_change_events = [e for e in events if e.event_type == "scene_change"]
        for event in scene_change_events:
            # Find frame index closest to event time
            event_frame_idx = min(range(len(frames_data)), 
                                key=lambda i: abs(frames_data[i]['timestamp'] - event.start_time))
            if event_frame_idx not in scene_boundaries:
                scene_boundaries.append(event_frame_idx)
        
        scene_boundaries.append(len(frames_data) - 1)  # End with last frame
        scene_boundaries.sort()
        
        # Create scenes
        for i in range(len(scene_boundaries) - 1):
            start_idx = scene_boundaries[i]
            end_idx = scene_boundaries[i + 1]
            
            scene_frames = frames_data[start_idx:end_idx + 1]
            if not scene_frames:
                continue
            
            # Analyze scene characteristics
            avg_brightness = np.mean([f.get('brightness', 0) for f in scene_frames])
            motion_values = [f.get('normalized_motion', 0) for f in scene_frames]
            avg_motion = np.mean(motion_values)
            max_motion = max(motion_values) if motion_values else 0
            
            # Determine motion pattern
            if max_motion > 1.0:
                motion_pattern = "dynamic"
            elif avg_motion > 0.3:
                motion_pattern = "moderate"
            else:
                motion_pattern = "static"
            
            # Find events in this scene
            scene_start_time = scene_frames[0]['timestamp']
            scene_end_time = scene_frames[-1]['timestamp']
            scene_events = [e for e in events 
                          if e.start_time >= scene_start_time and e.end_time <= scene_end_time]
            
            # Generate scene description (placeholder - would use actual frame descriptions)
            if avg_brightness > 150:
                lighting = "bright"
            elif avg_brightness < 100:
                lighting = "dark"
            else:
                lighting = "normal"
            
            description = f"Scene {i+1}: {lighting} {motion_pattern} sequence"
            if scene_events:
                description += f" with {len(scene_events)} detected events"
            
            scenes.append(VideoScene(
                scene_id=i + 1,
                start_time=scene_start_time,
                end_time=scene_end_time,
                description=description,
                avg_brightness=avg_brightness,
                motion_pattern=motion_pattern,
                key_events=scene_events
            ))
        
        print(f"🎬 Segmented video into {len(scenes)} scenes")
        return scenes
    
    def generate_detailed_summary(self, result: VideoAnalysisResult) -> str:
        """
        Generate a comprehensive narrative summary of the video content
        including scene descriptions, event analysis, and temporal patterns
        """
        if not result.success:
            return "Video analysis failed. Unable to generate summary."
        
        # 1. Video metadata summary
        duration_min = result.duration / 60
        summary = [
            f"📽️ Video Analysis Report: {os.path.basename(result.video_file)}",
            f"⏱️ Duration: {result.duration:.1f} seconds ({duration_min:.1f} minutes)",
            f"🎞️ Total Frames: {result.total_frames}",
            f"🔍 Analysis Method: {result.analysis_method} (Confidence: {result.confidence_score*100:.1f}%)",
            f"🧩 Scenes Detected: {len(result.scene_segments)}",
            f"🎯 Key Events Identified: {len(result.detected_events)}",
            ""
        ]
        
        # 2. Overall video characterization
        avg_brightness = np.mean([f.brightness for f in result.frame_descriptions]) if result.frame_descriptions else 0
        brightness_desc = "bright" if avg_brightness > 180 else "well-lit" if avg_brightness > 120 else "dimly-lit"
        
        motion_values = [f.motion_vector[0] for f in result.frame_descriptions] if result.frame_descriptions else []
        avg_motion = np.mean(motion_values) if motion_values else 0
        motion_desc = "dynamic" if avg_motion > 0.7 else "active" if avg_motion > 0.4 else "static"
        
        summary.append(f"🌅 Overall: A {brightness_desc}, {motion_desc} video with {len(result.key_objects)} key visual elements")
        
        # 3. Scene-by-scene breakdown
        if result.scene_segments:
            summary.append("\n🎬 SCENE BREAKDOWN:")
            for scene in result.scene_segments[:5]:  # Limit to first 5 scenes
                scene_duration = scene.end_time - scene.start_time
                event_count = len(scene.key_events)
                
                scene_summary = [
                    f"\nScene {scene.scene_id} ({scene.start_time:.1f}s-{scene.end_time:.1f}s, {scene_duration:.1f}s):",
                    f"  - {scene.description}",
                    f"  - Motion Pattern: {scene.motion_pattern.capitalize()}",
                    f"  - Brightness: {scene.avg_brightness:.1f} ({'bright' if scene.avg_brightness > 160 else 'normal' if scene.avg_brightness > 100 else 'dark'})"
                ]
                
                if event_count > 0:
                    scene_summary.append(f"  - Key Events: {event_count} significant events")
                    for event in scene.key_events[:3]:  # Show top 3 events
                        event_desc = f"    • {event.event_type.replace('_', ' ').title()} at {event.start_time:.1f}s"
                        if event.description:
                            event_desc += f": {event.description}"
                        scene_summary.append(event_desc)
                
                if scene.dominant_objects:
                    scene_summary.append(f"  - Dominant Objects: {', '.join(scene.dominant_objects[:5])}")
                
                summary.extend(scene_summary)
        
        # 4. Key event highlights
        if result.detected_events:
            summary.append("\n⚡ KEY EVENT HIGHLIGHTS:")
            event_types = defaultdict(list)
            for event in result.detected_events:
                event_types[event.event_type].append(event)
            
            for event_type, events in event_types.items():
                count = len(events)
                first_time = min(e.start_time for e in events) if events else 0
                last_time = max(e.end_time for e in events) if events else 0
                avg_duration = np.mean([e.end_time - e.start_time for e in events]) if events else 0
                avg_confidence = np.mean([e.confidence for e in events]) if events else 0
                
                summary.append(
                    f"  - {event_type.replace('_', ' ').title()} ({count}x): "
                    f"First at {first_time:.1f}s, last at {last_time:.1f}s, "
                    f"avg {avg_duration:.1f}s duration ({avg_confidence*100:.1f}% confidence)"
                )
        
        # 5. Object and character analysis
        if result.object_timeline:
            summary.append("\n🔍 OBJECT & CHARACTER ANALYSIS:")
            # Get top 10 objects by appearance frequency
            sorted_objects = sorted(
                result.object_timeline.items(),
                key=lambda x: x[1]['total_appearances'],
                reverse=True
            )[:10]
            
            for obj, timeline in sorted_objects:
                duration = timeline['last_appearance'] - timeline['first_appearance']
                presence = duration / result.duration * 100 if result.duration > 0 else 0
                summary.append(
                    f"  - {obj.title()}: "
                    f"Appeared {timeline['total_appearances']} times, "
                    f"present for {duration:.1f}s ({presence:.1f}% of video)"
                )
        
        # 6. Motion and temporal patterns
        summary.append("\n📈 TEMPORAL PATTERNS:")
        motion = result.motion_analysis
        if motion:
            summary.extend([
                f"  - Motion Trend: {motion.get('motion_trend', 'unknown').capitalize()}",
                f"  - Motion Distribution: "
                f"Low: {motion.get('motion_distribution', {}).get('low_motion_percentage', 0):.1f}%, "
                f"Medium: {motion.get('motion_distribution', {}).get('medium_motion_percentage', 0):.1f}%, "
                f"High: {motion.get('motion_distribution', {}).get('high_motion_percentage', 0):.1f}%"
            ])
        
        if 'scene_change_rate' in result.temporal_patterns:
            summary.append(
                f"  - Scene Change Frequency: "
                f"{result.temporal_patterns['scene_change_rate']:.2f} changes per second"
            )
        
        # 7. Narrative summary
        summary.append("\n📖 NARRATIVE SUMMARY:")
        if result.scene_segments:
            opening = result.scene_segments[0].description
            closing = result.scene_segments[-1].description
            
            # Find the most dynamic scene
            dynamic_scenes = [s for s in result.scene_segments if s.motion_pattern != "static"]
            most_dynamic_scene = None
            if dynamic_scenes:
                most_dynamic_scene = max(
                    dynamic_scenes,
                    key=lambda s: len(s.key_events),
                )
            
            # Find longest scene
            longest_scene = max(result.scene_segments, key=lambda s: s.end_time - s.start_time)
            
            narrative = [
                f"The video opens with {opening.lower()}.",
                f"It progresses through {len(result.scene_segments)} distinct scenes."
            ]
            
            if most_dynamic_scene:
                narrative.append(
                    f"The most dynamic sequence occurs between "
                    f"{most_dynamic_scene.start_time:.1f}-{most_dynamic_scene.end_time:.1f}s "
                    f"featuring {len(most_dynamic_scene.key_events)} significant events."
                )
            
            narrative.append(
                f"The longest scene ({longest_scene.end_time - longest_scene.start_time:.1f}s) "
                f"is scene {longest_scene.scene_id} showing {longest_scene.description.lower()}."
            )
            
            narrative.append(f"The video concludes with {closing.lower()}.")
            
            summary.extend(narrative)
        else:
            summary.append(result.scene_summary)
        
        return "\n".join(summary)
    
    def analyze_video_content_enhanced(self, video_path: str, enhanced_mode: bool = False) -> VideoAnalysisResult:
        """
        Enhanced video content analysis with event recognition and temporal understanding
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
        
        print(f"🎬 Enhanced video analysis: {os.path.basename(video_path)}")
        if enhanced_mode:
            print("🎵 Enhanced mode with event recognition enabled")
        
        # Extract frames with temporal information
        frames_data = self.extract_frames_with_temporal_info(video_path, frame_interval=30, enhanced_mode=enhanced_mode)
        
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
        
        # Analyze frames with context
        frame_descriptions = []
        all_objects = []
        
        print("🔍 Analyzing frames with temporal context...")
        
        # Process frames with reduced parallelism for stability
        max_workers = 2 if self.use_api else 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(self.describe_frame_with_context, frame_data['frame'], frame_data): frame_data 
                for frame_data in frames_data
            }
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_data = future_to_frame[future]
                try:
                    description, objects = future.result(timeout=30)
                    
                    frame_descriptions.append(VideoFrame(
                        frame_number=frame_data['frame_number'],
                        timestamp=frame_data['timestamp'],
                        description=description,
                        confidence=0.8,
                        objects_detected=objects,
                        motion_vector=(frame_data.get('motion_magnitude', 0), 0),
                        scene_change_score=frame_data.get('scene_change_score', 0),
                        brightness=frame_data.get('brightness', 0),
                        visual_features=frame_data.get('visual_features', {})
                    ))
                    
                    all_objects.extend(objects)
                    completed_count += 1
                    print(f"✅ Analyzed frame {completed_count}/{len(frames_data)}")
                    
                except Exception as e:
                    print(f"❌ Frame analysis error: {e}")
                    # Fallback to basic analysis
                    basic_desc, basic_objects = self.describe_frame_enhanced_cv(frame_data['frame'], frame_data)
                    frame_descriptions.append(VideoFrame(
                        frame_number=frame_data['frame_number'],
                        timestamp=frame_data['timestamp'],
                        description=basic_desc,
                        confidence=0.5,
                        objects_detected=basic_objects,
                        motion_vector=(frame_data.get('motion_magnitude', 0), 0),
                        scene_change_score=frame_data.get('scene_change_score', 0),
                        brightness=frame_data.get('brightness', 0),
                        visual_features=frame_data.get('visual_features', {})
                    ))
                    all_objects.extend(basic_objects)
        
        # Sort frames by timestamp
        frame_descriptions.sort(key=lambda x: x.timestamp)
        
        # Event detection
        print("🎯 Detecting events and patterns...")
        detected_events = self.detect_events_from_frames(frames_data)
        
        # Scene segmentation
        print("🎬 Segmenting video into scenes...")
        scene_segments = self.segment_video_into_scenes(frames_data, detected_events)
        
        # Enhanced scene summary generation
        scene_summary = self.generate_enhanced_scene_summary(frame_descriptions, detected_events, scene_segments, enhanced_mode)
        
        # Extract key objects with frequency analysis
        key_objects = self.extract_enhanced_key_objects(all_objects, detected_events)
        
        # Generate temporal patterns analysis
        temporal_patterns = self.analyze_temporal_patterns(frame_descriptions, detected_events)
        
        # Create object timeline
        object_timeline = self.create_object_timeline(frame_descriptions)
        
        # Motion analysis summary
        motion_analysis = self.analyze_motion_patterns(frame_descriptions, detected_events)
        
        # Generate detailed summary
        detailed_summary = self.generate_detailed_summary(
            VideoAnalysisResult(
                video_file=video_path,
                total_frames=total_frames,
                duration=duration,
                frame_descriptions=frame_descriptions,
                scene_summary=scene_summary,
                key_objects=key_objects,
                success=True,
                analysis_method="enhanced_temporal",
                confidence_score=0.85,
                detected_events=detected_events,
                scene_segments=scene_segments,
                temporal_patterns=temporal_patterns,
                object_timeline=object_timeline,
                motion_analysis=motion_analysis
            )
        )
        
        return VideoAnalysisResult(
            video_file=video_path,
            total_frames=total_frames,
            duration=duration,
            frame_descriptions=frame_descriptions,
            scene_summary=scene_summary,
            key_objects=key_objects,
            success=True,
            analysis_method="enhanced_temporal",
            confidence_score=0.85,
            detected_events=detected_events,
            scene_segments=scene_segments,
            temporal_patterns=temporal_patterns,
            object_timeline=object_timeline,
            motion_analysis=motion_analysis,
            detailed_summary=detailed_summary  # Add this
        )
    
    # ... rest of the class remains unchanged ...

# Update enhanced_analyze_video_only function
def enhanced_analyze_video_only(video_only_path: str, enhanced_mode: bool = False) -> Dict:
    try:
        analyzer = EnhancedVideoContentAnalyzer(use_ai_model=True, use_api=True)
        result = analyzer.analyze_video_content_enhanced(video_only_path, enhanced_mode=enhanced_mode)
        
        return {
            'success': result.success,
            'duration': result.duration,
            'total_frames': result.total_frames,
            'scene_summary': result.scene_summary,
            'key_objects': result.key_objects,
            'frame_count': len(result.frame_descriptions),
            'analysis_method': result.analysis_method,
            'enhanced_mode': enhanced_mode,
            'confidence_score': result.confidence_score,
            # Enhanced fields
            'detected_events': [
                {
                    'type': e.event_type,
                    'start_time': e.start_time,
                    'end_time': e.end_time,
                    'confidence': e.confidence,
                    'description': e.description
                } for e in result.detected_events
            ],
            'scene_segments': [
                {
                    'scene_id': s.scene_id,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'description': s.description,
                    'motion_pattern': s.motion_pattern
                } for s in result.scene_segments
            ],
            'temporal_patterns': result.temporal_patterns,
            'object_timeline': result.object_timeline,
            'motion_analysis': result.motion_analysis,
            'detailed_summary': result.detailed_summary  # Add this
        }
    except Exception as e:
        print(f"❌ Enhanced video analysis error: {e}")
        return {
            'success': False,
            'duration': 0,
            'total_frames': 0,
            'scene_summary': f'Enhanced video analysis failed: {str(e)}',
            'key_objects': [],
            'frame_count': 0,
            'analysis_method': 'Failed',
            'enhanced_mode': enhanced_mode,
            'error': str(e)
        }

# ... rest of the code remains unchanged ...