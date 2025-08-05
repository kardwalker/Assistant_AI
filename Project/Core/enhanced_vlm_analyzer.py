"""
Enhanced Video Content Analyzer using Vision-Language Models (VLM)
Integrates with the existing Agent system for comprehensive video analysis
"""

import cv2
import torch
import numpy as np
from PIL import Image
import base64
import io
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
load_dotenv()
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        BlipProcessor,
        BlipForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class VLMAnalysisResult:
    """Results from VLM-based video analysis"""
    frame_descriptions: List[Dict]
    detected_objects: List[str]
    detected_actions: List[str]
    scene_summary: str
    confidence_score: float
    processing_time: float
    model_used: str
    success: bool
    error_message: Optional[str] = None

class EnhancedVLMVideoAnalyzer:
    """Enhanced Video Content Analyzer using Vision-Language Models"""
    
    def __init__(self, 
                 model_type: str = "qwen",  # "openai", "llava", "blip", "qwen"
                 model_name: Optional[str] = None,
                 max_frames: int = 30,
                 frame_interval: int = 30):
        """
        Initialize VLM-based video analyzer
        
        Args:
            model_type: Type of model to use ("openai", "llava", "blip", "qwen")
            model_name: Specific model name (optional)
            max_frames: Maximum frames to process
            frame_interval: Frame sampling interval
        """
        self.model_type = model_type
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model based on type
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        try:
            if model_type == "openai" and OPENAI_AVAILABLE:
                self._init_openai()
            elif model_type == "llava" and TRANSFORMERS_AVAILABLE:
                self._init_llava(model_name)
            elif model_type == "blip" and TRANSFORMERS_AVAILABLE:
                self._init_blip(model_name)
            elif model_type == "qwen" and TRANSFORMERS_AVAILABLE:
                self._init_qwen(model_name)
            else:
                raise ValueError(f"Model type {model_type} not available or supported")
                
            logger.info(f"✅ VLM Analyzer initialized with {model_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VLM: {e}")
            self.model = None
    
    def _init_openai(self):
        """Initialize OpenAI GPT-4V"""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        if os.getenv("AZURE_API_KEY"):
            openai.api_type = "azure"
            openai.api_key = os.getenv("AZURE_API_KEY")
            openai.api_base = os.getenv("AZURE_ENDPOINT")
            openai.api_version = "2024-02-01"
        else:
            openai.api_key = api_key
            
        self.model_used = "gpt-4-vision-preview"
    
    def _init_llava(self, model_name: Optional[str] = None):
        """Initialize LLaVA model"""
        model_name = model_name or "llava-hf/llava-1.5-7b-hf"
        
        self.processor = LlavaProcessor.from_pretrained(model_name, use_fast=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model_used = model_name
    
    def _init_blip(self, model_name: Optional[str] = None):
        """Initialize BLIP model"""
        model_name = model_name or "Salesforce/blip-image-captioning-large"
        
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model_used = model_name
    
    def _init_qwen(self, model_name: Optional[str] = None):
        """Initialize Qwen VL model"""
        model_name = model_name or "Qwen/Qwen-VL-Chat"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )
        self.model_used = model_name
    
    def extract_frames_smart(self, video_path: str) -> List[np.ndarray]:
        """
        Smart frame extraction with memory management
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📹 Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            
            # Smart sampling based on video length
            if duration > 120:  # > 2 minutes
                self.frame_interval = max(self.frame_interval, total_frames // self.max_frames)
            
            frames = []
            frame_count = 0
            
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    # Resize frame to manage memory
                    frame_resized = cv2.resize(frame, (512, 512))
                    frames.append(frame_resized)
                
                frame_count += 1
            
            logger.info(f"✅ Extracted {len(frames)} frames for analysis")
            return frames
            
        finally:
            cap.release()
    
    def analyze_frame_with_vlm(self, frame: np.ndarray, prompt: str = None) -> Dict:
        """
        Analyze single frame using VLM
        """
        if self.model is None:
            return {"description": "Model not available", "objects": [], "actions": [], "confidence": 0.0}
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            if self.model_type == "openai":
                return self._analyze_with_openai(pil_image, prompt)
            elif self.model_type == "llava":
                return self._analyze_with_llava(pil_image, prompt)
            elif self.model_type == "blip":
                return self._analyze_with_blip(pil_image, prompt)
            elif self.model_type == "qwen":
                return self._analyze_with_qwen(pil_image, prompt)
            else:
                return {"description": "Unknown model type", "objects": [], "actions": [], "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {"description": f"Analysis failed: {str(e)}", "objects": [], "actions": [], "confidence": 0.0}
    
    def _analyze_with_openai(self, image: Image.Image, prompt: str = None) -> Dict:
        """Analyze frame using OpenAI GPT-4V"""
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        system_prompt = prompt or """Analyze this video frame and provide:
1. A detailed description of what you see
2. List of objects (comma-separated)
3. List of actions or activities (comma-separated)
4. Overall scene context

Format your response as:
DESCRIPTION: [detailed description]
OBJECTS: [object1, object2, ...]
ACTIONS: [action1, action2, ...]
CONTEXT: [scene context]"""
        
        try:
            if os.getenv("AZURE_API_KEY"):
                response = openai.ChatCompletion.create(
                    engine="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    max_tokens=300
                )
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    max_tokens=300
                )
            
            content = response.choices[0].message.content
            return self._parse_vlm_response(content)
            
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return {"description": "OpenAI analysis failed", "objects": [], "actions": [], "confidence": 0.3}
    
    def _analyze_with_llava(self, image: Image.Image, prompt: str = None) -> Dict:
        """Analyze frame using LLaVA"""
        prompt = prompt or "USER: Describe this image in detail, including objects and activities. ASSISTANT:"
        
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
                
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Extract the assistant's response
            if "ASSISTANT:" in response:
                description = response.split("ASSISTANT:")[-1].strip()
            else:
                description = response.strip()
            
            # Extract objects and actions using simple heuristics
            objects, actions = self._extract_objects_actions_from_text(description)
            
            return {
                "description": description,
                "objects": objects,
                "actions": actions,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"LLaVA analysis error: {e}")
            return {"description": "LLaVA analysis failed", "objects": [], "actions": [], "confidence": 0.3}
    
    def _analyze_with_blip(self, image: Image.Image, prompt: str = None) -> Dict:
        """Analyze frame using BLIP"""
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=100, num_beams=5)
                
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Extract objects and actions
            objects, actions = self._extract_objects_actions_from_text(description)
            
            return {
                "description": description,
                "objects": objects,
                "actions": actions,
                "confidence": 0.7
            }
            
        except Exception as e:
            logger.error(f"BLIP analysis error: {e}")
            return {"description": "BLIP analysis failed", "objects": [], "actions": [], "confidence": 0.3}
    
    def _analyze_with_qwen(self, image: Image.Image, prompt: str = None) -> Dict:
        """Analyze frame using Qwen-VL"""
        prompt = prompt or "Describe this image in detail, including all visible objects and any activities or actions."
        
        try:
            # Qwen-VL specific input format
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': prompt}
            ])
            
            inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                pred = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
                
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            
            # Extract meaningful part of response
            if prompt in response:
                description = response.split(prompt)[-1].strip()
            else:
                description = response.strip()
            
            objects, actions = self._extract_objects_actions_from_text(description)
            
            return {
                "description": description,
                "objects": objects,
                "actions": actions,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Qwen-VL analysis error: {e}")
            return {"description": "Qwen-VL analysis failed", "objects": [], "actions": [], "confidence": 0.3}
    
    def _parse_vlm_response(self, content: str) -> Dict:
        """Parse structured VLM response"""
        result = {"description": "", "objects": [], "actions": [], "confidence": 0.8}
        
        try:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("DESCRIPTION:"):
                    result["description"] = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("OBJECTS:"):
                    objects_str = line.replace("OBJECTS:", "").strip()
                    result["objects"] = [obj.strip() for obj in objects_str.split(',') if obj.strip()]
                elif line.startswith("ACTIONS:"):
                    actions_str = line.replace("ACTIONS:", "").strip()
                    result["actions"] = [act.strip() for act in actions_str.split(',') if act.strip()]
            
            # Fallback if structured parsing fails
            if not result["description"]:
                result["description"] = content
                result["objects"], result["actions"] = self._extract_objects_actions_from_text(content)
                
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            result["description"] = content
            result["objects"], result["actions"] = self._extract_objects_actions_from_text(content)
        
        return result
    
    def _extract_objects_actions_from_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract objects and actions from text using keyword matching"""
        text_lower = text.lower()
        
        # Common objects
        object_keywords = [
            'person', 'people', 'man', 'woman', 'child', 'face', 'hand', 'car', 'vehicle',
            'building', 'house', 'tree', 'sky', 'water', 'chair', 'table', 'book', 'phone',
            'computer', 'screen', 'food', 'drink', 'animal', 'dog', 'cat', 'bird', 'flower'
        ]
        
        # Common actions
        action_keywords = [
            'walking', 'running', 'sitting', 'standing', 'talking', 'eating', 'drinking',
            'reading', 'writing', 'playing', 'working', 'cooking', 'driving', 'dancing',
            'jumping', 'sleeping', 'laughing', 'smiling', 'waving', 'pointing'
        ]
        
        found_objects = [obj for obj in object_keywords if obj in text_lower]
        found_actions = [act for act in action_keywords if act in text_lower]
        
        return found_objects[:8], found_actions[:5]  # Limit results
    
    def analyze_video_with_vlm(self, video_path: str) -> VLMAnalysisResult:
        """
        Comprehensive video analysis using VLM
        """
        start_time = datetime.now()
        
        try:
            # Extract frames
            frames = self.extract_frames_smart(video_path)
            
            if not frames:
                return VLMAnalysisResult(
                    frame_descriptions=[],
                    detected_objects=[],
                    detected_actions=[],
                    scene_summary="No frames extracted",
                    confidence_score=0.0,
                    processing_time=0.0,
                    model_used=self.model_used if hasattr(self, 'model_used') else self.model_type,
                    success=False,
                    error_message="No frames extracted from video"
                )
            
            # Analyze frames
            frame_descriptions = []
            all_objects = []
            all_actions = []
            
            logger.info(f"🔍 Analyzing {len(frames)} frames with {self.model_type}")
            
            for i, frame in enumerate(frames):
                logger.info(f"Processing frame {i+1}/{len(frames)}")
                
                frame_analysis = self.analyze_frame_with_vlm(frame)
                frame_descriptions.append({
                    'frame_number': i,
                    'timestamp': i * (self.frame_interval / 30.0),  # Approximate timestamp
                    'description': frame_analysis['description'],
                    'objects': frame_analysis['objects'],
                    'actions': frame_analysis['actions'],
                    'confidence': frame_analysis['confidence']
                })
                
                all_objects.extend(frame_analysis['objects'])
                all_actions.extend(frame_analysis['actions'])
            
            # Generate scene summary
            scene_summary = self._generate_comprehensive_summary(frame_descriptions)
            
            # Get unique objects and actions with frequency
            unique_objects = list(set(all_objects))
            unique_actions = list(set(all_actions))
            
            # Calculate overall confidence
            confidences = [fd['confidence'] for fd in frame_descriptions]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VLMAnalysisResult(
                frame_descriptions=frame_descriptions,
                detected_objects=unique_objects,
                detected_actions=unique_actions,
                scene_summary=scene_summary,
                confidence_score=avg_confidence,
                processing_time=processing_time,
                model_used=self.model_used if hasattr(self, 'model_used') else self.model_type,
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"VLM video analysis error: {e}")
            
            return VLMAnalysisResult(
                frame_descriptions=[],
                detected_objects=[],
                detected_actions=[],
                scene_summary=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                model_used=self.model_used if hasattr(self, 'model_used') else self.model_type,
                success=False,
                error_message=str(e)
            )
    
    def _generate_comprehensive_summary(self, frame_descriptions: List[Dict]) -> str:
        """Generate comprehensive summary from frame analyses"""
        if not frame_descriptions:
            return "No frame descriptions available"
        
        # Extract key information
        descriptions = [fd['description'] for fd in frame_descriptions]
        all_objects = []
        all_actions = []
        
        for fd in frame_descriptions:
            all_objects.extend(fd['objects'])
            all_actions.extend(fd['actions'])
        
        # Get most common objects and actions
        from collections import Counter
        object_counts = Counter(all_objects)
        action_counts = Counter(all_actions)
        
        top_objects = [obj for obj, count in object_counts.most_common(8)]
        top_actions = [act for act, count in action_counts.most_common(5)]
        
        # Generate summary
        summary_parts = []
        
        if descriptions:
            # Use first, middle, and last descriptions for progression
            if len(descriptions) >= 3:
                summary_parts.append(f"Video progression: {descriptions[0]} → {descriptions[len(descriptions)//2]} → {descriptions[-1]}")
            else:
                summary_parts.append(f"Video content: {' | '.join(descriptions[:2])}")
        
        if top_objects:
            summary_parts.append(f"Key objects: {', '.join(top_objects)}")
        
        if top_actions:
            summary_parts.append(f"Main activities: {', '.join(top_actions)}")
        
        return " || ".join(summary_parts)


# Integration with existing Agent system
def integrate_vlm_analyzer_with_agent():
    """Example integration with the existing Agent system"""
    
    # This would replace or enhance the existing video_analyzer in Agent.py
    class VLMEnhancedVideoAnalyzer:
        def __init__(self, model_type="qwen"):
            self.vlm_analyzer = EnhancedVLMVideoAnalyzer(model_type=model_type)
        
        def analyze_video(self, video_path: str):
            """Compatible interface with existing Agent system"""
            result = self.vlm_analyzer.analyze_video_with_vlm(video_path)
            
            # Convert to format expected by Agent
            if result.success:
                return type('AnalysisResult', (), {
                    'summary': result.scene_summary,
                    'events': [
                        {
                            'timestamp': fd['timestamp'],
                            'type': 'scene_description',
                            'description': fd['description'],
                            'confidence': fd['confidence'],
                            'guideline_adherence': True
                        }
                        for fd in result.frame_descriptions
                    ]
                })()
            else:
                return None


# Example usage
if __name__ == "__main__":
    # Test different models - Qwen as primary, with fallbacks
    models_to_test = ["qwen", "blip", "openai"]  # Qwen first as default
    
    video_path = input("Video input")  # Replace with actual video
    
    for model_type in models_to_test:
        print(f"\n🧪 Testing {model_type.upper()} VLM Analysis")
        print("=" * 50)
        
        try:
            analyzer = EnhancedVLMVideoAnalyzer(model_type=model_type, max_frames=10)
            result = analyzer.analyze_video_with_vlm(video_path)
            
            if result.success:
                print(f"✅ Analysis successful using {result.model_used}")
                print(f"📊 Confidence: {result.confidence_score:.2f}")
                print(f"⏱️  Processing time: {result.processing_time:.1f}s")
                print(f"🎬 Summary: {result.scene_summary}")
                print(f"📦 Objects: {', '.join(result.detected_objects[:10])}")
                print(f"🏃 Actions: {', '.join(result.detected_actions[:8])}")
            else:
                print(f"❌ Analysis failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")

