"""
Enhanced Video Content Analyzer using Vision-Language Models (VLM)
Improved version with temporal event detection, intelligent sampling, and advanced summarization
"""

import cv2
import torch
import numpy as np
from PIL import Image
import base64
import io
import asyncio
import hashlib
import pickle
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Import dependencies with fallbacks
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

logger = logging.getLogger(__name__)

@dataclass
class TemporalEvent:
    """Enhanced event representation with temporal information"""
    event_type: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    duration: float
    confidence: float
    participants: List[str] = field(default_factory=list)
    location: str = ""
    intensity: float = 0.5
    objects_involved: List[str] = field(default_factory=list)
    emotional_context: str = ""
    event_category: str = ""

@dataclass
class SceneTransition:
    """Scene transition information"""
    frame: int
    timestamp: float
    transition_type: str
    confidence: float
    description: str

@dataclass
class EnhancedAnalysisResult:
    """Comprehensive analysis results"""
    frame_descriptions: List[Dict]
    temporal_events: List[TemporalEvent]
    scene_transitions: List[SceneTransition]
    detected_objects: List[str]
    detected_actions: List[str]
    video_summary: Dict
    confidence_score: float
    processing_time: float
    model_used: str
    success: bool
    error_message: Optional[str] = None

class IntelligentFrameSampler:
    """Smart frame sampling based on content density and motion"""
    
    def __init__(self, motion_threshold: float = 0.02, scene_change_threshold: float = 0.3):
        self.motion_threshold = motion_threshold
        self.scene_change_threshold = scene_change_threshold
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    def sample_frames_adaptively(self, video_path: str, target_frames: int = 30) -> List[Tuple[int, np.ndarray, float]]:
        """Sample frames based on content density and motion"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # First pass: analyze motion and scene changes
        motion_scores = []
        scene_changes = []
        prev_hist = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # Motion detection
            motion_mask = self.motion_detector.apply(frame)
            motion_score = np.sum(motion_mask > 0) / (frame.shape[0] * frame.shape[1])
            motion_scores.append((frame_idx, motion_score, timestamp))
            
            # Scene change detection using histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            if prev_hist is not None:
                correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                if correlation < self.scene_change_threshold:
                    scene_changes.append((frame_idx, 1.0 - correlation, timestamp))
            
            prev_hist = hist
            frame_idx += 1
        
        cap.release()
        
        # Select optimal frames
        selected_frames = self._select_optimal_frames(motion_scores, scene_changes, target_frames)
        
        # Extract selected frames
        return self._extract_selected_frames(video_path, selected_frames, fps)
    
    def _select_optimal_frames(self, motion_scores: List[Tuple[int, float, float]], 
                              scene_changes: List[Tuple[int, float, float]], 
                              target_frames: int) -> List[Tuple[int, float, float]]:
        """Select frames that maximize information content"""
        
        frame_priorities = defaultdict(float)
        
        # High motion frames (top 40% of target)
        motion_scores.sort(key=lambda x: x[1], reverse=True)
        high_motion_count = int(target_frames * 0.4)
        for i, (frame_idx, score, timestamp) in enumerate(motion_scores[:high_motion_count]):
            frame_priorities[frame_idx] = score + (high_motion_count - i) * 0.1
        
        # Scene change frames (top 30% of target)
        scene_change_count = int(target_frames * 0.3)
        scene_changes.sort(key=lambda x: x[1], reverse=True)
        for frame_idx, score, timestamp in scene_changes[:scene_change_count]:
            frame_priorities[frame_idx] += score + 1.0
        
        # Ensure temporal distribution (remaining 30%)
        if motion_scores:
            total_frames = motion_scores[-1][0]
            remaining_slots = target_frames - len(frame_priorities)
            
            if remaining_slots > 0:
                segment_size = total_frames // remaining_slots
                for i in range(0, total_frames, segment_size):
                    segment_frames = [(idx, score, ts) for idx, score, ts in motion_scores 
                                    if i <= idx < i + segment_size and idx not in frame_priorities]
                    if segment_frames:
                        best_frame = max(segment_frames, key=lambda x: x[1])
                        frame_priorities[best_frame[0]] = best_frame[1] + 0.3
        
        # Return top frames with timestamps
        selected = sorted(frame_priorities.items(), key=lambda x: x[1], reverse=True)[:target_frames]
        
        # Convert back to list with timestamps
        result = []
        for frame_idx, priority in selected:
            timestamp = next(ts for idx, _, ts in motion_scores if idx == frame_idx)
            result.append((frame_idx, priority, timestamp))
        
        return sorted(result, key=lambda x: x[0])  # Sort by frame index
    
    def _extract_selected_frames(self, video_path: str, selected_frames: List[Tuple[int, float, float]], fps: float) -> List[Tuple[int, np.ndarray, float]]:
        """Extract the selected frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_idx, priority, timestamp in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize for memory efficiency
                frame_resized = cv2.resize(frame, (512, 512))
                frames.append((frame_idx, frame_resized, timestamp))
        
        cap.release()
        logger.info(f"✅ Extracted {len(frames)} frames using intelligent sampling")
        return frames

class TemporalEventDetector:
    """Detect and track events across video timeline"""
    
    def __init__(self):
        self.event_templates = {
            'conversation': {
                'required_objects': ['person'],
                'required_actions': ['talking', 'speaking', 'discussing'],
                'min_participants': 2,
                'duration_range': (5, 300),
                'spatial_constraint': 'close_proximity',
                'category': 'social'
            },
            'presentation': {
                'required_objects': ['person', 'screen', 'board'],
                'required_actions': ['pointing', 'presenting', 'explaining'],
                'min_participants': 1,
                'duration_range': (30, 1800),
                'spatial_constraint': 'front_facing',
                'category': 'professional'
            },
            'cooking': {
                'required_objects': ['person', 'food', 'kitchen'],
                'required_actions': ['cooking', 'cutting', 'stirring', 'preparing'],
                'min_participants': 1,
                'duration_range': (60, 3600),
                'category': 'domestic'
            },
            'sports_activity': {
                'required_objects': ['person', 'ball', 'field'],
                'required_actions': ['running', 'kicking', 'throwing', 'playing'],
                'min_participants': 1,
                'duration_range': (10, 7200),
                'category': 'sports'
            },
            'meeting': {
                'required_objects': ['person', 'table', 'chair'],
                'required_actions': ['sitting', 'discussing', 'meeting'],
                'min_participants': 2,
                'duration_range': (300, 7200),
                'category': 'professional'
            }
        }
        
        self.continuity_threshold = 0.6  # Similarity threshold for event continuation
    
    def detect_temporal_events(self, frame_analyses: List[Dict]) -> List[TemporalEvent]:
        """Detect events that span multiple frames"""
        if not frame_analyses:
            return []
        
        events = []
        current_events = {}  # Track ongoing events
        
        for i, frame_analysis in enumerate(frame_analyses):
            frame_idx = frame_analysis.get('frame_index', i)
            timestamp = frame_analysis.get('timestamp', i / 30.0)
            
            # Check for event matches in current frame
            frame_events = self._match_event_templates(frame_analysis)
            
            # Update ongoing events
            events_to_remove = []
            for event_type, event in current_events.items():
                if self._is_event_continuing(event, frame_analysis, frame_idx):
                    # Event continues
                    event.end_frame = frame_idx
                    event.end_timestamp = timestamp
                    event.duration = timestamp - event.start_timestamp
                    event.confidence = min(event.confidence + 0.1, 1.0)
                else:
                    # Event ended
                    events.append(event)
                    events_to_remove.append(event_type)
            
            # Remove ended events
            for event_type in events_to_remove:
                del current_events[event_type]
            
            # Start new events
            for event_type, confidence in frame_events.items():
                if event_type not in current_events:
                    new_event = self._create_temporal_event(
                        event_type, frame_analysis, frame_idx, timestamp, confidence
                    )
                    current_events[event_type] = new_event
        
        # Add any remaining ongoing events
        events.extend(current_events.values())
        
        # Post-process events
        return self._merge_similar_events(events)
    
    def _match_event_templates(self, frame_analysis: Dict) -> Dict[str, float]:
        """Match frame content against event templates"""
        matches = {}
        
        objects = set(obj.lower() for obj in frame_analysis.get('objects', []))
        actions = set(act.lower() for act in frame_analysis.get('actions', []))
        description = frame_analysis.get('description', '').lower()
        
        for event_type, template in self.event_templates.items():
            score = 0.0
            max_score = 0.0
            
            # Check required objects
            required_objects = template.get('required_objects', [])
            object_matches = sum(1 for obj in required_objects if obj in objects or obj in description)
            if required_objects:
                score += (object_matches / len(required_objects)) * 0.4
                max_score += 0.4
            
            # Check required actions
            required_actions = template.get('required_actions', [])
            action_matches = sum(1 for act in required_actions if act in actions or act in description)
            if required_actions:
                score += (action_matches / len(required_actions)) * 0.4
                max_score += 0.4
            
            # Check participant count (approximate from description)
            min_participants = template.get('min_participants', 1)
            participant_indicators = ['person', 'people', 'man', 'woman', 'child']
            participant_count = sum(1 for ind in participant_indicators if ind in description)
            if participant_count >= min_participants:
                score += 0.2
            max_score += 0.2
            
            # Calculate final confidence
            if max_score > 0:
                confidence = score / max_score
                if confidence > 0.3:  # Minimum threshold
                    matches[event_type] = confidence
        
        return matches
    
    def _is_event_continuing(self, event: TemporalEvent, frame_analysis: Dict, frame_idx: int) -> bool:
        """Check if an event continues in the current frame"""
        # Check temporal continuity (no large gaps)
        if frame_idx - event.end_frame > 5:  # Max 5 frame gap
            return False
        
        # Check content similarity
        current_content = self._extract_content_features(frame_analysis)
        event_content = {
            'objects': event.objects_involved,
            'participants': event.participants,
            'category': event.event_category
        }
        
        similarity = self._calculate_content_similarity(current_content, event_content)
        return similarity > self.continuity_threshold
    
    def _create_temporal_event(self, event_type: str, frame_analysis: Dict, 
                              frame_idx: int, timestamp: float, confidence: float) -> TemporalEvent:
        """Create a new temporal event"""
        template = self.event_templates.get(event_type, {})
        
        # Extract participants (simple heuristic)
        description = frame_analysis.get('description', '')
        participants = []
        if 'person' in frame_analysis.get('objects', []):
            participants = ['person_1']  # Basic participant detection
        
        return TemporalEvent(
            event_type=event_type,
            start_frame=frame_idx,
            end_frame=frame_idx,
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            duration=0.0,
            confidence=confidence,
            participants=participants,
            location=self._extract_location(frame_analysis),
            intensity=confidence,
            objects_involved=frame_analysis.get('objects', []),
            emotional_context=self._extract_emotional_context(frame_analysis),
            event_category=template.get('category', 'general')
        )
    
    def _extract_content_features(self, frame_analysis: Dict) -> Dict:
        """Extract key content features for comparison"""
        return {
            'objects': set(frame_analysis.get('objects', [])),
            'actions': set(frame_analysis.get('actions', [])),
            'description_keywords': set(frame_analysis.get('description', '').lower().split())
        }
    
    def _calculate_content_similarity(self, content1: Dict, content2: Dict) -> float:
        """Calculate similarity between content features"""
        # Simple Jaccard similarity for objects and actions
        obj_sim = len(content1.get('objects', set()) & set(content2.get('objects', []))) / \
                 max(len(content1.get('objects', set()) | set(content2.get('objects', []))), 1)
        
        return obj_sim  # Simplified similarity metric
    
    def _extract_location(self, frame_analysis: Dict) -> str:
        """Extract location information from frame analysis"""
        description = frame_analysis.get('description', '').lower()
        
        indoor_keywords = ['room', 'office', 'kitchen', 'indoor', 'inside', 'building']
        outdoor_keywords = ['outdoor', 'outside', 'park', 'street', 'field', 'garden']
        
        if any(keyword in description for keyword in indoor_keywords):
            return 'indoor'
        elif any(keyword in description for keyword in outdoor_keywords):
            return 'outdoor'
        
        return 'unknown'
    
    def _extract_emotional_context(self, frame_analysis: Dict) -> str:
        """Extract emotional context from frame analysis"""
        description = frame_analysis.get('description', '').lower()
        
        positive_keywords = ['happy', 'smiling', 'laughing', 'cheerful', 'excited']
        negative_keywords = ['sad', 'angry', 'frustrated', 'worried', 'upset']
        neutral_keywords = ['calm', 'focused', 'serious', 'concentrated']
        
        if any(keyword in description for keyword in positive_keywords):
            return 'positive'
        elif any(keyword in description for keyword in negative_keywords):
            return 'negative'
        elif any(keyword in description for keyword in neutral_keywords):
            return 'neutral'
        
        return 'neutral'
    
    def _merge_similar_events(self, events: List[TemporalEvent]) -> List[TemporalEvent]:
        """Merge events that are very similar and temporally close"""
        if len(events) <= 1:
            return events
        
        merged_events = []
        events_sorted = sorted(events, key=lambda e: e.start_timestamp)
        
        current_event = events_sorted[0]
        
        for next_event in events_sorted[1:]:
            # Check if events should be merged
            if (current_event.event_type == next_event.event_type and
                next_event.start_timestamp - current_event.end_timestamp < 10.0 and  # 10 second gap
                current_event.event_category == next_event.event_category):
                
                # Merge events
                current_event.end_frame = next_event.end_frame
                current_event.end_timestamp = next_event.end_timestamp
                current_event.duration = current_event.end_timestamp - current_event.start_timestamp
                current_event.confidence = max(current_event.confidence, next_event.confidence)
                
                # Merge participants and objects
                current_event.participants = list(set(current_event.participants + next_event.participants))
                current_event.objects_involved = list(set(current_event.objects_involved + next_event.objects_involved))
            else:
                # Events are different, add current and start new
                merged_events.append(current_event)
                current_event = next_event
        
        # Add the last event
        merged_events.append(current_event)
        
        return merged_events

class SceneTransitionDetector:
    """Detect major transitions and scene changes"""
    
    def __init__(self, similarity_threshold: float = 0.4):
        self.similarity_threshold = similarity_threshold
    
    def detect_transitions(self, frame_analyses: List[Dict]) -> List[SceneTransition]:
        """Detect scene transitions between frames"""
        if len(frame_analyses) < 2:
            return []
        
        transitions = []
        
        for i in range(1, len(frame_analyses)):
            prev_frame = frame_analyses[i-1]
            curr_frame = frame_analyses[i]
            
            # Calculate similarity metrics
            similarity_score = self._calculate_frame_similarity(prev_frame, curr_frame)
            
            if similarity_score < self.similarity_threshold:
                transition = SceneTransition(
                    frame=curr_frame.get('frame_index', i),
                    timestamp=curr_frame.get('timestamp', i / 30.0),
                    transition_type=self._classify_transition(prev_frame, curr_frame),
                    confidence=1.0 - similarity_score,
                    description=self._describe_transition(prev_frame, curr_frame)
                )
                transitions.append(transition)
        
        return transitions
    
    def _calculate_frame_similarity(self, frame1: Dict, frame2: Dict) -> float:
        """Calculate similarity between two frames"""
        # Object similarity
        objects1 = set(frame1.get('objects', []))
        objects2 = set(frame2.get('objects', []))
        object_similarity = len(objects1 & objects2) / max(len(objects1 | objects2), 1)
        
        # Action similarity
        actions1 = set(frame1.get('actions', []))
        actions2 = set(frame2.get('actions', []))
        action_similarity = len(actions1 & actions2) / max(len(actions1 | actions2), 1)
        
        # Description similarity (simplified)
        desc1_words = set(frame1.get('description', '').lower().split())
        desc2_words = set(frame2.get('description', '').lower().split())
        desc_similarity = len(desc1_words & desc2_words) / max(len(desc1_words | desc2_words), 1)
        
        return (object_similarity + action_similarity + desc_similarity) / 3
    
    def _classify_transition(self, prev_frame: Dict, curr_frame: Dict) -> str:
        """Classify the type of transition"""
        prev_desc = prev_frame.get('description', '').lower()
        curr_desc = curr_frame.get('description', '').lower()
        
        # Location change
        if ('indoor' in prev_desc and 'outdoor' in curr_desc) or \
           ('outdoor' in prev_desc and 'indoor' in curr_desc):
            return 'location_change'
        
        # Activity change
        prev_actions = set(prev_frame.get('actions', []))
        curr_actions = set(curr_frame.get('actions', []))
        if len(prev_actions & curr_actions) == 0:
            return 'activity_change'
        
        # Scene change (different objects)
        prev_objects = set(prev_frame.get('objects', []))
        curr_objects = set(curr_frame.get('objects', []))
        if len(prev_objects & curr_objects) < len(prev_objects) * 0.3:
            return 'scene_change'
        
        return 'minor_change'
    
    def _describe_transition(self, prev_frame: Dict, curr_frame: Dict) -> str:
        """Generate description of the transition"""
        transition_type = self._classify_transition(prev_frame, curr_frame)
        
        if transition_type == 'location_change':
            return f"Scene moves from {self._get_scene_type(prev_frame)} to {self._get_scene_type(curr_frame)}"
        elif transition_type == 'activity_change':
            prev_activity = ', '.join(prev_frame.get('actions', [])[:2])
            curr_activity = ', '.join(curr_frame.get('actions', [])[:2])
            return f"Activity changes from {prev_activity} to {curr_activity}"
        elif transition_type == 'scene_change':
            return "Major scene change with different objects and setting"
        else:
            return "Minor visual change in scene"
    
    def _get_scene_type(self, frame: Dict) -> str:
        """Determine scene type from frame analysis"""
        description = frame.get('description', '').lower()
        
        if any(word in description for word in ['indoor', 'room', 'office', 'kitchen']):
            return 'indoor scene'
        elif any(word in description for word in ['outdoor', 'outside', 'street', 'park']):
            return 'outdoor scene'
        else:
            return 'scene'

class HierarchicalSummarizer:
    """Generate multi-level video summaries"""
    
    def __init__(self):
        self.summary_templates = {
            'narrative': "This {duration} video shows {main_activity} {location}. {key_events} The video {conclusion}",
            'documentary': "Documentary-style video covering {subject} over {duration}. Key segments include {highlights}",
            'instructional': "Instructional video demonstrating {process}. Main steps: {steps}",
            'social': "Social video featuring {participants} engaged in {activities}. Notable moments: {highlights}"
        }
    
    def generate_comprehensive_summary(self, 
                                     frame_analyses: List[Dict],
                                     temporal_events: List[TemporalEvent],
                                     scene_transitions: List[SceneTransition]) -> Dict:
        """Generate multi-level video summary"""
        
        if not frame_analyses:
            return {'error': 'No frame analyses provided'}
        
        video_duration = self._calculate_duration(frame_analyses)
        video_type = self._classify_video_type(temporal_events, frame_analyses)
        
        return {
            'brief_summary': self._generate_brief_summary(temporal_events, video_duration),
            'detailed_summary': self._generate_detailed_summary(temporal_events, scene_transitions, video_type),
            'structured_analysis': {
                'video_type': video_type,
                'duration': video_duration,
                'main_events': self._extract_key_events(temporal_events)[:5],
                'participants': self._identify_participants(frame_analyses),
                'locations': self._identify_locations(frame_analyses),
                'timeline': self._create_timeline(temporal_events, scene_transitions),
                'highlights': self._identify_highlights(temporal_events)[:3]
            },
            'narrative_structure': self._generate_narrative_structure(temporal_events, scene_transitions)
        }
    
    def _calculate_duration(self, frame_analyses: List[Dict]) -> str:
        """Calculate and format video duration"""
        if not frame_analyses:
            return "unknown duration"
        
        last_timestamp = frame_analyses[-1].get('timestamp', 0)
        
        if last_timestamp < 60:
            return f"{int(last_timestamp)} second"
        elif last_timestamp < 3600:
            minutes = int(last_timestamp // 60)
            return f"{minutes} minute"
        else:
            hours = int(last_timestamp // 3600)
            minutes = int((last_timestamp % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _classify_video_type(self, events: List[TemporalEvent], frame_analyses: List[Dict]) -> str:
        """Classify the overall video type"""
        if not events:
            return 'general'
        
        # Count event categories
        categories = Counter(event.event_category for event in events)
        most_common_category = categories.most_common(1)[0][0] if categories else 'general'
        
        # Check for instructional content
        descriptions = ' '.join(frame.get('description', '') for frame in frame_analyses).lower()
        if any(word in descriptions for word in ['tutorial', 'how to', 'step', 'demonstrate']):
            return 'instructional'
        
        # Check for social content
        if most_common_category == 'social' or len(self._identify_participants(frame_analyses)) > 2:
            return 'social'
        
        # Check for professional content
        if most_common_category == 'professional':
            return 'professional'
        
        return most_common_category
    
    def _generate_brief_summary(self, events: List[TemporalEvent], duration: str) -> str:
        """Generate one-sentence summary"""
        if not events:
            return f"A {duration} video with various activities."
        
        # Get most significant event
        main_event = max(events, key=lambda e: e.duration * e.confidence)
        
        participants_text = ""
        if main_event.participants:
            if len(main_event.participants) == 1:
                participants_text = f"featuring {main_event.participants[0]} "
            else:
                participants_text = f"featuring {len(main_event.participants)} people "
        
        location_text = f" {main_event.location}" if main_event.location != 'unknown' else ""
        
        return f"A {duration} video {participants_text}showing {main_event.event_type}{location_text}."
    
    def _generate_detailed_summary(self, events: List[TemporalEvent], 
                                  transitions: List[SceneTransition], video_type: str) -> str:
        """Generate detailed paragraph summary"""
        if not events:
            return "The video contains various activities but specific events could not be identified."
        
        # Opening
        duration = self._format_duration_from_events(events)
        main_events = sorted(events, key=lambda e: e.duration * e.confidence, reverse=True)[:3]
        
        summary_parts = []
        
        # Introduction
        if main_events:
            primary_event = main_events[0]
            location_desc = f" {primary_event.location}" if primary_event.location != 'unknown' else ""
            summary_parts.append(f"This {duration} video primarily shows {primary_event.event_type}{location_desc}")
        
        # Main content
        if len(main_events) > 1:
            other_activities = [event.event_type for event in main_events[1:]]
            summary_parts.append(f"Additional activities include {', '.join(other_activities)}")
        
        # Transitions and flow
        if len(transitions) > 0:
            significant_transitions = [t for t in transitions if t.confidence > 0.7]
            if significant_transitions:
                summary_parts.append(f"The video contains {len(significant_transitions)} major scene transitions")
        
        # Participants and emotional context
        all_participants = set()
        emotional_contexts = []
        for event in main_events:
            all_participants.update(event.participants)
            if event.emotional_context != 'neutral':
                emotional_contexts.append(event.emotional_context)
        
        if all_participants:
            summary_parts.append(f"involving {len(all_participants)} participant(s)")
        
        if emotional_contexts:
            dominant_emotion = Counter(emotional_contexts).most_common(1)[0][0]
            summary_parts.append(f"with an overall {dominant_emotion} tone")
        
        return '. '.join(summary_parts) + '.'
    
    def _extract_key_events(self, events: List[TemporalEvent]) -> List[Dict]:
        """Extract the most important events"""
        if not events:
            return []
        
        # Score events by duration, confidence, and uniqueness
        scored_events = []
        
        for event in events:
            # Calculate importance score
            duration_score = min(event.duration / 60.0, 1.0)  # Normalize to 1 minute
            confidence_score = event.confidence
            uniqueness_score = 1.0 / (1 + len([e for e in events if e.event_type == event.event_type]))
            
            importance_score = (duration_score * 0.4 + confidence_score * 0.4 + uniqueness_score * 0.2)
            
            scored_events.append({
                'event_type': event.event_type,
                'category': event.event_category,
                'start_time': self._format_timestamp(event.start_timestamp),
                'duration': self._format_duration(event.duration),
                'participants': event.participants,
                'location': event.location,
                'confidence': round(event.confidence, 2),
                'importance_score': round(importance_score, 2),
                'description': self._generate_event_description(event)
            })
        
        # Sort by importance and return top events
        return sorted(scored_events, key=lambda x: x['importance_score'], reverse=True)
    
    def _generate_event_description(self, event: TemporalEvent) -> str:
        """Generate natural language description of an event"""
        desc_parts = []
        
        # Add participants
        if event.participants:
            if len(event.participants) == 1:
                desc_parts.append(f"{event.participants[0]}")
            else:
                desc_parts.append(f"{len(event.participants)} people")
        
        # Add main activity
        activity_desc = event.event_type.replace('_', ' ')
        desc_parts.append(f"engaged in {activity_desc}")
        
        # Add location if known
        if event.location and event.location != 'unknown':
            desc_parts.append(f"in an {event.location} setting")
        
        # Add duration
        if event.duration > 60:
            desc_parts.append(f"for {int(event.duration // 60)} minutes")
        elif event.duration > 0:
            desc_parts.append(f"for {int(event.duration)} seconds")
        
        return ' '.join(desc_parts)
    
    def _identify_participants(self, frame_analyses: List[Dict]) -> List[str]:
        """Identify unique participants across video"""
        participants = set()
        person_count = 0
        
        for frame in frame_analyses:
            description = frame.get('description', '').lower()
            objects = frame.get('objects', [])
            
            # Count people in frame
            if 'person' in objects or 'people' in objects:
                # Simple heuristic: estimate people count from description
                people_indicators = ['man', 'woman', 'person', 'child', 'people']
                frame_people = sum(1 for indicator in people_indicators if indicator in description)
                person_count = max(person_count, frame_people or 1)
        
        # Generate participant list
        for i in range(min(person_count, 10)):  # Cap at 10 participants
            participants.add(f"person_{i+1}")
        
        return list(participants)
    
    def _identify_locations(self, frame_analyses: List[Dict]) -> List[str]:
        """Identify different locations in the video"""
        locations = set()
        
        location_keywords = {
            'office': ['office', 'desk', 'computer', 'meeting', 'workplace'],
            'kitchen': ['kitchen', 'cooking', 'food', 'stove', 'counter'],
            'outdoor': ['outdoor', 'outside', 'street', 'park', 'garden', 'field'],
            'living_room': ['living room', 'sofa', 'couch', 'tv', 'television'],
            'classroom': ['classroom', 'blackboard', 'students', 'teacher', 'lecture'],
            'restaurant': ['restaurant', 'dining', 'table', 'waiter', 'menu'],
            'gym': ['gym', 'exercise', 'workout', 'fitness', 'equipment']
        }
        
        for frame in frame_analyses:
            description = frame.get('description', '').lower()
            objects = [obj.lower() for obj in frame.get('objects', [])]
            
            for location_type, keywords in location_keywords.items():
                if any(keyword in description or keyword in objects for keyword in keywords):
                    locations.add(location_type)
        
        return list(locations) if locations else ['indoor']
    
    def _create_timeline(self, events: List[TemporalEvent], transitions: List[SceneTransition]) -> List[Dict]:
        """Create chronological timeline of video"""
        timeline_items = []
        
        # Add events to timeline
        for event in events:
            timeline_items.append({
                'timestamp': event.start_timestamp,
                'type': 'event_start',
                'description': f"{event.event_type} begins",
                'details': event.event_category,
                'duration': event.duration
            })
            
            if event.duration > 30:  # Only add end for longer events
                timeline_items.append({
                    'timestamp': event.end_timestamp,
                    'type': 'event_end',
                    'description': f"{event.event_type} ends",
                    'details': f"Lasted {self._format_duration(event.duration)}",
                    'duration': 0
                })
        
        # Add scene transitions
        for transition in transitions:
            if transition.confidence > 0.6:  # Only significant transitions
                timeline_items.append({
                    'timestamp': transition.timestamp,
                    'type': 'scene_transition',
                    'description': transition.description,
                    'details': transition.transition_type,
                    'duration': 0
                })
        
        # Sort by timestamp
        timeline_items.sort(key=lambda x: x['timestamp'])
        
        # Format timestamps
        for item in timeline_items:
            item['formatted_time'] = self._format_timestamp(item['timestamp'])
        
        return timeline_items[:20]  # Limit to 20 most important items
    
    def _identify_highlights(self, events: List[TemporalEvent]) -> List[Dict]:
        """Identify video highlights based on event characteristics"""
        highlights = []
        
        for event in events:
            highlight_score = 0.0
            
            # High confidence events
            if event.confidence > 0.8:
                highlight_score += 0.3
            
            # Long duration events
            if event.duration > 60:
                highlight_score += 0.2
            
            # Multiple participants
            if len(event.participants) > 1:
                highlight_score += 0.2
            
            # Emotional context
            if event.emotional_context in ['positive', 'negative']:
                highlight_score += 0.2
            
            # Unique event types
            if event.event_category in ['social', 'professional']:
                highlight_score += 0.1
            
            if highlight_score > 0.4:  # Threshold for highlights
                highlights.append({
                    'event_type': event.event_type,
                    'timestamp': self._format_timestamp(event.start_timestamp),
                    'description': self._generate_event_description(event),
                    'highlight_score': round(highlight_score, 2),
                    'why_highlighted': self._explain_highlight(event, highlight_score)
                })
        
        return sorted(highlights, key=lambda x: x['highlight_score'], reverse=True)
    
    def _explain_highlight(self, event: TemporalEvent, score: float) -> str:
        """Explain why an event was highlighted"""
        reasons = []
        
        if event.confidence > 0.8:
            reasons.append("high confidence detection")
        if event.duration > 60:
            reasons.append("extended duration")
        if len(event.participants) > 1:
            reasons.append("multiple participants")
        if event.emotional_context != 'neutral':
            reasons.append(f"{event.emotional_context} emotional context")
        
        return ', '.join(reasons) if reasons else "significant activity"
    
    def _generate_narrative_structure(self, events: List[TemporalEvent], 
                                    transitions: List[SceneTransition]) -> Dict:
        """Generate narrative structure analysis"""
        if not events:
            return {'error': 'No events to analyze'}
        
        # Analyze story arc
        total_duration = max(event.end_timestamp for event in events) if events else 0
        
        # Divide into acts (beginning, middle, end)
        act1_end = total_duration * 0.25
        act2_end = total_duration * 0.75
        
        act1_events = [e for e in events if e.start_timestamp <= act1_end]
        act2_events = [e for e in events if act1_end < e.start_timestamp <= act2_end]
        act3_events = [e for e in events if e.start_timestamp > act2_end]
        
        # Analyze pacing
        transition_density = len(transitions) / (total_duration / 60) if total_duration > 0 else 0
        
        return {
            'total_duration': self._format_duration(total_duration),
            'story_structure': {
                'act1_setup': {
                    'duration': self._format_duration(act1_end),
                    'events': len(act1_events),
                    'main_activities': list(set(e.event_type for e in act1_events))[:3]
                },
                'act2_development': {
                    'duration': self._format_duration(act2_end - act1_end),
                    'events': len(act2_events),
                    'main_activities': list(set(e.event_type for e in act2_events))[:3]
                },
                'act3_resolution': {
                    'duration': self._format_duration(total_duration - act2_end),
                    'events': len(act3_events),
                    'main_activities': list(set(e.event_type for e in act3_events))[:3]
                }
            },
            'pacing_analysis': {
                'transition_density': round(transition_density, 2),
                'pacing_type': self._classify_pacing(transition_density),
                'intensity_curve': self._analyze_intensity_curve(events)
            },
            'thematic_elements': {
                'dominant_themes': self._extract_themes(events),
                'emotional_journey': self._trace_emotional_journey(events),
                'character_development': self._analyze_character_presence(events)
            }
        }
    
    def _classify_pacing(self, transition_density: float) -> str:
        """Classify video pacing based on transition density"""
        if transition_density > 2.0:
            return "fast-paced"
        elif transition_density > 1.0:
            return "moderate-paced"
        elif transition_density > 0.5:
            return "slow-paced"
        else:
            return "very slow-paced"
    
    def _analyze_intensity_curve(self, events: List[TemporalEvent]) -> List[Dict]:
        """Analyze how intensity changes over time"""
        if not events:
            return []
        
        total_duration = max(event.end_timestamp for event in events)
        time_segments = 10  # Divide video into 10 segments
        segment_duration = total_duration / time_segments
        
        intensity_curve = []
        
        for i in range(time_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            # Calculate intensity for this segment
            segment_events = [e for e in events 
                            if e.start_timestamp < end_time and e.end_timestamp > start_time]
            
            if segment_events:
                avg_intensity = sum(e.intensity for e in segment_events) / len(segment_events)
                event_density = len(segment_events)
            else:
                avg_intensity = 0.0
                event_density = 0
            
            intensity_curve.append({
                'segment': i + 1,
                'time_range': f"{self._format_timestamp(start_time)}-{self._format_timestamp(end_time)}",
                'intensity': round(avg_intensity, 2),
                'event_density': event_density,
                'description': self._describe_segment_intensity(avg_intensity, event_density)
            })
        
        return intensity_curve
    
    def _describe_segment_intensity(self, intensity: float, density: int) -> str:
        """Describe the intensity of a video segment"""
        if intensity > 0.8 and density > 2:
            return "high activity, intense"
        elif intensity > 0.6:
            return "moderate activity"
        elif intensity > 0.3:
            return "low activity"
        else:
            return "minimal activity"
    
    def _extract_themes(self, events: List[TemporalEvent]) -> List[str]:
        """Extract dominant themes from events"""
        theme_keywords = {
            'education': ['presentation', 'teaching', 'learning', 'classroom'],
            'social': ['conversation', 'meeting', 'gathering', 'social'],
            'work': ['professional', 'office', 'business', 'meeting'],
            'leisure': ['playing', 'sports', 'entertainment', 'fun'],
            'domestic': ['cooking', 'home', 'family', 'domestic'],
            'creative': ['art', 'music', 'creative', 'performance']
        }
        
        theme_scores = defaultdict(int)
        
        for event in events:
            event_text = f"{event.event_type} {event.event_category}".lower()
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in event_text for keyword in keywords):
                    theme_scores[theme] += event.duration * event.confidence
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes[:3] if score > 0]
    
    def _trace_emotional_journey(self, events: List[TemporalEvent]) -> List[Dict]:
        """Trace emotional progression through video"""
        emotional_timeline = []
        
        for event in events:
            if event.emotional_context != 'neutral':
                emotional_timeline.append({
                    'timestamp': self._format_timestamp(event.start_timestamp),
                    'emotion': event.emotional_context,
                    'context': event.event_type,
                    'intensity': event.intensity
                })
        
        return sorted(emotional_timeline, key=lambda x: x['timestamp'])[:10]
    
    def _analyze_character_presence(self, events: List[TemporalEvent]) -> Dict:
        """Analyze character/participant presence throughout video"""
        participant_analysis = defaultdict(lambda: {'total_time': 0, 'events': 0, 'activities': set()})
        
        for event in events:
            for participant in event.participants:
                participant_analysis[participant]['total_time'] += event.duration
                participant_analysis[participant]['events'] += 1
                participant_analysis[participant]['activities'].add(event.event_type)
        
        # Convert to readable format
        character_summary = {}
        for participant, data in participant_analysis.items():
            character_summary[participant] = {
                'screen_time': self._format_duration(data['total_time']),
                'events_participated': data['events'],
                'main_activities': list(data['activities'])[:3]
            }
        
        return character_summary
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp as MM:SS"""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _format_duration(self, duration: float) -> str:
        """Format duration in human readable form"""
        if duration < 60:
            return f"{int(duration)}s"
        elif duration < 3600:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}m {seconds}s" if seconds > 0 else f"{minutes}m"
        else:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _format_duration_from_events(self, events: List[TemporalEvent]) -> str:
        """Calculate total duration from events"""
        if not events:
            return "short"
        
        max_timestamp = max(event.end_timestamp for event in events)
        return self._format_duration(max_timestamp)


class ParallelAnalysisEngine:
    """Parallel processing engine for video analysis"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1))
        self.frame_cache = {}
    
    async def analyze_video_parallel(self, video_path: str, vlm_analyzer) -> EnhancedAnalysisResult:
        """Main parallel analysis pipeline"""
        start_time = datetime.now()
        
        try:
            # Stage 1: Intelligent frame sampling
            logger.info("🔍 Stage 1: Intelligent frame sampling")
            sampler = IntelligentFrameSampler()
            frame_data = sampler.sample_frames_adaptively(video_path, target_frames=40)
            
            if not frame_data:
                raise ValueError("No frames could be extracted from video")
            
            # Stage 2: Parallel VLM analysis
            logger.info(f"🧠 Stage 2: Parallel VLM analysis of {len(frame_data)} frames")
            frame_analyses = await self._analyze_frames_parallel(frame_data, vlm_analyzer)
            
            # Stage 3: Temporal event detection
            logger.info("⏰ Stage 3: Temporal event detection")
            event_detector = TemporalEventDetector()
            temporal_events = event_detector.detect_temporal_events(frame_analyses)
            
            # Stage 4: Scene transition detection
            logger.info("🎬 Stage 4: Scene transition detection")
            transition_detector = SceneTransitionDetector()
            scene_transitions = transition_detector.detect_transitions(frame_analyses)
            
            # Stage 5: Hierarchical summarization
            logger.info("📝 Stage 5: Hierarchical summarization")
            summarizer = HierarchicalSummarizer()
            video_summary = summarizer.generate_comprehensive_summary(
                frame_analyses, temporal_events, scene_transitions
            )
            
            # Compile results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract comprehensive data
            all_objects = []
            all_actions = []
            for frame in frame_analyses:
                all_objects.extend(frame.get('objects', []))
                all_actions.extend(frame.get('actions', []))
            
            unique_objects = list(set(all_objects))
            unique_actions = list(set(all_actions))
            
            # Calculate overall confidence
            confidences = [frame.get('confidence', 0.5) for frame in frame_analyses]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return EnhancedAnalysisResult(
                frame_descriptions=frame_analyses,
                temporal_events=temporal_events,
                scene_transitions=scene_transitions,
                detected_objects=unique_objects,
                detected_actions=unique_actions,
                video_summary=video_summary,
                confidence_score=avg_confidence,
                processing_time=processing_time,
                model_used=vlm_analyzer.model_used if hasattr(vlm_analyzer, 'model_used') else vlm_analyzer.model_type,
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Parallel analysis failed: {e}")
            
            return EnhancedAnalysisResult(
                frame_descriptions=[],
                temporal_events=[],
                scene_transitions=[],
                detected_objects=[],
                detected_actions=[],
                video_summary={'error': str(e)},
                confidence_score=0.0,
                processing_time=processing_time,
                model_used="unknown",
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_frames_parallel(self, frame_data: List[Tuple[int, np.ndarray, float]], 
                                     vlm_analyzer) -> List[Dict]:
        """Analyze frames in parallel using asyncio"""
        
        async def analyze_single_frame(frame_info):
            frame_idx, frame, timestamp = frame_info
            
            # Check cache first
            frame_hash = self._get_frame_hash(frame)
            if frame_hash in self.frame_cache:
                cached_result = self.frame_cache[frame_hash].copy()
                cached_result.update({
                    'frame_index': frame_idx,
                    'timestamp': timestamp
                })
                return cached_result
            
            # Analyze frame
            try:
                result = vlm_analyzer.analyze_frame_with_vlm(frame)
                result.update({
                    'frame_index': frame_idx,
                    'timestamp': timestamp
                })
                
                # Cache result
                self.frame_cache[frame_hash] = result.copy()
                
                return result
                
            except Exception as e:
                logger.error(f"Frame {frame_idx} analysis failed: {e}")
                return {
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'description': f"Analysis failed: {str(e)}",
                    'objects': [],
                    'actions': [],
                    'confidence': 0.0
                }
        
        # Create tasks for parallel execution
        tasks = [analyze_single_frame(frame_info) for frame_info in frame_data]
        
        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_analyze(task):
            async with semaphore:
                return await task
        
        # Run analyses
        results = await asyncio.gather(*[bounded_analyze(task) for task in tasks])
        
        # Sort by frame index
        return sorted(results, key=lambda x: x['frame_index'])
    
    def _get_frame_hash(self, frame: np.ndarray) -> str:
        """Generate hash for frame caching"""
        # Simple hash based on frame content
        frame_small = cv2.resize(frame, (32, 32))
        frame_bytes = frame_small.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()


class CachingSystem:
    """Persistent caching for analysis results"""
    
    def __init__(self, cache_dir: str = "video_analysis_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "analysis_cache.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_cache (
                video_hash TEXT PRIMARY KEY,
                video_path TEXT,
                analysis_result BLOB,
                created_at TIMESTAMP,
                model_used TEXT,
                version TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frame_cache (
                frame_hash TEXT PRIMARY KEY,
                analysis_result BLOB,
                created_at TIMESTAMP,
                model_used TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_video_analysis(self, video_path: str, model_name: str) -> Optional[EnhancedAnalysisResult]:
        """Retrieve cached video analysis"""
        video_hash = self._get_video_hash(video_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT analysis_result FROM video_cache 
            WHERE video_hash = ? AND model_used = ?
        ''', (video_hash, model_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return pickle.loads(result[0])
            except:
                return None
        
        return None
    
    def cache_video_analysis(self, video_path: str, result: EnhancedAnalysisResult):
        """Cache video analysis result"""
        video_hash = self._get_video_hash(video_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO video_cache 
            (video_hash, video_path, analysis_result, created_at, model_used, version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            video_hash,
            video_path,
            pickle.dumps(result),
            datetime.now(),
            result.model_used,
            "1.0"
        ))
        
        conn.commit()
        conn.close()
    
    def _get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file"""
        # Use file size and modification time for quick hashing
        try:
            stat = os.stat(video_path)
            hash_input = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return hashlib.md5(video_path.encode()).hexdigest()


class AdvancedVLMVideoAnalyzer:
    """Main enhanced VLM video analyzer with all features"""
    
    def __init__(self, 
                 model_type: str = "qwen",
                 model_name: Optional[str] = None,
                 max_workers: int = None,
                 enable_caching: bool = True,
                 cache_dir: str = "video_analysis_cache"):
        
        # Initialize core VLM analyzer (using your existing enhanced analyzer)
        from enhanced_vlm_analyzer import EnhancedVLMVideoAnalyzer
        self.vlm_analyzer = EnhancedVLMVideoAnalyzer(
            model_type=model_type,
            model_name=model_name
        )
        
        # Initialize parallel processing
        self.parallel_engine = ParallelAnalysisEngine(max_workers=max_workers)
        
        # Initialize caching
        self.caching_enabled = enable_caching
        if enable_caching:
            self.cache_system = CachingSystem(cache_dir=cache_dir)
        
        logger.info(f"✅ Advanced VLM Analyzer initialized")
        logger.info(f"   Model: {model_type}")
        logger.info(f"   Parallel workers: {self.parallel_engine.max_workers}")
        logger.info(f"   Caching: {'enabled' if enable_caching else 'disabled'}")
    
    async def analyze_video_comprehensive(self, video_path: str, 
                                        force_reanalysis: bool = False) -> EnhancedAnalysisResult:
        """Comprehensive video analysis with all enhancements"""
        
        # Check cache first
        if self.caching_enabled and not force_reanalysis:
            cached_result = self.cache_system.get_video_analysis(
                video_path, self.vlm_analyzer.model_type
            )
            if cached_result:
                logger.info("✅ Using cached analysis result")
                return cached_result
        
        # Perform fresh analysis
        logger.info(f"🚀 Starting comprehensive analysis of: {os.path.basename(video_path)}")
        
        result = await self.parallel_engine.analyze_video_parallel(
            video_path, self.vlm_analyzer
        )
        
        # Cache result
        if self.caching_enabled and result.success:
            self.cache_system.cache_video_analysis(video_path, result)
            logger.info("💾 Analysis result cached")
        
        return result
    
    def analyze_video_sync(self, video_path: str, force_reanalysis: bool = False) -> EnhancedAnalysisResult:
        """Synchronous wrapper for the async analysis"""
        return asyncio.run(self.analyze_video_comprehensive(video_path, force_reanalysis))
    
    def generate_analysis_report(self, result: EnhancedAnalysisResult, 
                               output_format: str = "markdown") -> str:
        """Generate formatted analysis report"""
        
        if not result.success:
            return f"# Analysis Failed\n\nError: {result.error_message}"
        
        if output_format == "markdown":
            return self._generate_markdown_report(result)
        elif output_format == "json":
            return self._generate_json_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_markdown_report(self, result: EnhancedAnalysisResult) -> str:
        """Generate detailed markdown report"""
        
        report = f"""# 🎥 Video Analysis Report

## 📊 Overview
- **Processing Time**: {result.processing_time:.1f} seconds
- **Model Used**: {result.model_used}
- **Confidence Score**: {result.confidence_score:.2f}
- **Frames Analyzed**: {len(result.frame_descriptions)}

## 📝 Summary

### Brief Summary
{result.video_summary.get('brief_summary', 'No summary available')}

### Detailed Summary  
{result.video_summary.get('detailed_summary', 'No detailed summary available')}

## 🎯 Key Events ({len(result.temporal_events)})
"""
        
        # Add top events
        if result.temporal_events:
            for i, event in enumerate(result.temporal_events[:5], 1):
                report += f"""
### {i}. {event.event_type.title().replace('_', ' ')}
- **Duration**: {self._format_duration(event.duration)}
- **Confidence**: {event.confidence:.2f}
- **Category**: {event.event_category}
- **Participants**: {', '.join(event.participants) if event.participants else 'Unknown'}
- **Location**: {event.location}
"""
        
        # Add timeline
        timeline = result.video_summary.get('structured_analysis', {}).get('timeline', [])
        if timeline:
            report += f"\n## ⏰ Timeline\n"
            for item in timeline[:10]:
                report += f"- **{item['formatted_time']}**: {item['description']}\n"
        
        # Add highlights
        highlights = result.video_summary.get('structured_analysis', {}).get('highlights', [])
        if highlights:
            report += f"\n## ⭐ Highlights\n"
            for highlight in highlights:
                report += f"- **{highlight['timestamp']}**: {highlight['description']} ({highlight['why_highlighted']})\n"
        
        # Add detected objects and actions
        report += f"""
## 🏷️ Detected Elements

### Objects ({len(result.detected_objects)})
{', '.join(result.detected_objects[:20])}

### Actions ({len(result.detected_actions)})
{', '.join(result.detected_actions[:15])}

## 🎬 Scene Analysis

### Scene Transitions ({len(result.scene_transitions)})
"""
        
        for transition in result.scene_transitions[:5]:
            report += f"- **{self._format_timestamp(transition.timestamp)}**: {transition.description}\n"
        
        # Add narrative structure
        narrative = result.video_summary.get('narrative_structure', {})
        if narrative and 'story_structure' in narrative:
            story = narrative['story_structure']
            report += f"""
## 📖 Narrative Structure

### Story Arc
- **Setup ({story['act1_setup']['duration']})**: {len(story['act1_setup']['events'])} events - {', '.join(story['act1_setup']['main_activities'])}
- **Development ({story['act2_development']['duration']})**: {len(story['act2_development']['events'])} events - {', '.join(story['act2_development']['main_activities'])}  
- **Resolution ({story['act3_resolution']['duration']})**: {len(story['act3_resolution']['events'])} events - {', '.join(story['act3_resolution']['main_activities'])}

### Pacing
- **Type**: {narrative['pacing_analysis']['pacing_type']}
- **Transition Density**: {narrative['pacing_analysis']['transition_density']} per minute
"""
        
        return report
    
    def _format_duration(self, duration: float) -> str:
        """Format duration in human readable form"""
        if duration < 60:
            return f"{int(duration)}s"
        elif duration < 3600:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}m {seconds}s" if seconds > 0 else f"{minutes}m"
        else:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp as MM:SS"""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"


# Usage example and integration
async def main_demo():
    """Demonstration of the enhanced VLM analyzer"""
    
    # Initialize analyzer
    analyzer = AdvancedVLMVideoAnalyzer(
        model_type="qwen",  # Use Qwen as default
        max_workers=4,
        enable_caching=True
    )
    
    # Analyze video
    video_path = input("Pass the input video path : " )  # Replace with actual video
    
    print("🚀 Starting comprehensive video analysis...")
    result = await analyzer.analyze_video_comprehensive(video_path)
    
    if result.success:
        print("✅ Analysis completed successfully!")
        print(f"📊 Processing time: {result.processing_time:.1f}s")
        print(f"🎯 Found {len(result.temporal_events)} events")
        print(f"🎬 Found {len(result.scene_transitions)} scene transitions")
        
        # Generate report
        report = analyzer.generate_analysis_report(result, "markdown")
        
        # Save report
        with open("video_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("📝 Report saved to video_analysis_report.md")
        
    else:
        print(f"❌ Analysis failed: {result.error_message}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main_demo())