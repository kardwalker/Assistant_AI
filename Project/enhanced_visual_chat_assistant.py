"""
Enhanced Visual Chat Assistant Agent using VLM Core Components
Integrates with the Vuenagent enhanced VLM analyzer for comprehensive video analysis
"""

import os
import sys
import argparse
from typing import Any, Dict, List, TypedDict, Optional
from datetime import datetime
import logging

# LangChain and LangGraph imports
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add the Core directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Core'))

# Import VLM enhanced analyzer
from enhanced_vlm_analyzer import EnhancedVLMVideoAnalyzer, VLMAnalysisResult

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Initialize Azure OpenAI model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2023-05-15", 
    api_key=os.getenv("AZURE_API_KEY"),
    azure_deployment="gpt-4o-mini-hackthon",
    temperature=0.1,
)

def is_video_file(file_path: str) -> bool:
    """Check if file is a supported video format"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_VIDEO_FORMATS

class EnhancedVisualChatState(TypedDict):
    """
    Enhanced state for multi-turn conversation with VLM integration
    Compatible with LangGraph checkpointing
    """
    history: List[Dict[str, Any]]
    video_path: str
    vlm_analysis_result: Optional[VLMAnalysisResult]
    frame_descriptions: List[Dict]
    detected_objects: List[str]
    detected_actions: List[str]
    scene_summary: str
    confidence_score: float
    processing_time: float
    model_used: str
    analysis_complete: bool
    error_message: Optional[str]

# Enhanced VLM Video Processing Node
def enhanced_vlm_process_node(state: EnhancedVisualChatState, **kwargs):
    """
    Process video using enhanced VLM analyzer
    """
    video_path = state["video_path"]
    if not video_path:
        raise ValueError("No video file uploaded.")
    
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    
    logger.info(f"🎬 Starting enhanced VLM analysis for: {os.path.basename(video_path)}")
    
    # Initialize VLM analyzer with preferred model
    # Try different models in order of preference
    model_preferences = ["qwen", "blip", "openai"]  # Add "llava" if available
    
    vlm_analyzer = None
    for model_type in model_preferences:
        try:
            vlm_analyzer = EnhancedVLMVideoAnalyzer(
                model_type=model_type,
                max_frames=20,  # Reasonable number for chat context
                frame_interval=30
            )
            if vlm_analyzer.model is not None:
                logger.info(f"✅ Successfully initialized {model_type} VLM analyzer")
                break
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize {model_type}: {e}")
            continue
    
    if vlm_analyzer is None or vlm_analyzer.model is None:
        # Fallback to mock analysis
        logger.warning("🔄 Using fallback mock analysis")
        state["vlm_analysis_result"] = None
        state["scene_summary"] = "VLM analysis not available - using fallback mode"
        state["detected_objects"] = ["person", "scene", "background"]
        state["detected_actions"] = ["activity", "movement"]
        state["frame_descriptions"] = [{"description": "Mock frame analysis", "confidence": 0.5}]
        state["confidence_score"] = 0.5
        state["processing_time"] = 1.0
        state["model_used"] = "fallback"
        state["analysis_complete"] = True
        state["error_message"] = "VLM models not available"
        state["history"].append({
            "role": "system", 
            "content": "Video processed using fallback analysis (VLM not available)"
        })
        return state
    
    try:
        # Run enhanced VLM analysis
        analysis_result = vlm_analyzer.analyze_video_with_vlm(video_path)
        
        if analysis_result.success:
            # Store comprehensive results in state
            state["vlm_analysis_result"] = analysis_result
            state["frame_descriptions"] = analysis_result.frame_descriptions
            state["detected_objects"] = analysis_result.detected_objects
            state["detected_actions"] = analysis_result.detected_actions
            state["scene_summary"] = analysis_result.scene_summary
            state["confidence_score"] = analysis_result.confidence_score
            state["processing_time"] = analysis_result.processing_time
            state["model_used"] = analysis_result.model_used
            state["analysis_complete"] = True
            state["error_message"] = None
            
            logger.info(f"✅ VLM analysis completed successfully in {analysis_result.processing_time:.1f}s")
            logger.info(f"📊 Confidence: {analysis_result.confidence_score:.2f}")
            logger.info(f"🔍 Found {len(analysis_result.detected_objects)} objects, {len(analysis_result.detected_actions)} actions")
            
            state["history"].append({
                "role": "system",
                "content": f"Enhanced VLM analysis completed using {analysis_result.model_used}. "
                          f"Processed {len(analysis_result.frame_descriptions)} frames in {analysis_result.processing_time:.1f}s "
                          f"with {analysis_result.confidence_score:.2f} confidence."
            })
        else:
            # Analysis failed
            state["vlm_analysis_result"] = analysis_result
            state["scene_summary"] = f"Analysis failed: {analysis_result.error_message}"
            state["detected_objects"] = []
            state["detected_actions"] = []
            state["frame_descriptions"] = []
            state["confidence_score"] = 0.0
            state["processing_time"] = analysis_result.processing_time
            state["model_used"] = analysis_result.model_used
            state["analysis_complete"] = False
            state["error_message"] = analysis_result.error_message
            
            state["history"].append({
                "role": "system",
                "content": f"VLM analysis failed: {analysis_result.error_message}"
            })
    
    except Exception as e:
        logger.error(f"❌ Enhanced VLM processing error: {e}")
        state["vlm_analysis_result"] = None
        state["scene_summary"] = f"Processing error: {str(e)}"
        state["detected_objects"] = []
        state["detected_actions"] = []
        state["frame_descriptions"] = []
        state["confidence_score"] = 0.0
        state["processing_time"] = 0.0
        state["model_used"] = "error"
        state["analysis_complete"] = False
        state["error_message"] = str(e)
        
        state["history"].append({
            "role": "system",
            "content": f"Video processing failed: {str(e)}"
        })
    
    return state

def enhanced_event_detection_node(state: EnhancedVisualChatState, **kwargs):
    """
    Enhanced event detection using VLM frame descriptions
    """
    frame_descriptions = state.get("frame_descriptions", [])
    
    if not frame_descriptions:
        logger.warning("🔍 No frame descriptions available for event detection")
        state["history"].append({
            "role": "system",
            "content": "No frame descriptions available for detailed event detection"
        })
        return state
    
    try:
        # Extract events from frame descriptions
        events = []
        for frame_desc in frame_descriptions:
            if frame_desc.get('description'):
                # Create event from frame description
                timestamp = frame_desc.get('timestamp', 0)
                description = frame_desc.get('description', '')
                confidence = frame_desc.get('confidence', 0.5)
                
                # Extract meaningful events from description
                if any(keyword in description.lower() for keyword in ['person', 'people', 'action', 'movement']):
                    events.append({
                        'timestamp': timestamp,
                        'description': description,
                        'confidence': confidence,
                        'type': 'scene_event'
                    })
        
        # Store events in a format compatible with the chat system
        event_summaries = []
        for event in events[:10]:  # Limit to first 10 events
            event_summaries.append(f"At {event['timestamp']:.1f}s: {event['description'][:100]}...")
        
        state["events"] = event_summaries
        state["detected_events_objects"] = events
        
        logger.info(f"🎯 Detected {len(events)} events from VLM analysis")
        state["history"].append({
            "role": "system",
            "content": f"Enhanced event detection completed: {len(events)} events identified"
        })
    
    except Exception as e:
        logger.error(f"❌ Event detection error: {e}")
        state["events"] = []
        state["detected_events_objects"] = []
        state["history"].append({
            "role": "system",
            "content": f"Event detection failed: {str(e)}"
        })
    
    return state

def enhanced_summarization_node(state: EnhancedVisualChatState, **kwargs):
    """
    Enhanced summarization using VLM analysis results
    """
    vlm_result = state.get("vlm_analysis_result")
    scene_summary = state.get("scene_summary", "")
    detected_objects = state.get("detected_objects", [])
    detected_actions = state.get("detected_actions", [])
    processing_time = state.get("processing_time", 0)
    confidence_score = state.get("confidence_score", 0)
    model_used = state.get("model_used", "unknown")
    
    try:
        if vlm_result and vlm_result.success:
            # Create comprehensive summary using VLM results
            summary_parts = [
                f"🎬 Enhanced Video Analysis Report",
                f"📊 Analysis Engine: {model_used}",
                f"⏱️ Processing Time: {processing_time:.1f}s",
                f"🎯 Confidence Score: {confidence_score:.2f}/1.0",
                f"",
                f"📝 Scene Summary:",
                f"{scene_summary}",
                f"",
                f"🔍 Key Findings:"
            ]
            
            if detected_objects:
                top_objects = detected_objects[:8]  # Top 8 objects
                summary_parts.append(f"📦 Objects Detected: {', '.join(top_objects)}")
            
            if detected_actions:
                top_actions = detected_actions[:5]  # Top 5 actions
                summary_parts.append(f"🏃 Activities Observed: {', '.join(top_actions)}")
            
            # Add frame-by-frame insights if available
            frame_descriptions = state.get("frame_descriptions", [])
            if frame_descriptions:
                summary_parts.extend([
                    f"",
                    f"📹 Frame Analysis ({len(frame_descriptions)} frames processed):"
                ])
                
                # Show key frames
                for i, frame_desc in enumerate(frame_descriptions[:3]):  # First 3 frames
                    timestamp = frame_desc.get('timestamp', 0)
                    description = frame_desc.get('description', '')[:80]
                    summary_parts.append(f"  • {timestamp:.1f}s: {description}...")
                
                if len(frame_descriptions) > 3:
                    summary_parts.append(f"  • ... and {len(frame_descriptions) - 3} more frames")
            
            summary_parts.extend([
                f"",
                f"💡 Analysis Quality: {'High' if confidence_score > 0.7 else 'Medium' if confidence_score > 0.4 else 'Low'}"
            ])
            
            final_summary = "\n".join(summary_parts)
        else:
            # Fallback summary
            final_summary = f"Video analysis completed with limited results.\n"
            if scene_summary:
                final_summary += f"Summary: {scene_summary}\n"
            if detected_objects:
                final_summary += f"Objects: {', '.join(detected_objects[:5])}\n"
            if detected_actions:
                final_summary += f"Actions: {', '.join(detected_actions[:3])}"
        
        state["enhanced_summary"] = final_summary
        
        logger.info("📋 Enhanced summarization completed")
        state["history"].append({
            "role": "system",
            "content": "Enhanced video summarization completed"
        })
    
    except Exception as e:
        logger.error(f"❌ Summarization error: {e}")
        state["enhanced_summary"] = f"Summarization failed: {str(e)}"
        state["history"].append({
            "role": "system",
            "content": f"Summarization failed: {str(e)}"
        })
    
    return state

class EnhancedAgenticVisualChatAssistant:
    """Enhanced Visual Chat Assistant using VLM Core Components"""
    
    def __init__(self, llm=None, session_id: str = "enhanced_session"):
        self.llm = llm or model
        self.session_id = session_id
        self.memory = MemorySaver()
        self.state = self._initialize_state()
        self.agent = self._initialize_agent()
        self._build_enhanced_workflow()
    
    def _initialize_state(self) -> EnhancedVisualChatState:
        """Initialize enhanced state structure"""
        return {
            "history": [],
            "video_path": "",
            "vlm_analysis_result": None,
            "frame_descriptions": [],
            "detected_objects": [],
            "detected_actions": [],
            "scene_summary": "",
            "confidence_score": 0.0,
            "processing_time": 0.0,
            "model_used": "",
            "analysis_complete": False,
            "error_message": None
        }
    
    def _initialize_agent(self):
        """Initialize LangChain agent with enhanced tools"""
        tools = [
            Tool(
                name="enhanced_vlm_process",
                func=enhanced_vlm_process_node,
                description="Process video using enhanced VLM analysis with multiple model support"
            ),
            Tool(
                name="enhanced_event_detection",
                func=enhanced_event_detection_node,
                description="Detect events using VLM frame descriptions"
            ),
            Tool(
                name="enhanced_summarization",
                func=enhanced_summarization_node,
                description="Generate comprehensive summary from VLM analysis"
            )
        ]
        
        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent
    
    def _build_enhanced_workflow(self):
        """Build enhanced LangGraph workflow"""
        workflow = StateGraph(EnhancedVisualChatState)
        
        # Add nodes
        workflow.add_node("enhanced_vlm_process", enhanced_vlm_process_node)
        workflow.add_node("enhanced_event_detection", enhanced_event_detection_node)
        workflow.add_node("enhanced_summarization", enhanced_summarization_node)
        
        # Define edges
        workflow.add_edge("enhanced_vlm_process", "enhanced_event_detection")
        workflow.add_edge("enhanced_event_detection", "enhanced_summarization")
        workflow.add_edge("enhanced_summarization", END)
        
        # Set entry point
        workflow.set_entry_point("enhanced_vlm_process")
        
        # Compile with memory
        self.workflow = workflow.compile(checkpointer=self.memory)
    
    def upload_video(self, video_path: str):
        """Upload and validate video file"""
        if not is_video_file(video_path):
            raise ValueError(f"Unsupported file format: {video_path}")
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        self.state["video_path"] = video_path
        self.state["history"].append({
            "role": "system",
            "content": f"📹 Video uploaded: '{os.path.basename(video_path)}' - Ready for enhanced VLM analysis"
        })
        
        logger.info(f"📹 Video uploaded: {os.path.basename(video_path)}")
    
    def run_enhanced_analysis(self):
        """Run the enhanced VLM analysis workflow"""
        if not self.state["video_path"]:
            raise ValueError("No video file uploaded.")
        
        logger.info("🚀 Starting enhanced VLM analysis workflow...")
        
        # Create thread config for session management
        thread_config = {"configurable": {"thread_id": self.session_id}}
        
        try:
            # Run enhanced workflow
            results = self.workflow.invoke(self.state, config=thread_config)
            
            # Update local state
            self.state.update(results)
            
            # Return enhanced summary
            enhanced_summary = results.get("enhanced_summary", "Analysis completed")
            return enhanced_summary
        
        except Exception as e:
            logger.error(f"❌ Workflow execution error: {e}")
            error_summary = f"Enhanced analysis failed: {str(e)}"
            self.state["history"].append({
                "role": "system",
                "content": error_summary
            })
            return error_summary
    
    def ask(self, user_message: str) -> str:
        """Enhanced chat interface with VLM context"""
        self.state["history"].append({"role": "user", "content": user_message})
        
        message_lower = user_message.lower()
        
        # Video analysis request
        if any(keyword in message_lower for keyword in ["analyze", "video", "what's in", "describe"]):
            if not self.state["video_path"]:
                response = "Please upload a video file first to begin enhanced VLM analysis."
                self.state["history"].append({"role": "assistant", "content": response})
                return response
            else:
                response = self.run_enhanced_analysis()
                self.state["history"].append({"role": "assistant", "content": response})
                return response
        
        # Objects query
        elif any(keyword in message_lower for keyword in ["objects", "what objects", "items"]):
            objects = self.state.get("detected_objects", [])
            if objects:
                response = f"🔍 Detected objects: {', '.join(objects[:10])}"
                if len(objects) > 10:
                    response += f" and {len(objects) - 10} more..."
            else:
                response = "No objects detected yet. Please analyze a video first."
            self.state["history"].append({"role": "assistant", "content": response})
            return response
        
        # Actions query
        elif any(keyword in message_lower for keyword in ["actions", "activities", "what happened"]):
            actions = self.state.get("detected_actions", [])
            if actions:
                response = f"🏃 Detected activities: {', '.join(actions[:8])}"
            else:
                response = "No activities detected yet. Please analyze a video first."
            self.state["history"].append({"role": "assistant", "content": response})
            return response
        
        # Summary request
        elif any(keyword in message_lower for keyword in ["summary", "summarize", "overview"]):
            summary = self.state.get("enhanced_summary", "")
            if summary:
                response = summary
            else:
                response = "No analysis summary available yet. Please analyze a video first."
            self.state["history"].append({"role": "assistant", "content": response})
            return response
        
        # Technical details
        elif any(keyword in message_lower for keyword in ["details", "technical", "confidence", "model"]):
            if self.state.get("analysis_complete"):
                response = f"📊 Analysis Details:\n"
                response += f"• Model Used: {self.state.get('model_used', 'Unknown')}\n"
                response += f"• Processing Time: {self.state.get('processing_time', 0):.1f}s\n"
                response += f"• Confidence Score: {self.state.get('confidence_score', 0):.2f}/1.0\n"
                response += f"• Frames Analyzed: {len(self.state.get('frame_descriptions', []))}\n"
                response += f"• Objects Found: {len(self.state.get('detected_objects', []))}\n"
                response += f"• Actions Found: {len(self.state.get('detected_actions', []))}"
            else:
                response = "No technical details available yet. Please analyze a video first."
            self.state["history"].append({"role": "assistant", "content": response})
            return response
        
        # General chat - use LLM with context
        else:
            try:
                # Build context from video analysis if available
                context_info = ""
                if self.state.get("analysis_complete"):
                    context_info = f"\n\nCurrent video analysis context:\n"
                    context_info += f"- Scene: {self.state.get('scene_summary', 'Unknown')[:100]}...\n"
                    context_info += f"- Objects: {', '.join(self.state.get('detected_objects', [])[:5])}\n"
                    context_info += f"- Activities: {', '.join(self.state.get('detected_actions', [])[:3])}"
                
                # Create enhanced prompt
                enhanced_prompt = user_message + context_info
                
                # Get chat history for context
                chat_history = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                    for msg in self.state["history"][-10:]  # Last 10 messages
                ]
                
                response = self.llm.invoke(chat_history + [HumanMessage(content=enhanced_prompt)])
                self.state["history"].append({"role": "assistant", "content": response.content})
                return response.content
            
            except Exception as e:
                logger.error(f"❌ LLM chat error: {e}")
                response = "I'm having trouble processing that request. Please try again or ask about the video analysis."
                self.state["history"].append({"role": "assistant", "content": response})
                return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.state["history"]
    
    def get_analysis_results(self) -> Optional[VLMAnalysisResult]:
        """Get detailed VLM analysis results"""
        return self.state.get("vlm_analysis_result")
    
    def clear_session(self):
        """Clear session and reset state"""
        self.state = self._initialize_state()
        logger.info("🔄 Session cleared")
    
    def start_enhanced_interactive_session(self):
        """Start enhanced interactive session"""
        print("🎬 Enhanced Visual Understanding Chat Assistant")
        print("=" * 60)
        print("🚀 Powered by Advanced VLM Analysis (Qwen, BLIP, OpenAI)")
        print("\nWelcome! I can analyze videos using state-of-the-art Vision-Language Models.")
        print("\n📋 Enhanced Features:")
        print("  • Multi-model VLM support (Qwen, BLIP, OpenAI GPT-4V)")
        print("  • Detailed frame-by-frame analysis")
        print("  • Object and activity detection")
        print("  • Confidence scoring and technical metrics")
        print("  • Smart fallback mechanisms")
        print("\n💬 Commands:")
        print("  • 'upload <path>' or just paste video path")
        print("  • 'analyze' or 'describe video' - Run VLM analysis")
        print("  • 'objects' - Show detected objects")
        print("  • 'actions' - Show detected activities")
        print("  • 'summary' - Get comprehensive summary")
        print("  • 'details' - Show technical analysis info")
        print("  • 'help', 'history', 'clear', 'status'")
        print("  • 'exit' or 'quit' - End session")
        print("\n💡 Pro tip: The system will auto-detect video files and smart-fallback if VLM is unavailable!")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                # Exit conditions
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\n🤖 Enhanced Assistant: Thanks for using the Enhanced VLM Chat Assistant! 🎬")
                    break
                
                # Help command
                elif user_input.lower() == 'help':
                    print("\n🤖 Enhanced Assistant: Here are the enhanced commands:")
                    print("  🎬 Video Analysis:")
                    print("    • 'analyze video' - Run comprehensive VLM analysis")
                    print("    • 'describe what you see' - Detailed scene description")
                    print("    • 'what objects are in the video?' - Object detection")
                    print("    • 'what activities happen?' - Activity detection")
                    print("    • 'summarize the video' - Complete summary")
                    print("  📊 Technical:")
                    print("    • 'show details' - Analysis metrics and confidence")
                    print("    • 'which model was used?' - Model information")
                    print("  🛠️ Utility:")
                    print("    • 'upload <path>' - Upload video file")
                    print("    • 'status' - Current session status")
                    print("    • 'history' - Conversation history")
                    print("    • 'clear' - Reset session")
                    continue
                
                # Upload command
                elif user_input.lower().startswith('upload '):
                    video_path = user_input[7:].strip().strip('"\'')
                    if not video_path:
                        print("\n🤖 Enhanced Assistant: Please provide a video path. Usage: upload <path>")
                        continue
                    
                    try:
                        self.upload_video(video_path)
                        print(f"\n🤖 Enhanced Assistant: ✅ Video uploaded successfully!")
                        print(f"   📁 File: {os.path.basename(video_path)}")
                        print("   🚀 Ready for enhanced VLM analysis!")
                    except Exception as e:
                        print(f"\n🤖 Enhanced Assistant: ❌ Upload failed: {e}")
                    continue
                
                # Status command
                elif user_input.lower() == 'status':
                    video_status = "✅ Uploaded" if self.state["video_path"] else "❌ No video"
                    analysis_status = "✅ Complete" if self.state.get("analysis_complete") else "❌ Not analyzed"
                    
                    print(f"\n🤖 Enhanced Assistant: 📊 Session Status:")
                    print(f"  📹 Video: {video_status}")
                    print(f"  🔍 Analysis: {analysis_status}")
                    
                    if self.state.get("analysis_complete"):
                        print(f"  🤖 Model Used: {self.state.get('model_used', 'Unknown')}")
                        print(f"  🎯 Confidence: {self.state.get('confidence_score', 0):.2f}")
                        print(f"  ⏱️ Processing: {self.state.get('processing_time', 0):.1f}s")
                        print(f"  📦 Objects: {len(self.state.get('detected_objects', []))}")
                        print(f"  🏃 Actions: {len(self.state.get('detected_actions', []))}")
                    
                    print(f"  💬 Messages: {len(self.state['history'])}")
                    print(f"  🆔 Session: {self.session_id}")
                    continue
                
                # History command
                elif user_input.lower() == 'history':
                    history = self.get_conversation_history()
                    if not history:
                        print("\n🤖 Enhanced Assistant: No conversation history yet.")
                    else:
                        print(f"\n🤖 Enhanced Assistant: 📜 History ({len(history)} messages):")
                        for i, msg in enumerate(history[-8:], 1):  # Last 8 messages
                            role_emoji = "👤" if msg["role"] == "user" else "🤖" if msg["role"] == "assistant" else "⚙️"
                            content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                            print(f"  {i}. {role_emoji} {msg['role'].title()}: {content_preview}")
                    continue
                
                # Clear command
                elif user_input.lower() == 'clear':
                    self.clear_session()
                    print("\n🤖 Enhanced Assistant: 🔄 Session cleared! Ready for fresh analysis.")
                    continue
                
                # Skip empty inputs
                elif not user_input:
                    continue
                
                # Auto-detect video file paths
                elif (user_input.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                      and (os.path.exists(user_input) or '\\' in user_input or '/' in user_input)):
                    print(f"\n🤖 Enhanced Assistant: 🎬 Auto-detected video file!")
                    print(f"   📁 Path: {user_input}")
                    try:
                        self.upload_video(user_input)
                        print(f"   ✅ Uploaded: {os.path.basename(user_input)}")
                        print("   💡 Try saying 'analyze video' to start VLM analysis!")
                    except Exception as e:
                        print(f"   ❌ Upload failed: {e}")
                    continue
                
                # Main chat processing
                print("\n🤖 Enhanced Assistant: ", end="", flush=True)
                response = self.ask(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n🤖 Enhanced Assistant: Session interrupted. Goodbye! 🎬")
                break
            except EOFError:
                print("\n\n🤖 Enhanced Assistant: Session ended. Goodbye! 🎬")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("💡 Please try again or type 'exit' to quit.")


# Example usage and main execution
if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Enhanced Visual Understanding Chat Assistant with VLM")
    parser.add_argument("--video", "-v", type=str, help="Path to video file for automatic upload")
    parser.add_argument("--session-id", "-s", type=str, default="enhanced_vlm_session",
                       help="Session ID for memory management")
    parser.add_argument("--model-preference", "-m", type=str, default="qwen",
                       choices=["qwen", "blip", "openai", "llava"],
                       help="Preferred VLM model (will fallback if unavailable)")
    args = parser.parse_args()
    
    print("🚀 Initializing Enhanced VLM Chat Assistant...")
    
    try:
        # Initialize enhanced assistant
        assistant = EnhancedAgenticVisualChatAssistant(session_id=args.session_id)
        
        # Auto-upload video if provided
        video_uploaded = False
        if args.video:
            try:
                assistant.upload_video(args.video)
                print(f"✅ Auto-uploaded video: {os.path.basename(args.video)}")
                video_uploaded = True
            except Exception as e:
                print(f"⚠️ Auto-upload failed: {e}")
        
        # Try fallback hardcoded path (from your original example)
        if not video_uploaded:
            fallback_paths = [
                r"D:\Work\Virtual_house\Clique\WhatsApp Video 2025-08-05 at 15.19.09_61fe1513.mp4",
                # Add your video paths here
            ]
            
            for video_path in fallback_paths:
                if os.path.exists(video_path):
                    try:
                        assistant.upload_video(video_path)
                        print(f"✅ Loaded fallback video: {os.path.basename(video_path)}")
                        break
                    except Exception as e:
                        print(f"⚠️ Fallback upload failed: {e}")
        
        # Start enhanced interactive session
        assistant.start_enhanced_interactive_session()
        
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced VLM Assistant: {e}")
        print("💡 Please check your environment setup and dependencies.")
        sys.exit(1)
