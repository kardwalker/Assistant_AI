from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Dict, Any, List, Optional
from schema import ChatRequest, ChatResponse, ConversationState, ChatMessage, VideoAnalysis, VideoEvent
import os
from dotenv import load_dotenv
import asyncio
from langgraph.checkpoint.memory import MemorySaver
import uuid
from summazier import AudioSummarizer
from video_analyzer import VideoContentAnalyzer
from separator import process_video
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.tools import Tool
from datetime import datetime
import json
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompts
SUPERVISOR_PROMPT = """
You are a supervisor agent for video analysis. Your role is to coordinate between different specialized agents:
1. Video Separator - Extracts audio and video components
2. Audio Summarizer - Transcribes and summarizes audio content
3. Video Content Analyzer - Analyzes visual content and detects events

Based on the user's request, decide which tools to use and in what order. Always provide a comprehensive analysis.

Current conversation context: {context}
Previous messages: {messages}
"""

ANALYSIS_PROMPT = """
You are an expert video content analyst. Based on the following analysis results, provide a comprehensive summary:

Audio Analysis: {audio_summary}
Video Analysis: {video_summary}
Video Events: {video_events}

Provide insights about:
1. Content summary
2. Key events and timestamps
3. Any guideline violations or concerning content
4. Overall assessment

Format your response in a clear, structured manner.
"""

class VideoAnalysisState(ConversationState):
    """Extended state for video analysis workflow"""
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_only_path: Optional[str] = None
    audio_summary: Optional[str] = None
    video_summary: Optional[str] = None
    video_events: Optional[List[VideoEvent]] = None
    processing_stage: str = "initial"
    error_message: Optional[str] = None


class VideoAnalysisAgent:
    """Main agent class for video analysis using LangGraph"""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-02-01",
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"),
            temperature=0.1
        )
        
        # Initialize specialized components
        self.audio_summarizer = AudioSummarizer()
        self.video_analyzer = VideoContentAnalyzer()
        
        # Initialize memory for conversation management
        self.memory = MemorySaver()
        self.sessions: Dict[str, VideoAnalysisState] = {}
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        # Output parsers
        self.str_parser = StrOutputParser()
        self.video_analysis_parser = PydanticOutputParser(pydantic_object=VideoAnalysis)
        
        logger.info("VideoAnalysisAgent initialized successfully")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with parallel processing"""
        workflow = StateGraph(VideoAnalysisState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("video_separator", self._video_separator_node)
        workflow.add_node("parallel_analyzer", self._parallel_analyzer_node)
        workflow.add_node("final_analysis", self._final_analysis_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the workflow edges for parallel processing
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            {
                "separate_video": "video_separator",
                "error": "error_handler"
            }
        )
        workflow.add_edge("video_separator", "parallel_analyzer")
        workflow.add_edge("parallel_analyzer", "final_analysis")
        workflow.add_edge("final_analysis", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _supervisor_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Supervisor node that coordinates the analysis"""
        try:
            # Create supervisor prompt
            supervisor_prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
            
            # Format context and messages for the prompt
            context = json.dumps(state.context, indent=2) if state.context else "No context"
            messages_str = "\n".join([f"{msg.role}: {msg.content}" for msg in state.messages[-5:]])
            
            # Get supervisor decision
            chain = supervisor_prompt | self.llm | self.str_parser
            response = chain.invoke({
                "context": context,
                "messages": messages_str
            })
            
            # Update state
            state.processing_stage = "supervised"
            state.context["supervisor_decision"] = response
            
            logger.info(f"Supervisor decision: {response}")
            return state
            
        except Exception as e:
            logger.error(f"Error in supervisor node: {str(e)}")
            state.error_message = f"Supervisor error: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _supervisor_router(self, state: VideoAnalysisState) -> str:
        """Route based on supervisor decision and current state"""
        if state.error_message:
            return "error"
        
        if state.video_path and state.processing_stage == "supervised":
            return "separate_video"
        
        return "error"
    
    def _video_separator_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Separate video into audio and video components"""
        try:
            if not state.video_path or not os.path.exists(state.video_path):
                raise ValueError(f"Video file not found: {state.video_path}")
            
            logger.info(f"Processing video: {state.video_path}")
            
            # Use the separator tool
            result = process_video(state.video_path)
            
            if result['success']:
                state.audio_path = result['audio']
                state.video_only_path = result['video']
                state.processing_stage = "separated"
                logger.info(f"Video separated successfully: audio={result['audio']}, video={result['video']}")
            else:
                raise Exception("Video separation failed")
                
            return state
            
        except Exception as e:
            logger.error(f"Error in video separator: {str(e)}")
            state.error_message = f"Video separation error: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _parallel_analyzer_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Process audio and video analysis simultaneously"""
        try:
            logger.info("Starting parallel audio and video analysis...")
            
            # Use concurrent futures for parallel execution
            import concurrent.futures
            
            # Create executor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                audio_future = executor.submit(self._analyze_audio_sync, state)
                video_future = executor.submit(self._analyze_video_sync, state)
                
                # Wait for both to complete
                audio_result = audio_future.result()
                video_result = video_future.result()
            
            # Update state with results
            state.audio_summary = audio_result
            state.video_summary = video_result.get('summary', '')
            state.video_events = video_result.get('events', [])
            state.processing_stage = "parallel_analyzed"
            
            logger.info("Parallel analysis completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in parallel analyzer: {str(e)}")
            state.error_message = f"Parallel analysis error: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _analyze_audio_sync(self, state: VideoAnalysisState) -> str:
        """Synchronous audio analysis for parallel execution"""
        if not state.audio_path or not os.path.exists(state.audio_path):
            raise ValueError(f"Audio file not found: {state.audio_path}")
        
        logger.info(f"Analyzing audio: {state.audio_path}")
        
        summary_result = self.audio_summarizer.summarize_audio_content(state.audio_path)
        if summary_result['success']:
            return summary_result['summary']
        else:
            raise Exception(f"Audio analysis failed: {summary_result.get('error', 'Unknown error')}")
    
    def _analyze_video_sync(self, state: VideoAnalysisState) -> Dict[str, Any]:
        """Synchronous video analysis for parallel execution"""
        if not state.video_only_path or not os.path.exists(state.video_only_path):
            raise ValueError(f"Video file not found: {state.video_only_path}")
        
        logger.info(f"Analyzing video: {state.video_only_path}")
        
        analysis_result = self.video_analyzer.analyze_video(state.video_only_path)
        
        if analysis_result:
            # Convert to VideoEvent objects
            events = [
                VideoEvent(
                    timestamp=event.get('timestamp', 0.0),
                    event_type=event.get('type', 'unknown'),
                    description=event.get('description', ''),
                    confidence=event.get('confidence', 0.0),
                    guideline_adherence=event.get('guideline_adherence', True)
                ) for event in analysis_result.events
            ]
            
            return {
                'summary': analysis_result.summary,
                'events': events
            }
        else:
            raise Exception("Video analysis returned no results")
    
    def _audio_analyzer_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Analyze audio content (kept for backward compatibility)"""
        try:
            if not state.audio_path or not os.path.exists(state.audio_path):
                raise ValueError(f"Audio file not found: {state.audio_path}")
            
            logger.info(f"Analyzing audio: {state.audio_path}")
            
            # Use audio summarizer
            summary_result = self.audio_summarizer.summarize_audio_content(state.audio_path)
            
            if summary_result['success']:
                state.audio_summary = summary_result['summary']
                state.processing_stage = "audio_analyzed"
                logger.info("Audio analysis completed successfully")
            else:
                raise Exception(f"Audio analysis failed: {summary_result.get('error', 'Unknown error')}")
                
            return state
            
        except Exception as e:
            logger.error(f"Error in audio analyzer: {str(e)}")
            state.error_message = f"Audio analysis error: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _video_analyzer_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Analyze video content (kept for backward compatibility)"""
        try:
            if not state.video_only_path or not os.path.exists(state.video_only_path):
                raise ValueError(f"Video file not found: {state.video_only_path}")
            
            logger.info(f"Analyzing video: {state.video_only_path}")
            
            # Use video analyzer
            analysis_result = self.video_analyzer.analyze_video(state.video_only_path)
            
            if analysis_result:
                state.video_summary = analysis_result.summary
                # Convert to VideoEvent objects if needed
                state.video_events = [
                    VideoEvent(
                        timestamp=event.get('timestamp', 0.0),
                        event_type=event.get('type', 'unknown'),
                        description=event.get('description', ''),
                        confidence=event.get('confidence', 0.0),
                        guideline_adherence=event.get('guideline_adherence', True)
                    ) for event in analysis_result.events
                ]
                state.processing_stage = "video_analyzed"
                logger.info("Video analysis completed successfully")
            else:
                raise Exception("Video analysis returned no results")
                
            return state
            
        except Exception as e:
            logger.error(f"Error in video analyzer: {str(e)}")
            state.error_message = f"Video analysis error: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _final_analysis_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Generate final comprehensive analysis"""
        try:
            # Create final analysis prompt
            analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
            
            # Format video events
            events_str = "\n".join([
                f"- {event.timestamp}s: {event.event_type} - {event.description} (confidence: {event.confidence})"
                for event in (state.video_events or [])
            ])
            
            # Generate final analysis
            chain = analysis_prompt | self.llm | self.str_parser
            final_response = chain.invoke({
                "audio_summary": state.audio_summary or "No audio analysis available",
                "video_summary": state.video_summary or "No video analysis available",
                "video_events": events_str or "No events detected"
            })
            
            # Create comprehensive video analysis
            video_analysis = VideoAnalysis(
                video_id=str(uuid.uuid4()),
                duration=0.0,  # Would need to get this from video metadata
                events=state.video_events or [],
                summary=final_response,
                guideline_violations=[e for e in (state.video_events or []) if not e.guideline_adherence],
                processed_at=datetime.now()
            )
            
            # Update state
            state.video_analysis = video_analysis
            state.processing_stage = "completed"
            
            # Add response message
            response_message = ChatMessage(
                role="assistant",
                content=final_response,
                timestamp=datetime.now(),
                metadata={"analysis_id": video_analysis.video_id}
            )
            state.messages.append(response_message)
            
            logger.info("Final analysis completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in final analysis: {str(e)}")
            state.error_message = f"Final analysis error: {str(e)}"
            state.processing_stage = "error"
            return state
    
    def _error_handler_node(self, state: VideoAnalysisState) -> VideoAnalysisState:
        """Handle errors and provide user feedback"""
        error_message = state.error_message or "An unknown error occurred"
        
        error_response = ChatMessage(
            role="assistant",
            content=f"I encountered an error while processing your video: {error_message}. Please try again or contact support if the issue persists.",
            timestamp=datetime.now(),
            metadata={"error": True}
        )
        
        state.messages.append(error_response)
        state.processing_stage = "error_handled"
        
        logger.error(f"Error handled: {error_message}")
        return state
    
    async def process_video_analysis(self, video_path: str, session_id: str, user_message: str = None) -> ChatResponse:
        """Process a video analysis request"""
        try:
            # Get or create session state
            if session_id not in self.sessions:
                self.sessions[session_id] = VideoAnalysisState(
                    session_id=session_id,
                    messages=[],
                    context={}
                )
            
            state = self.sessions[session_id]
            
            # Add user message
            if user_message:
                user_msg = ChatMessage(
                    role="user",
                    content=user_message,
                    timestamp=datetime.now()
                )
                state.messages.append(user_msg)
            
            # Set video path
            state.video_path = video_path
            
            # Run the workflow
            config = {"configurable": {"thread_id": session_id}}
            result_state = await self.workflow.ainvoke(state, config=config)
            
            # Update session
            self.sessions[session_id] = result_state
            
            # Get the last assistant message
            assistant_messages = [msg for msg in result_state.messages if msg.role == "assistant"]
            last_response = assistant_messages[-1].content if assistant_messages else "Analysis completed"
            
            return ChatResponse(
                response=last_response,
                session_id=session_id,
                context_used=bool(result_state.video_analysis)
            )
            
        except Exception as e:
            logger.error(f"Error in process_video_analysis: {str(e)}")
            error_response = ChatResponse(
                response=f"An error occurred during video analysis: {str(e)}",
                session_id=session_id,
                context_used=False
            )
            return error_response
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle general chat requests with context awareness"""
        try:
            # Get session state
            if request.session_id not in self.sessions:
                self.sessions[request.session_id] = VideoAnalysisState(
                    session_id=request.session_id,
                    messages=[],
                    context={}
                )
            
            state = self.sessions[request.session_id]
            
            # Add user message
            user_msg = ChatMessage(
                role="user",
                content=request.message,
                timestamp=datetime.now()
            )
            state.messages.append(user_msg)
            
            # Check if there's previous video analysis context
            context_prompt = ""
            if state.video_analysis:
                context_prompt = f"""
                Previous video analysis context:
                - Video ID: {state.video_analysis.video_id}
                - Summary: {state.video_analysis.summary}
                - Events: {len(state.video_analysis.events)} detected
                - Violations: {len(state.video_analysis.guideline_violations)} found
                
                User question: {request.message}
                
                Answer based on the video analysis context when relevant.
                """
            else:
                context_prompt = f"User message: {request.message}"
            
            # Generate response using LLM
            response = await self.llm.ainvoke(context_prompt)
            
            # Add assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content,
                timestamp=datetime.now()
            )
            state.messages.append(assistant_msg)
            
            # Update session
            self.sessions[request.session_id] = state
            
            return ChatResponse(
                response=response.content,
                session_id=request.session_id,
                context_used=bool(state.video_analysis)
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return ChatResponse(
                response=f"I encountered an error: {str(e)}",
                session_id=request.session_id,
                context_used=False
            )
    
    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        """Get conversation history for a session"""
        if session_id in self.sessions:
            return self.sessions[session_id].messages
        return []
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session's data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


# Example usage and testing
async def main():
    """Example usage of the VideoAnalysisAgent"""
    agent = VideoAnalysisAgent()
    
    # Example video analysis
    video_path = "C:\\Users\\Aman\\Desktop\\Virtual__house\\Vuencode_AI_Hackthon\\Vuenagent\\WhatsApp Video 2024-04-10 at 16.41.51_71572a51.mp4"
    session_id = str(uuid.uuid4())
    
    print("Starting video analysis...")
    result = await agent.process_video_analysis(
        video_path=video_path,
        session_id=session_id,
        user_message="Please analyze this video and provide a comprehensive summary"
    )
    
    print(f"Analysis Result: {result.response}")
    
    # Follow-up questions
    follow_up = await agent.chat(ChatRequest(
        message="What were the main events detected in the video?",
        session_id=session_id
    ))
    
    print(f"Follow-up Response: {follow_up.response}")


if __name__ == "__main__":
    asyncio.run(main())
















 