from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoEvent(BaseModel):
    timestamp: float = Field(description="Event timestamp in seconds")
    event_type: str = Field(description="Type of event detected")
    description: str = Field(description="Detailed event description")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    guideline_adherence: bool = Field(description="Whether event follows guidelines")

class VideoAnalysis(BaseModel):
    video_id: str
    duration: float
    events: List[VideoEvent]
    summary: str
    guideline_violations: List[VideoEvent]
    processed_at: datetime

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class ConversationState(BaseModel):
    session_id: str
    video_analysis: Optional[VideoAnalysis] = None
    messages: List[ChatMessage] = []
    context: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: bool = False