"""Core data models for the agent framework."""

from typing import Literal, Union, List, Dict, Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import uuid
from datetime import datetime


class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


ContentItem = Union[Message, ToolCall, ToolResult]

class ToolConfirmation(BaseModel):
    """User's decision on a pending tool call."""
    
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None  # Reason for rejection (if not approved)

class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    
    tool_call: ToolCall
    confirmation_message: str

class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    session_id: Optional[str] = None  # Link to session for persistence
    
    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)
    
    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1

class Session(BaseModel):
    """Container for persistent conversation state across multiple run() calls."""
    
    session_id: str
    user_id: str | None = None
    events: list[Event] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

from abc import ABC, abstractmethod
 
class BaseSessionManager(ABC):
    """Abstract base class for session management."""
    
    @abstractmethod
    async def create(
        self, 
        session_id: str, 
        user_id: str | None = None
    ) -> Session:
        """Create a new session."""
        pass
    
    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID. Returns None if not found."""
        pass
    
    @abstractmethod
    async def save(self, session: Session) -> None:
        """Persist session changes to storage."""
        pass
    
    async def get_or_create(
        self, 
        session_id: str, 
        user_id: str | None = None
    ) -> Session:
        """Get existing session or create new one."""
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id, user_id)
        return session

class InMemorySessionManager(BaseSessionManager):
    """In-memory session storage for development and testing."""
    
    def __init__(self):
        self._sessions: dict[str, Session] = {}
    
    async def create(
        self, 
        session_id: str, 
        user_id: str | None = None
    ) -> Session:
        """Create a new session."""
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        session = Session(
            session_id=session_id,
            user_id=user_id
        )
        self._sessions[session_id] = session
        return session
    
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)
    
    async def save(self, session: Session) -> None:
        """Save session to storage."""
        self._sessions[session.session_id] = session