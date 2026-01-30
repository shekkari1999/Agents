"""FastAPI web application for the Agent Framework."""

import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent_framework import (
    Agent, LlmClient, InMemorySessionManager, 
    display_trace, ExecutionContext, format_trace
)
from agent_tools import calculator, search_web, read_file, list_files, unzip_file, read_media_file

# Load environment variables
load_dotenv()

app = FastAPI(title="Agent Chat", description="AI Agent with Tools")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session manager (shared across requests)
session_manager = InMemorySessionManager()

# Upload directory for files
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Available tools
TOOLS = [calculator, search_web, read_file, list_files, unzip_file, read_media_file]

# Create agent  
def create_agent(use_session: bool = True) -> Agent:
    """Create an agent instance."""
    
    # Include the actual upload directory path in instructions
    upload_path = str(UPLOAD_DIR.absolute())
    
    instructions = f"""You are a helpful AI assistant with access to various tools.

You can:
- Perform calculations using the calculator
- Search the web for current information
- Read excel files using the read_file tool
- List files in directories using the list_files tool
- Extract zip files using the unzip_file tool
- Read pdf using read_media_file

IMPORTANT - Uploaded files location:
Files uploaded by users are stored at: {upload_path}
To see uploaded files, use: list_files("{upload_path}")
To read a file, use: read_file("{upload_path}/filename.ext")

Always be helpful and use your tools when needed to provide accurate answers."""

    return Agent(
        model=LlmClient(model="gpt-4o-mini"),
        tools=TOOLS,
        instructions=instructions,
        max_steps=10,
        session_manager=session_manager if use_session else None
    )


# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_session: bool = True


class ChatResponse(BaseModel):
    response: str
    session_id: str
    events_count: int
    tools_used: List[str]
    trace_text: str = ""  # Simple text-based trace like display_trace


class ToolInfo(BaseModel):
    name: str
    description: str


class SessionInfo(BaseModel):
    session_id: str
    events_count: int
    created_at: str


# API Endpoints
@app.get("/")
async def root():
    """Serve the chat interface."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/tools")
async def get_tools() -> List[ToolInfo]:
    """Get list of available tools."""
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
        )
        for tool in TOOLS
    ]


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent."""
    
    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Create agent
    agent = create_agent(use_session=request.use_session)
    
    try:
        # Run the agent
        if request.use_session:
            result = await agent.run(request.message, session_id=session_id)
        else:
            result = await agent.run(request.message)
        
        # Extract tools used
        tools_used = []
        for event in result.context.events:
            for item in event.content:
                if hasattr(item, 'name') and item.type == "tool_call":
                    if item.name not in tools_used:
                        tools_used.append(item.name)
        
        # Use your format_trace function directly!
        trace_text = format_trace(result.context)
        
        return ChatResponse(
            response=str(result.output) if result.output else "I couldn't generate a response.",
            session_id=session_id,
            events_count=len(result.context.events),
            tools_used=tools_used,
            trace_text=trace_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for the agent to access."""
    
    # Save file to uploads directory
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "message": f"File uploaded successfully. You can reference it at: {file_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/uploads")
async def list_uploads():
    """List uploaded files."""
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            files.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size
            })
    return files


@app.delete("/api/uploads/{filename}")
async def delete_upload(filename: str):
    """Delete an uploaded file."""
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"message": f"Deleted {filename}"}
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/sessions")
async def list_sessions() -> List[SessionInfo]:
    """List all active sessions."""
    sessions = []
    for sid, session in session_manager._sessions.items():
        sessions.append(SessionInfo(
            session_id=sid,
            events_count=len(session.events),
            created_at=session.created_at.isoformat()
        ))
    return sessions


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session to clear conversation history."""
    if session_id in session_manager._sessions:
        del session_manager._sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")


# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

