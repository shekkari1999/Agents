# Episode 10: Web Deployment

**Duration**: 35 minutes  
**What to Build**: `web_app/app.py`, `web_app/static/index.html`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Complete web application
- Chat interface
- File uploads
- Session management

**Hook Statement**: "Today we'll deploy our agent as a web application. This is the final piece - making it accessible to users through a beautiful interface!"

---

### 2. Problem (3 min)
**Why deploy as a web app?**

**The Challenge:**
- Command-line isn't user-friendly
- Need file uploads
- Want real-time interaction
- Need session management UI

**The Solution:**
- FastAPI backend
- Modern frontend
- RESTful API
- Static file serving

---

### 3. Concept: Web Architecture (5 min)

**Architecture:**
```
Frontend (HTML/CSS/JS)
    ↓ HTTP Requests
FastAPI Backend
    ↓
Agent Framework
    ↓
LLM API
```

**Components:**
- Backend: FastAPI server
- Frontend: Static HTML/CSS/JS
- API: RESTful endpoints
- File handling: Upload directory

---

### 4. Live Coding: Building Web App (25 min)

#### Step 1: FastAPI Setup (3 min)
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uuid

app = FastAPI(title="Agent Chat")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
```

**Key Points:**
- FastAPI app
- CORS enabled
- Static file serving

**Live Coding**: Setup FastAPI

---

#### Step 2: Agent Creation (2 min)
```python
from agent_framework import Agent, LlmClient, InMemorySessionManager
from agent_tools import calculator, search_web, read_file

session_manager = InMemorySessionManager()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def create_agent(use_session: bool = True) -> Agent:
    """Create an agent instance."""
    return Agent(
        model=LlmClient(model="gpt-4o-mini"),
        tools=[calculator, search_web, read_file],
        instructions="You are a helpful assistant.",
        max_steps=10,
        session_manager=session_manager if use_session else None
    )
```

**Key Points:**
- Shared session manager
- Configurable tools
- Session toggle

**Live Coding**: Build agent creation

---

#### Step 3: API Models (2 min)
```python
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_session: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str
    events_count: int
    tools_used: List[str]
    trace: str
```

**Key Points:**
- Request/response models
- Session support
- Trace included

**Live Coding**: Build API models

---

#### Step 4: Chat Endpoint (5 min)
```python
@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent."""
    session_id = request.session_id or str(uuid.uuid4())
    agent = create_agent(use_session=request.use_session)
    
    try:
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
        
        # Format trace
        from agent_framework import format_trace
        trace_output = format_trace(result.context)
        
        return ChatResponse(
            response=str(result.output) if result.output else "No response.",
            session_id=session_id,
            events_count=len(result.context.events),
            tools_used=tools_used,
            trace=trace_output
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Key Points:**
- Handles sessions
- Extracts tools
- Formats trace
- Error handling

**Live Coding**: Build chat endpoint

---

#### Step 5: File Upload Endpoint (3 min)
```python
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file."""
    file_path = UPLOAD_DIR / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {"filename": file.filename, "path": str(file_path)}
```

**Key Points:**
- Saves to uploads directory
- Returns file info

**Live Coding**: Build upload endpoint

---

#### Step 6: Frontend HTML Structure (5 min)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Agent Chat</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Tools</h2>
            <div id="tools-list"></div>
            
            <h2>Files</h2>
            <input type="file" id="file-input">
            <div id="files-list"></div>
        </div>
        
        <div class="chat-area">
            <div class="chat-header">
                <h1>Agent Chat</h1>
                <button id="trace-btn">View Trace</button>
            </div>
            <div id="messages"></div>
            <div class="input-area">
                <input type="text" id="message-input" placeholder="Type a message...">
                <button id="send-btn">Send</button>
            </div>
        </div>
    </div>
    
    <script src="/static/script.js"></script>
</body>
</html>
```

**Key Points:**
- Sidebar for tools/files
- Chat area
- Input area
- Trace button

**Live Coding**: Build HTML structure

---

#### Step 7: Frontend JavaScript (5 min)
```javascript
let sessionId = null;

async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value;
    if (!message) return;
    
    // Add user message to UI
    addMessage('user', message);
    input.value = '';
    
    // Send to API
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            message: message,
            session_id: sessionId,
            use_session: true
        })
    });
    
    const data = await response.json();
    sessionId = data.session_id;
    
    // Add agent response
    addMessage('assistant', data.response);
}

function addMessage(role, content) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = content;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
```

**Key Points:**
- Session management
- Message sending
- UI updates

**Live Coding**: Build JavaScript

---

### 5. Demo: Complete Web App (3 min)

**Show:**
- Chat interface
- File upload
- Tool listing
- Trace display
- Session persistence

---

### 6. Deployment Tips (2 min)

**Development:**
```bash
uvicorn web_app.app:app --reload
```

**Production:**
```bash
uvicorn web_app.app:app --host 0.0.0.0 --port 8000
```

**Docker:**
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 7. Next Steps (1 min)

**What We Built:**
- Complete web application
- Full agent framework
- Production-ready system

**Future Enhancements:**
- WebSocket for streaming
- User authentication
- Database sessions
- Monitoring and logging

---

## Key Takeaways

1. **FastAPI** provides easy API creation
2. **Static files** for frontend
3. **RESTful API** for communication
4. **Session management** via API
5. **File handling** for uploads

---

## Common Mistakes

**Mistake 1: Not handling CORS**
```python
# Wrong - CORS errors
app = FastAPI()

# Right - CORS enabled
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

**Mistake 2: Not serving static files**
```python
# Wrong - can't load CSS/JS
app = FastAPI()

# Right - serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
```

---

## Exercises

1. Add WebSocket support
2. Implement user authentication
3. Add database session storage
4. Create admin dashboard

---

**Previous Episode**: [Episode 9: Session & Memory Management](./EPISODE_09_MEMORY.md)  
**Series Complete!** 🎉

---

## Series Summary

You've built:
- Complete agent framework
- Tool system
- MCP integration
- Memory management
- Web deployment

**Congratulations on completing the series!**

