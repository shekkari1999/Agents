# Agent Chat Web Application

A modern chat interface for interacting with the AI agent framework.

## Features

- Real-time chat with AI agent
- Session memory toggle (on/off)
- File upload support
- Display of available tools
- Tool usage indicators in responses

## Running the Application

### Option 1: Direct run

```bash
cd web_app
python app.py
```

### Option 2: With uvicorn (recommended)

```bash
uvicorn web_app.app:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/tools` | GET | List available tools |
| `/api/chat` | POST | Send message to agent |
| `/api/upload` | POST | Upload a file |
| `/api/uploads` | GET | List uploaded files |
| `/api/uploads/{filename}` | DELETE | Delete uploaded file |
| `/api/sessions` | GET | List active sessions |
| `/api/sessions/{session_id}` | DELETE | Clear a session |

## Chat Request Format

```json
{
    "message": "Your message here",
    "session_id": "optional-session-id",
    "use_session": true
}
```

## Chat Response Format

```json
{
    "response": "Agent's response",
    "session_id": "session-uuid",
    "events_count": 4,
    "tools_used": ["calculator", "search_web"]
}
```

