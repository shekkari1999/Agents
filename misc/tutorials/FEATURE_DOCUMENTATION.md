# AI Agent Framework: Complete Feature Documentation

## Overview

This document provides a comprehensive inventory of all features implemented in the AI Agent Framework. This framework allows you to build AI agents that can reason, use tools, maintain conversation state, and be deployed as web applications.

---

## Core Framework Features

### 1. Agent Execution Engine (`agent_framework/agent.py`)

The `Agent` class is the heart of the framework, orchestrating the entire execution flow.

#### Key Features:

- **Think-Act-Observe Loop**: Multi-step reasoning where the agent thinks (calls LLM), acts (executes tools), and observes (processes results) in a continuous cycle
- **Structured Output**: Support for Pydantic models as output types, ensuring type-safe responses
- **Max Steps Control**: Configurable iteration limits to prevent infinite loops
- **Tool Confirmation**: Optional user approval for tool execution before running
- **Pending Tool Calls**: Suspend execution and wait for user confirmation when required
- **Callbacks**: `before_tool_callbacks` and `after_tool_callbacks` for extensibility
- **Session Integration**: Persistent conversation state across multiple runs

#### Key Methods:

```python
Agent.__init__(
    model: LlmClient,
    tools: List[BaseTool] = None,
    instructions: str = "",
    max_steps: int = 5,
    output_type: Optional[Type[BaseModel]] = None,
    session_manager: BaseSessionManager | None = None
)

Agent.run(
    user_input: str,
    context: ExecutionContext = None,
    session_id: Optional[str] = None,
    tool_confirmations: Optional[List[ToolConfirmation]] = None
) -> AgentResult

Agent.step(context: ExecutionContext)  # Single iteration
Agent.think(llm_request: LlmRequest) -> LlmResponse  # LLM call
Agent.act(context: ExecutionContext, tool_calls: List[ToolCall]) -> List[ToolResult]  # Tool execution
```

---

### 2. Data Models (`agent_framework/models.py`)

All data structures used throughout the framework.

#### Message
Represents a text message in the conversation.

```python
class Message(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str
```

#### ToolCall
LLM's request to execute a tool.

```python
class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict
```

#### ToolResult
Result from tool execution (success or error).

```python
class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list
```

#### Event
A recorded occurrence during agent execution with timestamp.

```python
class Event(BaseModel):
    id: str
    execution_id: str
    timestamp: float
    author: str  # "user" or agent name
    content: List[ContentItem]
```

#### ExecutionContext
Central state container (dataclass) for all execution state.

```python
@dataclass
class ExecutionContext:
    execution_id: str
    events: List[Event]
    current_step: int
    state: Dict[str, Any]
    final_result: Optional[str | BaseModel]
    session_id: Optional[str]
```

#### Session
Persistent conversation state across multiple `run()` calls.

```python
class Session(BaseModel):
    session_id: str
    user_id: str | None
    events: list[Event]
    state: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

#### Session Management
- **BaseSessionManager**: Abstract interface for session storage
- **InMemorySessionManager**: In-memory implementation for development/testing

#### Tool Confirmation
- **ToolConfirmation**: User's decision on a pending tool call (approved/rejected with optional modifications)
- **PendingToolCall**: Tool calls awaiting user confirmation

---

### 3. LLM Client (`agent_framework/llm.py`)

Unified interface for interacting with LLM APIs.

#### LlmClient
Multi-provider support via LiteLLM (OpenAI, Anthropic, local models).

```python
class LlmClient:
    def __init__(self, model: str, **config)
    async def generate(self, request: LlmRequest) -> LlmResponse
```

#### LlmRequest
Structured request model.

```python
class LlmRequest(BaseModel):
    instructions: List[str]
    contents: List[ContentItem]
    tools: List[BaseTool]
    tool_choice: Optional[str]  # "auto", "required", or None
```

#### LlmResponse
Structured response model.

```python
class LlmResponse(BaseModel):
    content: List[ContentItem]
    error_message: Optional[str]
    usage_metadata: Dict[str, Any]
```

#### build_messages()
Converts internal models to API message format.

```python
def build_messages(request: LlmRequest) -> List[dict]
```

---

### 4. Tool System (`agent_framework/tools.py`)

Complete tool abstraction layer.

#### BaseTool
Abstract base class for all tools.

```python
class BaseTool(ABC):
    name: str
    description: str
    tool_definition: Dict[str, Any]
    requires_confirmation: bool
    async def execute(self, context: ExecutionContext, **kwargs) -> Any
```

#### FunctionTool
Wraps Python functions as tools.

```python
class FunctionTool(BaseTool):
    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False
    )
```

#### @tool Decorator
Syntactic sugar for tool creation.

```python
@tool
def my_function(x: int) -> int:
    """Description for LLM."""
    return x * 2
```

#### Features:
- **Automatic Schema Generation**: From function type hints
- **Context-Aware Tools**: Optional ExecutionContext parameter
- **Tool Confirmation**: Per-tool confirmation requirements
- **Custom Tool Definitions**: Override auto-generated schemas

---

### 5. MCP Integration (`agent_framework/mcp.py`)

Integration with Model Context Protocol servers.

#### load_mcp_tools()
Discovers and loads tools from MCP servers.

```python
async def load_mcp_tools(connection: Dict) -> List[BaseTool]
```

#### Features:
- **MCP Tool Wrapping**: Converts MCP tools to FunctionTool
- **Stdio Client**: Connects to MCP servers via stdio
- **Schema Conversion**: MCP schemas to OpenAI format

#### Example:
```python
connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
}
tools = await load_mcp_tools(connection)
```

---

### 6. Memory Management (`agent_framework/memory.py`)

Token optimization and conversation history management.

#### Token Counting
Accurate token counting using tiktoken.

```python
def count_tokens(request: LlmRequest, model_id: str = "gpt-4") -> int
```

#### Sliding Window
Keeps only the most recent N messages.

```python
def apply_sliding_window(
    context: ExecutionContext,
    request: LlmRequest,
    window_size: int = 20
) -> None
```

#### Compaction
Replaces tool calls/results with compact references.

```python
def apply_compaction(context: ExecutionContext, request: LlmRequest) -> None
```

#### Summarization
LLM-based history compression.

```python
async def apply_summarization(
    context: ExecutionContext,
    request: LlmRequest,
    llm_client: LlmClient,
    keep_recent: int = 5
) -> None
```

#### ContextOptimizer
Hierarchical optimization strategy.

```python
class ContextOptimizer:
    def __init__(
        self,
        llm_client: LlmClient,
        token_threshold: int = 50000,
        enable_compaction: bool = True,
        enable_summarization: bool = True
    )
```

---

### 7. Callbacks (`agent_framework/callbacks.py`)

Extensibility hooks for agent execution.

#### create_optimizer_callback()
Factory for optimization callbacks.

```python
def create_optimizer_callback(
    apply_optimization: Callable,
    threshold: int = 50000,
    model_id: str = "gpt-4"
) -> Callable
```

#### Features:
- **Before LLM Callbacks**: Modify requests before API calls
- **After Tool Callbacks**: Process tool results
- **Async Support**: Both sync and async callbacks

---

### 8. Utilities (`agent_framework/utils.py`)

Helper functions for tool definitions and trace display.

#### Schema Generation
```python
function_to_input_schema(func) -> dict
format_tool_definition(name, description, parameters) -> dict
function_to_tool_definition(func) -> dict
```

#### Trace Display
```python
format_trace(context: ExecutionContext) -> str
display_trace(context: ExecutionContext) -> None
```

#### MCP Conversion
```python
mcp_tools_to_openai_format(mcp_tools) -> list[dict]
```

---

## Built-in Tools (`agent_tools/`)

### File Tools (`file_tools.py`)

#### read_file()
Reads text files (supports .txt, .csv, .json).

```python
@tool
def read_file(file_path: str) -> str
```

#### read_media_file()
Reads PDFs, Excel files, and images.

```python
@tool
def read_media_file(file_path: str) -> str
```

Supports:
- PDFs (via pymupdf)
- Excel files (via pandas/openpyxl)
- Images (via PIL)

#### list_files()
Lists directory contents.

```python
@tool
def list_files(directory_path: str) -> str
```

#### unzip_file()
Extracts zip archives.

```python
@tool
def unzip_file(zip_path: str, extract_to: str = None) -> str
```

---

### Web Tools (`web_tools.py`)

#### search_web()
Tavily API integration for web search.

```python
@tool
def search_web(query: str, max_results: int = 5) -> str
```

---

### Math Tools (`math_tools.py`)

#### calculator()
Safe eval-based calculator.

```python
@tool
def calculator(expression: str) -> str
```

---

## Web Application (`web_app/`)

### Backend (`app.py`)

FastAPI server providing RESTful API for agent interaction.

#### Endpoints:

- `GET /`: Serves the chat interface
- `POST /api/chat`: Send message to agent
- `POST /api/upload`: Upload files
- `GET /api/uploads`: List uploaded files
- `DELETE /api/uploads/{filename}`: Delete uploaded file
- `GET /api/tools`: List available tools
- `GET /api/sessions/{session_id}`: Get session info
- `DELETE /api/sessions/{session_id}`: Clear session

#### Features:
- **File Upload**: Handle user file uploads
- **Session Management**: API-based session handling
- **Tool Listing**: Expose available tools via API
- **Trace Display**: Return formatted execution traces
- **CORS Support**: Cross-origin requests enabled

---

### Frontend (`static/index.html`)

Modern chat interface with full framework integration.

#### Features:
- **Chat Interface**: Real-time conversation UI
- **File Upload UI**: Drag-and-drop file uploads
- **Tool List Display**: Show available tools in sidebar
- **Session Toggle**: Enable/disable session persistence
- **Trace Modal**: View execution traces in formatted view
- **Session ID Display**: Show current session ID
- **Clear Session**: Reset conversation state
- **Responsive Design**: Works on desktop and mobile

---

## Additional Features

### GAIA Evaluation (`gaia/`)

Benchmark integration for evaluating agent performance.

- **Problem Loading**: Load GAIA benchmark problems
- **File Handling**: Download and extract attached files
- **Evaluation**: Run agent on benchmark problems
- **Results**: Track accuracy and solvability

### RAG Examples (`rag/`)

Examples of retrieval-augmented generation.

- **Chunking**: Text chunking strategies
- **Embeddings**: Vector embeddings using OpenAI
- **Vector Search**: Cosine similarity search
- **Integration**: Using RAG with agents

### Example Scripts

- **example_agent.py**: Basic agent usage example
- **test_session.py**: Session persistence demonstration

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User/Application                      │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                      Agent.run()                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Think     │→ │     Act      │→ │   Observe    │  │
│  │  (LLM Call)  │  │ (Tool Exec)  │  │ (Process)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────┬────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  LlmClient  │ │   Tools     │ │  Execution  │
│             │ │             │ │   Context   │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   LiteLLM   │ │  MCP Tools  │ │   Session   │
│             │ │             │ │   Manager   │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

## Usage Examples

### Basic Agent

```python
from agent_framework import Agent, LlmClient
from agent_tools import calculator, search_web

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator, search_web],
    instructions="You are a helpful assistant.",
    max_steps=10
)

result = await agent.run("What is 123 * 456?")
print(result.output)
```

### Agent with Session

```python
from agent_framework import Agent, LlmClient, InMemorySessionManager

session_manager = InMemorySessionManager()

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator],
    session_manager=session_manager
)

# First conversation
result1 = await agent.run("My name is Alice", session_id="user-123")

# Second conversation (remembers context)
result2 = await agent.run("What's my name?", session_id="user-123")
```

### Agent with Structured Output

```python
from pydantic import BaseModel
from typing import Literal

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[],
    instructions="Analyze sentiment.",
    output_type=SentimentAnalysis
)

result = await agent.run("I love this product!")
print(result.output.sentiment)  # "positive"
```

### Agent with Memory Optimization

```python
from agent_framework import Agent, LlmClient, create_optimizer_callback
from agent_framework.memory import apply_sliding_window

optimizer = create_optimizer_callback(
    apply_optimization=apply_sliding_window,
    threshold=30000
)

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator],
    before_llm_callback=optimizer
)
```

---

## Design Decisions

### Why Pydantic for Models?
- Runtime validation catches errors early
- Automatic serialization/deserialization
- Type safety with IDE support

### Why Dataclass for ExecutionContext?
- Mutable state needs to be lightweight
- No validation needed (internal use)
- Better performance for frequent updates

### Why LiteLLM?
- Multi-provider support (OpenAI, Anthropic, local)
- Unified API interface
- Easy to switch models

### Why MCP?
- Standard protocol for tool discovery
- Decouples tool servers from agents
- Easy integration of external tools

---

## Future Enhancements

Potential areas for expansion:

1. **Database Session Manager**: Persistent storage for sessions
2. **Streaming Responses**: Real-time token streaming
3. **Multi-Agent Coordination**: Agents working together
4. **Tool Marketplace**: Discover and share tools
5. **Advanced Memory**: Vector-based memory retrieval
6. **Cost Tracking**: Monitor API usage and costs
7. **Rate Limiting**: Built-in rate limiting
8. **Monitoring**: Observability and logging

---

## Conclusion

This framework provides a complete foundation for building production-ready AI agents. It balances flexibility with structure, allowing you to build simple chatbots or complex multi-agent systems.

For tutorials and examples, see the `misc/tutorials/` directory.

