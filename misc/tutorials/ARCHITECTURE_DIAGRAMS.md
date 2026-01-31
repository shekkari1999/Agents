# Architecture Diagrams

This document contains visual diagrams of the agent framework architecture using Mermaid syntax.

---

## System Overview

```mermaid
graph TB
    User[User/Application] --> Agent[Agent.run]
    Agent --> Think[Think: LLM Call]
    Agent --> Act[Act: Tool Execution]
    Agent --> Observe[Observe: Process Results]
    
    Think --> LlmClient[LlmClient]
    Act --> Tools[Tools]
    Observe --> Context[ExecutionContext]
    
    LlmClient --> LiteLLM[LiteLLM]
    Tools --> FunctionTools[FunctionTools]
    Tools --> MCPTools[MCP Tools]
    Context --> Session[Session Manager]
    
    LiteLLM --> OpenAI[OpenAI API]
    LiteLLM --> Anthropic[Anthropic API]
    LiteLLM --> Local[Local Models]
    
    Session --> Memory[Memory Storage]
```

---

## Agent Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Context
    participant LLM
    participant Tools
    
    User->>Agent: run(user_input)
    Agent->>Context: Create/load ExecutionContext
    Agent->>Context: Add user event
    
    loop Until completion or max_steps
        Agent->>Agent: step(context)
        Agent->>LLM: think(llm_request)
        LLM-->>Agent: LlmResponse (with tool_calls)
        Agent->>Context: Add response event
        
        alt Tool calls present
            Agent->>Tools: act(context, tool_calls)
            Tools-->>Agent: ToolResults
            Agent->>Context: Add tool results event
        end
        
        Agent->>Agent: Check completion
    end
    
    Agent-->>User: AgentResult
```

---

## Data Model Relationships

```mermaid
classDiagram
    class ExecutionContext {
        +str execution_id
        +List[Event] events
        +int current_step
        +Dict state
        +str final_result
        +str session_id
        +add_event()
        +increment_step()
    }
    
    class Event {
        +str id
        +str execution_id
        +float timestamp
        +str author
        +List[ContentItem] content
    }
    
    class Message {
        +str type
        +str role
        +str content
    }
    
    class ToolCall {
        +str type
        +str tool_call_id
        +str name
        +dict arguments
    }
    
    class ToolResult {
        +str type
        +str tool_call_id
        +str name
        +str status
        +list content
    }
    
    class Session {
        +str session_id
        +str user_id
        +List[Event] events
        +dict state
        +datetime created_at
        +datetime updated_at
    }
    
    ExecutionContext "1" *-- "many" Event
    Event "1" *-- "many" ContentItem
    ContentItem <|-- Message
    ContentItem <|-- ToolCall
    ContentItem <|-- ToolResult
    Session "1" o-- "many" Event
```

---

## Tool System Architecture

```mermaid
graph LR
    PythonFunction[Python Function] --> FunctionTool[FunctionTool]
    FunctionTool --> BaseTool[BaseTool]
    BaseTool --> ToolDefinition[Tool Definition JSON Schema]
    
    MCPServer[MCP Server] --> MCPTool[MCP Tool]
    MCPTool --> FunctionTool
    
    FunctionTool --> Agent[Agent]
    Agent --> LLM[LLM API]
    
    LLM --> ToolCall[ToolCall]
    ToolCall --> Agent
    Agent --> FunctionTool
    FunctionTool --> ToolResult[ToolResult]
    ToolResult --> Agent
```

---

## Memory Optimization Flow

```mermaid
flowchart TD
    Start[LLM Request Created] --> CountTokens[Count Tokens]
    CountTokens --> CheckThreshold{Tokens < Threshold?}
    
    CheckThreshold -->|Yes| NoOptimization[No Optimization Needed]
    CheckThreshold -->|No| ApplyCompaction[Apply Compaction]
    
    ApplyCompaction --> CountAgain[Count Tokens Again]
    CountAgain --> CheckAgain{Tokens < Threshold?}
    
    CheckAgain -->|Yes| Done[Done]
    CheckAgain -->|No| ApplySummarization[Apply Summarization]
    
    ApplySummarization --> Done
    NoOptimization --> Done
```

---

## Session Management Flow

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant SessionManager
    participant Storage
    
    User->>Agent: run(input, session_id)
    Agent->>SessionManager: get_or_create(session_id)
    
    alt Session exists
        SessionManager->>Storage: get(session_id)
        Storage-->>SessionManager: Session
        SessionManager->>Agent: Load session
        Agent->>Agent: Restore events/state
    else New session
        SessionManager->>Storage: create(session_id)
        Storage-->>SessionManager: New Session
        SessionManager->>Agent: New session
    end
    
    Agent->>Agent: Execute agent loop
    Agent->>SessionManager: save(session)
    SessionManager->>Storage: Persist session
    Agent-->>User: Result
```

---

## MCP Integration Flow

```mermaid
sequenceDiagram
    participant Agent
    participant MCPClient[MCP Client]
    participant MCPServer[MCP Server]
    participant FunctionTool
    
    Agent->>MCPClient: load_mcp_tools(connection)
    MCPClient->>MCPServer: Connect via stdio
    MCPServer-->>MCPClient: Connection established
    
    MCPClient->>MCPServer: list_tools()
    MCPServer-->>MCPClient: Tool definitions
    
    loop For each MCP tool
        MCPClient->>FunctionTool: _create_mcp_tool()
        FunctionTool-->>MCPClient: Wrapped tool
    end
    
    MCPClient-->>Agent: List of FunctionTools
    
    Note over Agent,FunctionTool: Agent can now use MCP tools
    
    Agent->>FunctionTool: Execute tool
    FunctionTool->>MCPServer: call_tool(name, args)
    MCPServer-->>FunctionTool: Result
    FunctionTool-->>Agent: ToolResult
```

---

## Web Application Architecture

```mermaid
graph TB
    Browser[Browser] --> Frontend[HTML/CSS/JS]
    Frontend --> API[FastAPI Backend]
    
    API --> ChatEndpoint[/api/chat]
    API --> UploadEndpoint[/api/upload]
    API --> ToolsEndpoint[/api/tools]
    API --> SessionsEndpoint[/api/sessions]
    
    ChatEndpoint --> Agent[Agent Framework]
    UploadEndpoint --> FileStorage[File Storage]
    ToolsEndpoint --> ToolRegistry[Tool Registry]
    SessionsEndpoint --> SessionManager[Session Manager]
    
    Agent --> LlmClient[LlmClient]
    Agent --> Tools[Tools]
    Agent --> Memory[Memory Manager]
    
    LlmClient --> LiteLLM[LiteLLM]
    Tools --> FunctionTools[Function Tools]
    Memory --> SessionManager
```

---

## Request/Response Flow

```mermaid
sequenceDiagram
    participant Frontend
    participant API
    participant Agent
    participant LLM
    participant Tools
    
    Frontend->>API: POST /api/chat {message, session_id}
    API->>Agent: run(message, session_id)
    
    Agent->>Agent: Load session
    Agent->>Agent: step(context)
    Agent->>LLM: generate(request)
    LLM-->>Agent: Response with tool_calls
    
    alt Tool calls present
        Agent->>Tools: act(context, tool_calls)
        Tools-->>Agent: ToolResults
        Agent->>Agent: step(context) [continue]
    end
    
    Agent->>Agent: Save session
    Agent-->>API: AgentResult
    API->>API: Format response
    API-->>Frontend: {response, trace, tools_used}
    Frontend->>Frontend: Display message
```

---

## Tool Execution Details

```mermaid
flowchart TD
    Start[ToolCall Received] --> FindTool[Find Tool by Name]
    FindTool --> ToolExists{Tool Found?}
    
    ToolExists -->|No| Error[Raise ValueError]
    ToolExists -->|Yes| CheckConfirmation{Requires Confirmation?}
    
    CheckConfirmation -->|Yes| Pending[Create PendingToolCall]
    CheckConfirmation -->|No| Execute[Execute Tool]
    
    Execute --> ToolSuccess{Success?}
    ToolSuccess -->|Yes| SuccessResult[ToolResult: success]
    ToolSuccess -->|No| ErrorResult[ToolResult: error]
    
    Pending --> WaitUser[Wait for User Confirmation]
    WaitUser --> UserApproved{Approved?}
    UserApproved -->|Yes| Execute
    UserApproved -->|No| RejectedResult[ToolResult: rejected]
    
    SuccessResult --> Return[Return ToolResult]
    ErrorResult --> Return
    RejectedResult --> Return
    Error --> Return
```

---

## Memory Optimization Strategies

```mermaid
graph TB
    Request[LlmRequest] --> Count[Count Tokens]
    Count --> Threshold{> Threshold?}
    
    Threshold -->|No| Skip[Skip Optimization]
    Threshold -->|Yes| Strategy{Choose Strategy}
    
    Strategy --> SlidingWindow[Sliding Window<br/>Keep Recent N]
    Strategy --> Compaction[Compaction<br/>Replace with References]
    Strategy --> Summarization[Summarization<br/>LLM Compression]
    
    SlidingWindow --> Reduced1[Reduced Tokens]
    Compaction --> Reduced2[Reduced Tokens]
    Summarization --> Reduced3[Reduced Tokens]
    
    Reduced1 --> Final[Optimized Request]
    Reduced2 --> Final
    Reduced3 --> Final
    Skip --> Final
```

---

## Component Dependencies

```mermaid
graph TD
    Agent --> LlmClient
    Agent --> Tools
    Agent --> Models
    Agent --> Memory
    Agent --> Callbacks
    
    LlmClient --> Models
    LlmClient --> LiteLLM
    
    Tools --> Models
    Tools --> Utils
    
    Memory --> Models
    Memory --> LlmClient
    
    Callbacks --> Memory
    Callbacks --> Models
    
    MCP --> Tools
    MCP --> Models
    
    WebApp --> Agent
    WebApp --> Models
    
    Models --> Pydantic
    Tools --> ABC
```

---

## State Management

```mermaid
stateDiagram-v2
    [*] --> Initialized: Agent created
    Initialized --> Running: run() called
    Running --> Thinking: step() called
    Thinking --> Acting: Tool calls received
    Acting --> Thinking: Tool results processed
    Thinking --> Completed: Final response
    Acting --> Completed: Final response
    Completed --> [*]
    
    Running --> Pending: Tool confirmation required
    Pending --> Running: Confirmation received
```

---

## Error Handling Flow

```mermaid
flowchart TD
    Start[Operation] --> Try{Try}
    Try -->|Success| Success[Return Result]
    Try -->|Error| Catch[Catch Exception]
    
    Catch --> ErrorType{Error Type?}
    ErrorType -->|LLM Error| LLMError[Return LlmResponse with error_message]
    ErrorType -->|Tool Error| ToolError[Return ToolResult with error status]
    ErrorType -->|Network Error| NetworkError[Retry or Return Error]
    ErrorType -->|Validation Error| ValidationError[Return Validation Error]
    
    LLMError --> Log[Log Error]
    ToolError --> Log
    NetworkError --> Log
    ValidationError --> Log
    
    Log --> ReturnError[Return Error to User]
    Success --> [*]
    ReturnError --> [*]
```

---

## Usage Examples

These diagrams can be:
1. Included in video thumbnails
2. Shown during explanations
3. Added to documentation
4. Used in presentations
5. Embedded in blog posts

To render these diagrams:
- Use Mermaid Live Editor: https://mermaid.live/
- Use GitHub (renders automatically in .md files)
- Use VS Code with Mermaid extension
- Use documentation tools like MkDocs

