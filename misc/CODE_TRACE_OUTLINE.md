# Code Trace Outline - Agent Framework

This document provides a systematic guide to understanding and tracing through the agent framework codebase.

## 📋 Table of Contents
1. [Entry Points](#entry-points)
2. [Suggested Reading Order](#suggested-reading-order)
3. [Function Call Flow](#function-call-flow)
4. [Component Relationships](#component-relationships)
5. [Key Functions Reference](#key-functions-reference)

---

## 🚀 Entry Points

### Primary Entry Point
**File:** `examples/demo.py` or `example.py` or `examples/gaia_evaluation.py`

```python
agent = Agent(...)
result = await agent.run("question")
```

**Flow:** `agent.run()` → `agent.step()` → `agent.think()` → `agent.act()`

---

## 📖 Suggested Reading Order

### Phase 1: Foundation (Data Models)
**Start here to understand the data structures**

1. **`agent_framework/models.py`**
   - Read in order:
     - `Message` (line ~10-15)
     - `ToolCall` (line ~18-25)
     - `ToolResult` (line ~28-35)
     - `ContentItem` (line ~38-45)
     - `Event` (line ~48-60)
     - `ExecutionContext` (line ~63-85)
   
   **Why:** These are the core data structures used throughout the framework.

### Phase 2: Tool System
**Understand how tools are defined and executed**

2. **`agent_framework/tools.py`**
   - Read in order:
     - `BaseTool` abstract class (line ~10-20)
     - `FunctionTool.__init__()` (line ~25-45)
     - `FunctionTool.execute()` (line ~47-70)
     - `FunctionTool.to_definition()` (line ~72-85)
     - `@tool` decorator (line ~88-120)
   
   **Why:** Tools are the extension mechanism - understand how functions become tools.

3. **`agent_framework/utils.py`**
   - Read in order:
     - `function_to_input_schema()` (line ~10-50)
     - `format_tool_definition()` (line ~52-75)
     - `mcp_tools_to_openai_format()` (line ~77-100)
   
   **Why:** These convert Python functions to LLM-understandable tool definitions.

### Phase 3: LLM Integration
**Understand how we communicate with LLMs**

4. **`agent_framework/llm.py`**
   - Read in order:
     - `LlmRequest` (line ~10-25)
     - `LlmResponse` (line ~28-40)
     - `LlmClient.__init__()` (line ~43-50)
     - `LlmClient._build_messages()` (line ~52-85)
     - `LlmClient._parse_response()` (line ~87-120)
     - `LlmClient.generate()` (line ~122-146)
   
   **Why:** This is the interface to the LLM - understand request/response format.

### Phase 4: Agent Core Logic
**The main orchestration logic**

5. **`agent_framework/agent.py`**
   - Read in order:
     - `Agent.__init__()` (line ~29-43)
     - `Agent._setup_tools()` (line ~45-46)
     - `Agent._prepare_llm_request()` (line ~48-85)
     - `Agent.think()` (line ~87-95)
     - `Agent.act()` (line ~97-115)
     - `Agent.step()` (line ~116-171)
     - `Agent.run()` (line ~173-200)
     - `Agent._is_final_response()` (line ~202-206)
     - `Agent._extract_final_result()` (line ~208-225)
   
   **Why:** This is the main agent loop - understand the step-by-step execution.

### Phase 5: MCP Integration (Optional)
**External tool loading**

6. **`agent_framework/mcp.py`**
   - Read in order:
     - `_extract_text_content()` (line ~10-20)
     - `_create_mcp_tool()` (line ~22-60)
     - `load_mcp_tools()` (line ~62-100)
   
   **Why:** Understand how external MCP servers provide tools.

### Phase 6: Examples
**See it all in action**

7. **`examples/demo.py`** - Simple example
8. **`examples/gaia_evaluation.py`** - Complex example with structured output

---

## 🔄 Function Call Flow

### Main Execution Flow

```
User Code
  │
  ├─> Agent.run(user_input)
  │     │
  │     ├─> Creates ExecutionContext
  │     ├─> Adds user Event
  │     │
  │     └─> Loop: while not final_result and step < max_steps
  │           │
  │           └─> Agent.step(context)
  │                 │
  │                 ├─> Agent._prepare_llm_request(context)
  │                 │     │
  │                 │     ├─> Flattens events → contents
  │                 │     ├─> Adds tool info to instructions
  │                 │     └─> Returns LlmRequest
  │                 │
  │                 ├─> Agent.think(llm_request)
  │                 │     │
  │                 │     └─> LlmClient.generate(request)
  │                 │           │
  │                 │           ├─> LlmClient._build_messages()
  │                 │           ├─> litellm.acompletion()
  │                 │           └─> LlmClient._parse_response()
  │                 │
  │                 ├─> If tool_calls exist:
  │                 │     │
  │                 │     └─> Agent.act(context, tool_calls)
  │                 │           │
  │                 │           ├─> For each tool_call:
  │                 │           │     │
  │                 │           │     └─> tool.execute(context, **args)
  │                 │           │           │
  │                 │           │           └─> FunctionTool.execute()
  │                 │           │                 │
  │                 │           │                 └─> Calls wrapped function
  │                 │           │
  │                 │           └─> Returns ToolResult[]
  │                 │
  │                 └─> If output_type and no tool_calls:
  │                       │
  │                       └─> Final LLM call with structured output
  │
  └─> Returns AgentResult(output, context)
```

### Tool Execution Flow

```
Agent.act(context, tool_calls)
  │
  ├─> For each ToolCall:
  │     │
  │     ├─> Find tool by name: tool = self.tools[call.name]
  │     │
  │     ├─> Call: tool.execute(context, **call.arguments)
  │     │     │
  │     │     └─> FunctionTool.execute()
  │     │           │
  │     │           ├─> Inspect function signature
  │     │           ├─> If has 'context' param → pass it
  │     │           ├─> Call: func(**kwargs)
  │     │           └─> Wrap result in ToolResult
  │     │
  │     └─> Collect ToolResult
  │
  └─> Return list of ToolResults
```

### LLM Request Building Flow

```
Agent._prepare_llm_request(context)
  │
  ├─> Flatten events → contents
  │     │
  │     └─> For each Event:
  │           └─> For each ContentItem in event.content:
  │                 └─> Add to flat_contents
  │
  ├─> Build instructions
  │     │
  │     ├─> Add self.instructions
  │     └─> If tools exist:
  │           └─> Append tool descriptions
  │
  ├─> Set response_format (if enforce_output_type)
  │
  └─> Return LlmRequest(instructions, contents, tools, response_format)
```

---

## 🔗 Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                             │
│  (examples/demo.py, example.py, gaia_evaluation.py)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Creates
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                         Agent                                │
│  (agent_framework/agent.py)                                  │
│                                                              │
│  - Orchestrates reasoning loop                              │
│  - Manages ExecutionContext                                 │
│  - Calls LLM and executes tools                             │
└──────┬───────────────────────────────┬───────────────────────┘
       │                               │
       │ Uses                          │ Uses
       ▼                               ▼
┌──────────────────┐         ┌──────────────────┐
│   LlmClient      │         │   BaseTool       │
│  (llm.py)        │         │   (tools.py)     │
│                  │         │                  │
│  - Sends requests│         │  - FunctionTool  │
│  - Parses resp.  │         │  - MCP Tools     │
└──────┬───────────┘         └──────┬───────────┘
       │                            │
       │ Uses                       │ Uses
       ▼                            ▼
┌──────────────────┐         ┌──────────────────┐
│   LiteLLM        │         │   Utils          │
│  (external)      │         │  (utils.py)      │
│                  │         │                  │
│  - API calls     │         │  - Schema conv.  │
│  - Streaming     │         │  - Format tools  │
└──────────────────┘         └──────────────────┘
                                     │
                                     │ Uses
                                     ▼
                            ┌──────────────────┐
                            │   MCP Client     │
                            │  (mcp.py)        │
                            │                  │
                            │  - Load tools    │
                            │  - Execute tools │
                            └──────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Data Models                              │
│  (models.py)                                                │
│                                                             │
│  - Message, ToolCall, ToolResult                           │
│  - Event, ExecutionContext                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Key Functions Reference

### Agent Class (`agent.py`)

| Function | Purpose | Called By | Calls |
|----------|---------|-----------|-------|
| `__init__()` | Initialize agent with model, tools, instructions | User code | `_setup_tools()` |
| `run()` | Main entry point - execute agent loop | User code | `step()` |
| `step()` | Execute one reasoning step | `run()` | `_prepare_llm_request()`, `think()`, `act()` |
| `think()` | Get LLM's decision | `step()` | `LlmClient.generate()` |
| `act()` | Execute tool calls | `step()` | `tool.execute()` |
| `_prepare_llm_request()` | Build LLM request from context | `step()` | - |
| `_is_final_response()` | Check if response is final | `run()` | - |
| `_extract_final_result()` | Extract structured output | `run()` | - |

### LlmClient Class (`llm.py`)

| Function | Purpose | Called By | Calls |
|----------|---------|-----------|-------|
| `generate()` | Make async LLM API call | `Agent.think()` | `_build_messages()`, `_parse_response()`, `litellm.acompletion()` |
| `_build_messages()` | Convert request to OpenAI format | `generate()` | - |
| `_parse_response()` | Parse LLM response | `generate()` | - |

### FunctionTool Class (`tools.py`)

| Function | Purpose | Called By | Calls |
|----------|---------|-----------|-------|
| `execute()` | Execute the wrapped function | `Agent.act()` | Wrapped function |
| `to_definition()` | Convert to OpenAI tool format | `Agent._prepare_llm_request()` | `format_tool_definition()` |

### Utility Functions (`utils.py`)

| Function | Purpose | Called By |
|----------|---------|-----------|
| `function_to_input_schema()` | Convert function to JSON Schema | `FunctionTool.to_definition()` |
| `format_tool_definition()` | Format tool in OpenAI format | `FunctionTool.to_definition()` |
| `display_trace()` | Pretty print execution trace | User code |

---

## 🎯 Tracing a Specific Execution

### Example: "What is 1234 * 5678?"

1. **User calls:** `agent.run("What is 1234 * 5678?")`
   - Location: `agent.py:173`

2. **Agent.run() creates context:**
   - Creates `ExecutionContext`
   - Adds user `Event` with `Message(role="user", content="...")`
   - Location: `agent.py:180-189`

3. **First iteration of loop:**
   - Calls `agent.step(context)`
   - Location: `agent.py:193`

4. **Agent.step() prepares request:**
   - Calls `_prepare_llm_request(context)`
   - Flattens events, adds tool info
   - Location: `agent.py:116-120`

5. **Agent.think() calls LLM:**
   - Calls `llm_client.generate(request)`
   - Location: `agent.py:87-95`

6. **LlmClient.generate() processes:**
   - `_build_messages()` converts to OpenAI format
   - `litellm.acompletion()` makes API call
   - `_parse_response()` parses response
   - Location: `llm.py:122-146`

7. **LLM returns with tool call:**
   - Response contains `ToolCall(name="calculator", arguments={"expression": "1234 * 5678"})`
   - Location: `llm.py:100-120` (parsing)

8. **Agent.act() executes tool:**
   - Finds `calculator` tool
   - Calls `tool.execute(context, expression="1234 * 5678")`
   - Location: `agent.py:97-115`

9. **FunctionTool.execute() runs function:**
   - Calls `calculator("1234 * 5678")`
   - Returns `7006652`
   - Wraps in `ToolResult`
   - Location: `tools.py:47-70`

10. **Tool result added to context:**
    - Creates new `Event` with `ToolResult`
    - Location: `agent.py:155-161`

11. **Second iteration:**
    - LLM sees tool result
    - Returns final answer: `"7006652"`
    - Location: `agent.py:116-171`

12. **Agent.run() detects final response:**
    - `_is_final_response()` returns True
    - `_extract_final_result()` extracts answer
    - Returns `AgentResult`
    - Location: `agent.py:196-200`

---

## 🔍 Debugging Tips

1. **Start with `Agent.run()`** - This is where execution begins
2. **Follow `step()` calls** - Each step is one reasoning iteration
3. **Check `ExecutionContext.events`** - This contains the full history
4. **Use `display_trace()`** - Visualize the execution flow
5. **Inspect `LlmRequest`** - See what's sent to the LLM
6. **Check `LlmResponse`** - See what the LLM returned
7. **Verify tool definitions** - Ensure tools are properly formatted

---

## 📝 Notes

- **Async/Await:** Most functions are async - use `await` when calling
- **ExecutionContext:** Threaded through all operations - contains state
- **Events:** Every action (user input, LLM response, tool call) is an Event
- **Tool Execution:** Tools can optionally receive `ExecutionContext` parameter
- **Structured Output:** Only enforced on final LLM call (not during tool usage)

