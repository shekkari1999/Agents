# Episode 8: MCP Integration

**Duration**: 30 minutes  
**What to Build**: `agent_framework/mcp.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Loading tools from MCP servers
- Tavily search integration
- External tool discovery

**Hook Statement**: "Today we'll integrate with MCP - a protocol that lets us discover and use tools from external servers. This opens up a whole ecosystem of tools!"

---

### 2. Problem (3 min)
**Why do we need MCP?**

**The Challenge:**
- Tools live on separate servers
- Need standard protocol
- Want tool discovery
- Don't want to hardcode tools

**The Solution:**
- Model Context Protocol (MCP)
- Stdio communication
- Tool discovery
- Automatic wrapping

---

### 3. Concept: What is MCP? (5 min)

**MCP (Model Context Protocol):**
- Standard protocol for tool servers
- JSON-RPC over stdio
- Tool discovery
- Tool execution

**Benefits:**
- Decouples tools from agents
- Standard interface
- Easy integration
- Tool marketplace potential

**Flow:**
1. Connect to MCP server
2. Discover available tools
3. Wrap as FunctionTool
4. Use in agent

---

### 4. Live Coding: Building MCP Integration (20 min)

#### Step 1: Setup and Imports (2 min)
```python
import os
from typing import Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools import BaseTool, FunctionTool
```

**Dependencies:**
- `mcp` package
- Stdio client for communication

**Live Coding**: Setup imports

---

#### Step 2: Helper Function - Extract Text (2 min)
```python
def _extract_text_content(result) -> str:
    """Extract text content from MCP tool result."""
    if not hasattr(result, 'content'):
        return str(result)
    
    texts = []
    for item in result.content:
        if hasattr(item, 'text'):
            texts.append(item.text)
        else:
            texts.append(str(item))
    
    return "\n\n".join(texts)
```

**Purpose**: MCP returns structured content, we need text.

**Live Coding**: Build extract function

---

#### Step 3: load_mcp_tools() Function (6 min)
```python
async def load_mcp_tools(connection: Dict) -> List[BaseTool]:
    """Load tools from an MCP server and convert to FunctionTools.
    
    Args:
        connection: Dictionary with connection parameters:
            - command: Command to run the MCP server
            - args: Arguments for the command
            - env: Environment variables (optional)
    
    Returns:
        List of BaseTool instances wrapping MCP tools
    """
    tools = []
    
    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            
            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)
    
    return tools
```

**Key Points:**
- Connects via stdio
- Initializes session
- Lists tools
- Wraps each tool

**Live Coding**: Build load_mcp_tools()

---

#### Step 4: _create_mcp_tool() Function (8 min)
```python
def _create_mcp_tool(mcp_tool, connection: Dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""
    
    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)
    
    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        }
    }
    
    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition
    )
```

**Key Points:**
- Creates wrapper function
- Connects on each call
- Uses MCP schema
- Returns FunctionTool

**Live Coding**: Build _create_mcp_tool()

---

#### Step 5: Testing MCP Integration (2 min)
```python
import os
from agent_framework.mcp import load_mcp_tools

connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
}

tools = await load_mcp_tools(connection)
print(f"Loaded {len(tools)} tools")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")
```

**Expected Output:**
```
Loaded 5 tools
  - tavily_search: Search the web...
  - tavily_extract: Extract content from URLs...
  ...
```

---

### 5. Demo: Using MCP Tools (3 min)

**Show:**
- Loading Tavily tools
- Using in agent
- Web search working
- Tool discovery

---

### 6. MCP Server Setup (2 min)

**Tavily MCP Server:**
```bash
# Install via npx
npx -y tavily-mcp@latest

# Or set up locally
npm install -g tavily-mcp
```

**Other MCP Servers:**
- File system tools
- Database tools
- API tools
- Custom servers

---

### 7. Next Steps (1 min)

**Preview Episode 9:**
- Session management
- Memory optimization
- Token management

**What We Built:**
- MCP integration
- Tool discovery
- External tool support

---

## Key Takeaways

1. **MCP** provides standard tool protocol
2. **Stdio communication** for tool servers
3. **Tool discovery** finds available tools
4. **Automatic wrapping** converts to FunctionTool
5. **Decoupled** tools from agents

---

## Common Mistakes

**Mistake 1: Not handling connection errors**
```python
# Wrong - crashes if server unavailable
tools = await load_mcp_tools(connection)

# Right - handle gracefully
try:
    tools = await load_mcp_tools(connection)
except Exception as e:
    print(f"Failed to load MCP tools: {e}")
    tools = []
```

**Mistake 2: Forgetting environment variables**
```python
# Wrong - missing API key
connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"]
}

# Right - include env
connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
}
```

---

## Exercises

1. Add connection pooling
2. Implement tool caching
3. Add MCP server health checks
4. Create custom MCP server

---

**Previous Episode**: [Episode 7: Tool Execution](./EPISODE_07_TOOL_EXECUTION.md)  
**Next Episode**: [Episode 9: Session & Memory Management](./EPISODE_09_MEMORY.md)

