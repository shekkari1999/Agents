# Episode 7: Tool Execution & Complete Agent

**Duration**: 35 minutes  
**What to Build**: Complete `agent_framework/agent.py` with tool execution  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Complete agent with tool execution
- Calculator and web search working
- Multi-step problem solving

**Hook Statement**: "Today we'll complete the agent by adding tool execution. This is where everything comes together - the agent can now actually do things!"

---

### 2. Problem (3 min)
**Why do we need tool execution?**

**The Challenge:**
- LLM decides to use tools
- Need to execute them
- Handle results
- Continue reasoning

**The Solution:**
- Act() method
- Tool dispatch
- Result handling
- Error management

---

### 3. Concept: Tool Execution Flow (5 min)

**The Flow:**
1. LLM returns tool calls
2. Agent finds tools by name
3. Executes each tool
4. Records results
5. Continues reasoning

**Key Steps:**
- Parse tool calls from response
- Map to tool instances
- Execute with arguments
- Handle errors
- Create ToolResult objects

---

### 4. Live Coding: Completing the Agent (25 min)

#### Step 1: Update step() to Handle Tools (3 min)
```python
async def step(self, context: ExecutionContext):
    """Execute one step of the agent loop."""
    
    llm_request = self._prepare_llm_request(context)
    llm_response = await self.think(llm_request)
    
    # Record LLM response
    response_event = Event(
        execution_id=context.execution_id,
        author=self.name,
        content=llm_response.content,
    )
    context.add_event(response_event)
    
    # Execute tools if the LLM requested any
    tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
    if tool_calls:
        tool_results = await self.act(context, tool_calls)
        tool_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=tool_results,
        )
        context.add_event(tool_event)
    
    context.increment_step()
```

**Key Changes:**
- Extract tool calls
- Call act() if needed
- Record tool results

**Live Coding**: Update step()

---

#### Step 2: Update _prepare_llm_request() (2 min)
```python
def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
    """Convert execution context to LLM request."""
    flat_contents = []
    for event in context.events:
        flat_contents.extend(event.content)
    
    # Determine tool choice strategy
    tool_choice = "auto" if self.tools else None
    
    return LlmRequest(
        instructions=[self.instructions] if self.instructions else [],
        contents=flat_contents,
        tools=self.tools,  # Now includes tools!
        tool_choice=tool_choice
    )
```

**Key Changes:**
- Include tools in request
- Set tool_choice

**Live Coding**: Update _prepare_llm_request()

---

#### Step 3: act() Method - Tool Execution (10 min)
```python
from .models import ToolResult

async def act(
    self, 
    context: ExecutionContext, 
    tool_calls: List[ToolCall]
) -> List[ToolResult]:
    """Execute tool calls and return results."""
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []

    for tool_call in tool_calls:
        if tool_call.name not in tools_dict:
            raise ValueError(f"Tool '{tool_call.name}' not found")
        
        tool = tools_dict[tool_call.name]
        
        # Execute tool
        try:
            tool_response = await tool(context, **tool_call.arguments)
            status = "success"
        except Exception as e:
            tool_response = str(e)
            status = "error"
        
        tool_result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            status=status,
            content=[tool_response],
        )
        
        results.append(tool_result)
    
    return results
```

**Key Points:**
- Map tool names to instances
- Execute each tool
- Handle errors
- Create ToolResult objects

**Live Coding**: Build act()

---

#### Step 4: Update Completion Detection (3 min)
```python
def _is_final_response(self, event: Event) -> bool:
    """Check if this event contains a final response."""
    has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
    has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
    # Final if no tool activity
    return not has_tool_calls and not has_tool_results

def _extract_final_result(self, event: Event) -> str:
    """Extract final result from event."""
    for item in event.content:
        if isinstance(item, Message) and item.role == "assistant":
            return item.content
    return None
```

**Key Points:**
- Check for tool activity
- Extract message if final

**Live Coding**: Update completion detection

---

#### Step 5: Testing Complete Agent (4 min)
```python
from agent_framework import Agent, LlmClient
from agent_tools import calculator

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator],
    instructions="Use tools when needed.",
    max_steps=10
)

result = await agent.run("What is 123 * 456?")
print(result.output)  # Should use calculator
print(f"Steps: {result.context.current_step}")
```

**Multi-Step Example:**
```python
from agent_tools import calculator, search_web

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator, search_web],
    instructions="Use tools to answer accurately.",
    max_steps=10
)

result = await agent.run(
    "What's the current temperature in New York, and what's that in Celsius?"
)
# Should: search web, then calculate conversion
```

---

### 5. Demo: Complete Agent (3 min)

**Show:**
- Calculator tool working
- Web search tool working
- Multi-step reasoning
- Execution trace

---

### 6. Error Handling (2 min)

**Tool Not Found:**
```python
if tool_call.name not in tools_dict:
    raise ValueError(f"Tool '{tool_call.name}' not found")
```

**Tool Execution Error:**
```python
try:
    tool_response = await tool(context, **tool_call.arguments)
    status = "success"
except Exception as e:
    tool_response = str(e)
    status = "error"
```

**Best Practices:**
- Always handle errors
- Return error in ToolResult
- Let agent continue reasoning
- Log errors for debugging

---

### 7. Next Steps (1 min)

**Preview Episode 8:**
- MCP integration
- External tool servers
- Tool discovery

**What We Built:**
- Complete agent with tools
- Tool execution
- Error handling

---

## Key Takeaways

1. **act()** executes tools from LLM decisions
2. **Tool dispatch** maps names to instances
3. **Error handling** is crucial
4. **ToolResult** records outcomes
5. **Multi-step** reasoning enabled

---

## Common Mistakes

**Mistake 1: Not handling tool errors**
```python
# Wrong - crashes on error
tool_response = await tool(context, **tool_call.arguments)

# Right - handles gracefully
try:
    tool_response = await tool(context, **tool_call.arguments)
    status = "success"
except Exception as e:
    tool_response = str(e)
    status = "error"
```

**Mistake 2: Forgetting to record tool results**
```python
# Wrong - agent doesn't see results
await tool(context, **tool_call.arguments)
# Missing: recording ToolResult

# Right - records for next step
tool_result = ToolResult(...)
context.add_event(Event(content=[tool_result]))
```

---

## Exercises

1. Add tool execution timeout
2. Implement tool result caching
3. Add tool usage statistics
4. Create tool execution logger

---

**Previous Episode**: [Episode 6: Building the Tool System](./EPISODE_06_TOOL_SYSTEM.md)  
**Next Episode**: [Episode 8: MCP Integration](./EPISODE_08_MCP.md)

