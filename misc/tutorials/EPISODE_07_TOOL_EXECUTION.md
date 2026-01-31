# Episode 7: Tool Execution & Complete Agent

**Duration**: 45 minutes  
**What to Build**: Complete `agent_framework/agent.py` with tool execution and confirmation  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Complete agent with tool execution
- Calculator and web search working
- Multi-step problem solving
- Tool confirmation workflow

**Hook Statement**: "Today we'll complete the agent by adding tool execution and safety confirmations. This is where everything comes together - the agent can now actually do things, safely!"

---

### 2. Problem (3 min)
**Why do we need tool execution?**

**The Challenge:**
- LLM decides to use tools
- Need to execute them
- Handle results
- Continue reasoning
- Some tools are dangerous!

**The Solution:**
- Act() method
- Tool dispatch
- Result handling
- Error management
- Confirmation workflow for safety

---

### 3. Concept: Tool Execution Flow (5 min)

**The Flow:**
1. LLM returns tool calls
2. Agent finds tools by name
3. Check if confirmation required
4. If yes: pause and return pending calls
5. User submits confirmation
6. Execute approved tools
7. Record results
8. Continue reasoning

**Key Steps:**
- Parse tool calls from response
- Map to tool instances
- Check `requires_confirmation`
- Handle pending state
- Process confirmations
- Create ToolResult objects

---

### 4. Live Coding: Completing the Agent (35 min)

#### Step 1: Update AgentResult (3 min)
```python
from dataclasses import dataclass
from typing import Literal
from pydantic import Field

@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext
    status: Literal["complete", "pending", "error"] = "complete"
    pending_tool_calls: list[PendingToolCall] = Field(default_factory=list)
```

**Key Points:**
- `status`: Indicates if execution is complete, pending, or errored
- `pending_tool_calls`: List of tools awaiting user confirmation
- Enables pausing and resuming agent execution

**Live Coding**: Update AgentResult

---

#### Step 2: Update step() to Handle Tools (5 min)
```python
async def step(self, context: ExecutionContext):
    """Execute one step of the agent loop."""
    
    # Process pending confirmations if both are present
    if ("pending_tool_calls" in context.state and 
        "tool_confirmations" in context.state):
        confirmation_results = await self._process_confirmations(context)
        
        # Add results as an event so they appear in contents
        if confirmation_results:
            confirmation_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=confirmation_results,
            )
            context.add_event(confirmation_event)
        
        # Clear processed state
        del context.state["pending_tool_calls"]
        del context.state["tool_confirmations"]
    
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
- Check for pending confirmations at start
- Process confirmations before LLM call
- Clear processed state after handling

**Live Coding**: Update step()

---

#### Step 3: act() Method with Confirmation (10 min)
```python
from .models import ToolResult, PendingToolCall

async def act(
    self, 
    context: ExecutionContext, 
    tool_calls: List[ToolCall]
) -> List[ToolResult]:
    """Execute tool calls and return results."""
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []
    pending_calls = []  # Track tools needing confirmation

    for tool_call in tool_calls:
        if tool_call.name not in tools_dict:
            raise ValueError(f"Tool '{tool_call.name}' not found")
        
        tool = tools_dict[tool_call.name]
        
        # Check if confirmation is required
        if tool.requires_confirmation:
            pending = PendingToolCall(
                tool_call=tool_call,
                confirmation_message=tool.get_confirmation_message(
                    tool_call.arguments
                )
            )
            pending_calls.append(pending)
            continue  # Skip execution, wait for confirmation
        
        # Execute tool if no confirmation needed
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
    
    # Store pending calls in state for later processing
    if pending_calls:
        context.state["pending_tool_calls"] = [
            p.model_dump() for p in pending_calls
        ]
    
    return results
```

**Key Points:**
- Check `tool.requires_confirmation`
- Create PendingToolCall with message
- Store in `context.state` for persistence
- Skip execution for pending tools

**Live Coding**: Build act() with confirmation

---

#### Step 4: Process Confirmations (10 min)
```python
async def _process_confirmations(
    self,
    context: ExecutionContext
) -> List[ToolResult]:
    """Process user confirmations and execute approved tools."""
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []

    # Restore pending tool calls from state
    pending_map = {
        p["tool_call"]["tool_call_id"]: PendingToolCall.model_validate(p)
        for p in context.state["pending_tool_calls"]
    }

    # Build confirmation lookup by tool_call_id
    confirmation_map = {
        c["tool_call_id"]: ToolConfirmation.model_validate(c)
        for c in context.state["tool_confirmations"]
    }

    # Process ALL pending tool calls
    for tool_call_id, pending in pending_map.items():
        tool = tools_dict.get(pending.tool_call.name)
        confirmation = confirmation_map.get(tool_call_id)

        if confirmation and confirmation.approved:
            # Merge original arguments with modifications
            arguments = {
                **pending.tool_call.arguments,
                **(confirmation.modified_arguments or {})
            }

            # Execute the approved tool
            try:
                output = await tool(context, **arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call_id,
                    name=pending.tool_call.name,
                    status="success",
                    content=[output],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call_id,
                    name=pending.tool_call.name,
                    status="error",
                    content=[str(e)],
                ))
        else:
            # Rejected: either explicitly or not in confirmation list
            if confirmation:
                reason = confirmation.reason or "Tool execution was rejected by user."
            else:
                reason = "Tool execution was not approved."

            results.append(ToolResult(
                tool_call_id=tool_call_id,
                name=pending.tool_call.name,
                status="error",
                content=[reason],
            ))

    return results
```

**Key Points:**
- Deserialize pending calls from state
- Match confirmations by `tool_call_id`
- Execute approved tools with merged arguments
- Return error result for rejected tools
- LLM sees rejection reason and can adapt

**Live Coding**: Build _process_confirmations()

---

#### Step 5: Update run() for Pending State (5 min)
```python
async def run(
    self, 
    user_input: str, 
    context: ExecutionContext = None,
    tool_confirmations: Optional[List[ToolConfirmation]] = None
) -> AgentResult:
    """Execute the agent with optional confirmation support."""
    
    # Store confirmations in state if provided
    if tool_confirmations:
        if context is None:
            context = ExecutionContext()
        context.state["tool_confirmations"] = [
            c.model_dump() for c in tool_confirmations
        ]
    
    # Create or reuse context
    if context is None:
        context = ExecutionContext()
    
    # Add user input as the first event
    user_event = Event(
        execution_id=context.execution_id,
        author="user",
        content=[Message(role="user", content=user_input)]
    )
    context.add_event(user_event)
    
    # Execute steps until completion or max steps reached
    while not context.final_result and context.current_step < self.max_steps:
        await self.step(context)
        
        # Check for pending confirmations after each step
        if context.state.get("pending_tool_calls"):
            pending_calls = [
                PendingToolCall.model_validate(p)
                for p in context.state["pending_tool_calls"]
            ]
            return AgentResult(
                status="pending",
                context=context,
                pending_tool_calls=pending_calls,
            )
        
        # Check if the last event is a final response
        last_event = context.events[-1]
        if self._is_final_response(last_event):
            context.final_result = self._extract_final_result(last_event)
    
    return AgentResult(output=context.final_result, context=context)
```

**Key Points:**
- Accept `tool_confirmations` parameter
- Store in context.state for processing
- Return "pending" status when confirmations needed
- Resume execution when confirmations provided

**Live Coding**: Update run()

---

#### Step 6: Testing Confirmation Workflow (2 min)

**Create Dangerous Tool:**
```python
from agent_framework import tool

@tool(
    requires_confirmation=True,
    confirmation_message="Delete file '{arguments[filename]}'? This cannot be undone."
)
def delete_file(filename: str) -> str:
    """Delete a file from the filesystem."""
    import os
    os.remove(filename)
    return f"Deleted {filename}"
```

**Test Workflow:**
```python
from agent_framework import Agent, LlmClient, ToolConfirmation

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[delete_file],
    instructions="Help manage files."
)

# First call - gets pending
result = await agent.run("Delete the file named 'test.txt'")
print(result.status)  # "pending"
print(result.pending_tool_calls[0].confirmation_message)
# "Delete file 'test.txt'? This cannot be undone."

# User approves
confirmation = ToolConfirmation(
    tool_call_id=result.pending_tool_calls[0].tool_call.tool_call_id,
    approved=True
)

# Resume with confirmation
result = await agent.run(
    "",  # Empty because we're resuming
    context=result.context,
    tool_confirmations=[confirmation]
)
print(result.status)  # "complete"
print(result.output)  # Agent's response about deletion
```

**Test Rejection:**
```python
# User rejects with reason
confirmation = ToolConfirmation(
    tool_call_id=result.pending_tool_calls[0].tool_call.tool_call_id,
    approved=False,
    reason="I don't want to delete that file"
)

result = await agent.run("", context=result.context, tool_confirmations=[confirmation])
# Agent sees the rejection reason and responds appropriately
```

---

### 5. Demo: Complete Agent (3 min)

**Show:**
- Calculator tool working (no confirmation)
- Delete file tool pausing for confirmation
- User approving/rejecting
- Agent adapting to rejection
- Multi-step reasoning with confirmations

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

**Confirmation Not Provided:**
```python
if confirmation:
    reason = confirmation.reason or "Tool execution was rejected by user."
else:
    reason = "Tool execution was not approved."
```

**Best Practices:**
- Always handle errors gracefully
- Return error in ToolResult
- Let agent continue reasoning
- Provide clear rejection reasons

---

### 7. Next Steps (1 min)

**Preview Episode 8:**
- MCP integration
- External tool servers
- Tool discovery

**What We Built:**
- Complete agent with tools
- Tool execution
- Confirmation workflow
- Error handling

---

## Key Takeaways

1. **act()** executes tools from LLM decisions
2. **Tool dispatch** maps names to instances
3. **requires_confirmation** pauses for user approval
4. **PendingToolCall** stores awaiting tools
5. **ToolConfirmation** carries user's decision
6. **_process_confirmations()** handles approved/rejected tools
7. **AgentResult.status** indicates execution state

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

**Mistake 2: Forgetting to store pending calls**
```python
# Wrong - pending calls lost
if tool.requires_confirmation:
    pending_calls.append(...)
# Missing: storing in context.state

# Right - persists for later
if pending_calls:
    context.state["pending_tool_calls"] = [p.model_dump() for p in pending_calls]
```

**Mistake 3: Not clearing processed state**
```python
# Wrong - processes same confirmations repeatedly
confirmation_results = await self._process_confirmations(context)

# Right - clears after processing
del context.state["pending_tool_calls"]
del context.state["tool_confirmations"]
```

**Mistake 4: Ignoring modified arguments**
```python
# Wrong - ignores user modifications
output = await tool(context, **pending.tool_call.arguments)

# Right - merges modifications
arguments = {
    **pending.tool_call.arguments,
    **(confirmation.modified_arguments or {})
}
output = await tool(context, **arguments)
```

---

## Exercises

1. **Add Tool Timeout**: Implement a timeout for tool execution
2. **Implement Auto-Approve**: Add a flag to auto-approve tools for testing
3. **Add Confirmation Expiry**: Make pending confirmations expire after a timeout
4. **Create Approval Logger**: Log all confirmation decisions for audit

---

## Complete Confirmation Flow

```
User: "Delete test.txt"
    |
    v
Agent.run() -> Agent.step() -> Agent.act()
    |
    v
act() sees requires_confirmation=True
    |
    v
Creates PendingToolCall, stores in context.state
    |
    v
Returns to run(), detects pending_tool_calls
    |
    v
Returns AgentResult(status="pending", pending_tool_calls=[...])
    |
    v
User sees confirmation message, submits ToolConfirmation
    |
    v
Agent.run(context=..., tool_confirmations=[...])
    |
    v
step() calls _process_confirmations()
    |
    v
Executes approved tools, returns ToolResults
    |
    v
Agent continues reasoning with results
    |
    v
Returns AgentResult(status="complete", output="File deleted")
```

---

**Previous Episode**: [Episode 6: Building the Tool System](./EPISODE_06_TOOL_SYSTEM.md)  
**Next Episode**: [Episode 8: MCP Integration](./EPISODE_08_MCP.md)
