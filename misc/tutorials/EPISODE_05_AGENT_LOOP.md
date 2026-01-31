# Episode 5: The Basic Agent Loop

**Duration**: 35 minutes  
**What to Build**: Basic `agent_framework/agent.py` (no tools yet)  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Agent that can have conversations
- Multi-step reasoning
- Execution tracking

**Hook Statement**: "Today we'll build the core agent loop - the brain that orchestrates everything. This is where the magic happens."

---

### 2. Problem (3 min)
**Why do we need an agent loop?**

**The Challenge:**
- LLMs are stateless
- Need to maintain conversation history
- Need to track execution steps
- Need to know when to stop

**The Solution:**
- Think-Act-Observe cycle
- ExecutionContext for state
- Event recording
- Step-by-step execution

---

### 3. Concept: Think-Act-Observe Cycle (5 min)

**The Cycle:**
1. **Think**: Call LLM with context
2. **Act**: Execute tools if needed (next episode)
3. **Observe**: Process results and continue

**Why This Pattern?**
- Mimics human reasoning
- Allows multi-step problem solving
- Clear separation of concerns
- Easy to debug

---

### 4. Live Coding: Building the Agent (25 min)

#### Step 1: Agent.__init__ (3 min)
```python
from dataclasses import dataclass
from typing import List, Optional
from .llm import LlmClient
from .models import ExecutionContext

class Agent:
    """Agent that can reason and use tools to solve tasks."""
    
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 5,
        name: str = "agent"
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.tools = tools or []
```

**Key Parameters:**
- `model`: LLM client
- `tools`: Available tools (empty for now)
- `instructions`: System prompt
- `max_steps`: Safety limit
- `name`: Agent identifier

**Live Coding**: Build __init__

---

#### Step 2: Agent.run() Method (5 min)
```python
from .models import Event, Message, AgentResult

async def run(
    self, 
    user_input: str, 
    context: ExecutionContext = None
) -> AgentResult:
    """Execute the agent."""
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
        # Check if the last event is a final response
        last_event = context.events[-1]
        if self._is_final_response(last_event):
            context.final_result = self._extract_final_result(last_event)
    
    return AgentResult(output=context.final_result, context=context)
```

**Key Points:**
- Creates context if needed
- Records user input
- Loops until done
- Checks for completion

**Live Coding**: Build run()

---

#### Step 3: Agent.step() Method (5 min)
```python
async def step(self, context: ExecutionContext):
    """Execute one step of the agent loop."""
    
    # Prepare LLM request
    llm_request = self._prepare_llm_request(context)
   
    # Get LLM's decision (Think)
    llm_response = await self.think(llm_request)
    
    # Record LLM response as an event
    response_event = Event(
        execution_id=context.execution_id,
        author=self.name,
        content=llm_response.content,
    )
    context.add_event(response_event)
    
    # Execute tools if needed (Act) - next episode!
    # For now, just record the response
    
    context.increment_step()
```

**Key Points:**
- Prepares request
- Calls LLM
- Records response
- Increments step

**Live Coding**: Build step()

---

#### Step 4: _prepare_llm_request() Method (4 min)
```python
from .llm import LlmRequest

def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
    """Convert execution context to LLM request."""
    # Flatten events into content items
    flat_contents = []
    for event in context.events:
        flat_contents.extend(event.content)
    
    return LlmRequest(
        instructions=[self.instructions] if self.instructions else [],
        contents=flat_contents,
        tools=self.tools,  # Empty for now
        tool_choice=None  # No tools yet
    )
```

**Key Points:**
- Flattens events
- Includes instructions
- Adds conversation history
- No tools yet

**Live Coding**: Build _prepare_llm_request()

---

#### Step 5: think() Method (2 min)
```python
async def think(self, llm_request: LlmRequest) -> LlmResponse:
    """Get LLM's response/decision."""
    return await self.model.generate(llm_request)
```

**Simple wrapper** around LLM client.

**Live Coding**: Build think()

---

#### Step 6: Completion Detection (3 min)
```python
def _is_final_response(self, event: Event) -> bool:
    """Check if this event contains a final response."""
    # For now, if it's a message and no tool calls, it's final
    has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
    has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
    return not has_tool_calls and not has_tool_results

def _extract_final_result(self, event: Event) -> str:
    """Extract final result from event."""
    for item in event.content:
        if isinstance(item, Message) and item.role == "assistant":
            return item.content
    return None
```

**Key Points:**
- Checks for tool activity
- Extracts message content
- Simple for now

**Live Coding**: Build completion detection

---

#### Step 7: AgentResult (3 min)
```python
@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext
    status: Literal["complete", "pending", "error"] = "complete"
```

**Simple result container.**

**Live Coding**: Build AgentResult

---

### 5. Testing the Agent (3 min)

**Basic Conversation:**
```python
from agent_framework import Agent, LlmClient

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    instructions="You are a helpful assistant.",
    max_steps=5
)

result = await agent.run("Hello! My name is Alice.")
print(result.output)
print(f"Steps: {result.context.current_step}")
```

**Multi-Turn:**
```python
# First turn
result1 = await agent.run("My name is Alice")
print(result1.output)

# Second turn (new context - doesn't remember)
result2 = await agent.run("What's my name?")
print(result2.output)  # Doesn't know!
```

**Note**: Session persistence comes later!

---

### 6. Demo: Working Agent (2 min)

**Show:**
- Basic conversation
- Multi-step reasoning
- Execution trace
- Step counting

---

### 7. Next Steps (1 min)

**Preview Episode 6:**
- Building the tool system
- Creating tools
- Tool definitions

**What We Built:**
- Basic agent loop
- Conversation handling
- Execution tracking

---

## Key Takeaways

1. **Think-Act-Observe** cycle is the core pattern
2. **ExecutionContext** tracks all state
3. **Events** record every step
4. **Max steps** prevents infinite loops
5. **Completion detection** knows when to stop

---

## Common Mistakes

**Mistake 1: Not incrementing step**
```python
# Wrong - infinite loop!
while not context.final_result:
    await self.step(context)
    # Missing: context.increment_step()

# Right
await self.step(context)  # step() increments internally
```

**Mistake 2: Not checking max steps**
```python
# Wrong - can loop forever
while not context.final_result:
    await self.step(context)

# Right
while not context.final_result and context.current_step < self.max_steps:
    await self.step(context)
```

---

## Exercises

1. Add verbose logging
2. Implement step-by-step trace display
3. Add error handling for LLM failures
4. Create a "thinking" indicator

---

**Previous Episode**: [Episode 4: The LLM Client](./EPISODE_04_LLM_CLIENT.md)  
**Next Episode**: [Episode 6: Building the Tool System](./EPISODE_06_TOOL_SYSTEM.md)

