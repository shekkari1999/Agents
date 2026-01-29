"""Agent class for executing multi-step reasoning with tools."""

from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel

from .models import (
    ExecutionContext, 
    Event, 
    Message, 
    ToolCall, 
    ToolResult
)
from .tools import BaseTool
from .llm import LlmClient, LlmRequest, LlmResponse


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext


class Agent:
    """Agent that can reason and use tools to solve tasks."""
    
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.tools = self._setup_tools(tools or [])

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        return tools
    
    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Convert execution context to LLM request."""
        # Flatten events into content items
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)
        
        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice="auto" if self.tools else None,
        )
    
    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Get LLM's response/decision."""
        return await self.model.generate(llm_request)
    
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
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue
            
            tool = tools_dict[tool_call.name]
            
            try:
                output = await tool.execute(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[str(output)],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                ))
        
        return results
    
    async def step(self, context: ExecutionContext):
        """Execute one step of the agent loop."""
        # Prepare what to send to the LLM
        llm_request = self._prepare_llm_request(context)
        
        # Get LLM's decision
        llm_response = await self.think(llm_request)
        
        # Record LLM response as an event
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

    async def run(
        self, 
        user_input: str, 
        context: ExecutionContext = None
    ) -> AgentResult:
        """Run the agent with user input."""
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

    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results
    
    def _extract_final_result(self, event: Event) -> str:
        """Extract the final result from an event."""
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None
