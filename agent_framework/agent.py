"""Agent class for executing multi-step reasoning with tools."""

from dataclasses import dataclass
from typing import List, Optional, Type
from pydantic import BaseModel
import json

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
        output_type: Optional[Type[BaseModel]] = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.output_type = output_type
        self.tools = self._setup_tools(tools or [])

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        return tools
    
    def _prepare_llm_request(self, context: ExecutionContext, enforce_output_type: bool = False) -> LlmRequest:
        """Convert execution context to LLM request.
        
        Args:
            context: Execution context with conversation history
            enforce_output_type: If True, enforce structured output format.
                                Only set to True when expecting final answer.
        """
        # Flatten events into content items
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)
        
        # Only enforce structured output if explicitly requested (for final answer)
        # This allows tool calls to happen first
        response_format = self.output_type if (enforce_output_type and self.output_type) else None
        
        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice="auto" if self.tools else None,
            response_format=response_format,
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
        # Check if we should enforce structured output
        # Only enforce if: we have output_type AND the last event had tool results (meaning tools were used)
        # This allows tool calls to happen first, then we enforce format for final answer
        should_enforce_output = False
        if self.output_type and len(context.events) > 0:
            last_event = context.events[-1]
            # If last event had tool results, we might be ready for final structured answer
            has_tool_results = any(isinstance(item, ToolResult) for item in last_event.content)
            if has_tool_results:
                # Check if the event before that had tool calls
                if len(context.events) >= 2:
                    prev_event = context.events[-2]
                    had_tool_calls = any(isinstance(item, ToolCall) for item in prev_event.content)
                    # If we had tool calls and got results, next response should be final
                    should_enforce_output = had_tool_calls
        
        # Prepare LLM request - don't enforce output type to allow tool calls
        llm_request = self._prepare_llm_request(context, enforce_output_type=should_enforce_output)
        
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
        elif self.output_type and not should_enforce_output:
            # No tool calls but we didn't enforce output type - make one more call to get structured output
            final_request = self._prepare_llm_request(context, enforce_output_type=True)
            final_response = await self.think(final_request)
            
            # Replace the last event with the structured response
            if context.events:
                context.events[-1] = Event(
                    execution_id=context.execution_id,
                    author=self.name,
                    content=final_response.content,
                )
        
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
    
    def _extract_final_result(self, event: Event):
        """Extract the final result from an event."""
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                content = item.content
                
                # If output_type is specified, parse as structured output
                if self.output_type:
                    try:
                        content_json = json.loads(content)
                        return self.output_type.model_validate(content_json)
                    except (json.JSONDecodeError, ValueError):
                        # If parsing fails, return as string
                        return content
                
                return content
        return None
