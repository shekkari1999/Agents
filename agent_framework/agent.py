"""Agent class for executing multi-step reasoning with tools."""

from dataclasses import dataclass
from typing import List, Optional, Type, Callable, Literal
from pydantic import BaseModel, Field
from .tools import tool
import inspect
import json

from .models import (
    ExecutionContext, 
    Event, 
    Message, 
    ToolCall, 
    ToolResult,
    PendingToolCall,
    ToolConfirmation,
    BaseSessionManager,
    InMemorySessionManager
)
from .tools import BaseTool
from .llm import LlmClient, LlmRequest, LlmResponse


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext
    status: Literal["complete", "pending", "error"] = "complete"
    pending_tool_calls: list[PendingToolCall] = Field(default_factory=list)


class Agent:
    """Agent that can reason and use tools to solve tasks."""
    
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 5,
        name: str = "agent", 
        output_type: Optional[Type[BaseModel]] = None,
        before_tool_callbacks: List[Callable] = None,
        after_tool_callbacks: List[Callable] = None,
        session_manager: BaseSessionManager | None = None


    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name  
        self.output_type = output_type
        self.output_tool_name = None  
        self.tools = self._setup_tools(tools or [])
        # Initialize callback lists
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []

        # Session manager
        self.session_manager = session_manager or InMemorySessionManager()


    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        if self.output_type is not None:
            @tool(
                name="final_answer",
                description="Return the final structured answer matching the required schema."
            )
            def final_answer(output: self.output_type) -> self.output_type:
                return output
            
            tools = list(tools)  # Create a copy to avoid modifying the original
            tools.append(final_answer)
            self.output_tool_name = "final_answer"
        
        return tools
    
    async def run(
        self, 
        user_input: str, 
        context: ExecutionContext = None,
        session_id: Optional[str] = None,
        tool_confirmations: Optional[List[ToolConfirmation]] = None
    ) -> AgentResult:
        """Execute the agent with optional session support.
        
        Args:
            user_input: User's input message
            context: Optional execution context (creates new if None)
            session_id: Optional session ID for persistent conversations
            tool_confirmations: Optional list of tool confirmations for pending calls
        """
        # Load or create session if session_id is provided
        session = None
        if session_id and self.session_manager:
            session = await self.session_manager.get_or_create(session_id)
            # Load session data into context if context is new
            if context is None:
                context = ExecutionContext()
                # Restore events and state from session
                context.events = session.events.copy()
                context.state = session.state.copy()
                context.execution_id = session.session_id
            context.session_id = session_id
    
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
                # Save session state before returning
                if session:
                    session.events = context.events
                    session.state = context.state
                    await self.session_manager.save(session)
                return AgentResult(
                    status="pending",
                    context=context,
                    pending_tool_calls=pending_calls,
                )
            # Check if the last event is a final response
            last_event = context.events[-1]
            if self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)
        
        # Save session after execution completes
        if session:
            session.events = context.events
            session.state = context.state
            await self.session_manager.save(session)

     
        return AgentResult(output=context.final_result, context=context)


    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        if self.output_tool_name:
        # For structured output: check if final_answer tool succeeded
            for item in event.content:
                if (isinstance(item, ToolResult) 
                    and item.name == self.output_tool_name 
                    and item.status == "success"):
                    return True
            return False
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results
    
    def _extract_final_result(self, event: Event) -> str:
        if self.output_tool_name:
            # Extract structured output from final_answer tool result
            for item in event.content:
                if (isinstance(item, ToolResult) 
                    and item.name == self.output_tool_name 
                    and item.status == "success" 
                    and item.content):
                    return item.content[0]
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    async def step(self, context: ExecutionContext):
        """Execute one step of the agent loop."""
        
        # Process pending confirmations if both are present (before preparing request)
        if ("pending_tool_calls" in context.state and "tool_confirmations" in context.state):
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
       
    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
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
        # Determine tool choice strategy
        if self.output_tool_name:
            tool_choice = "required"  # Force tool usage for structured output
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice = tool_choice 
        )
    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Get LLM's response/decision."""
        return await self.model.generate(llm_request) 
    async def act(
    self, 
    context: ExecutionContext, 
    tool_calls: List[ToolCall]
) -> List[ToolResult]:
        tools_dict = {tool.name: tool for tool in self.tools}
        results = []
        pending_calls = []  # ADD THIS

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                raise ValueError(f"Tool '{tool_call.name}' not found")
            
            tool = tools_dict[tool_call.name]
            
            tool_response = None
            status = "success"
            
            # Stage 1: Execute before_tool_callbacks
            for callback in self.before_tool_callbacks:
                result = callback(context, tool_call)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    tool_response = result
                    break
                # Check if confirmation is required
            if tool.requires_confirmation:
                pending = PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=tool.get_confirmation_message(
                        tool_call.arguments
                    )
                )
                pending_calls.append(pending)
                continue
                
            # Stage 2: Execute actual tool only if callback didn't provide a result
            if tool_response is None:
                try:
                    tool_response = await tool(context, **tool_call.arguments)
                except Exception as e:
                    tool_response = str(e)
                    status = "error"
            
            tool_result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status=status,
                content=[tool_response],
            )
            
            # Stage 3: Execute after_tool_callbacks
            for callback in self.after_tool_callbacks:
                result = callback(context, tool_result)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    tool_result = result
                    break
            
            results.append(tool_result)
        if pending_calls:
            context.state["pending_tool_calls"] = [p.model_dump() for p in pending_calls]
        
        return results
    
    async def _process_confirmations(
    self,
    context: ExecutionContext
) -> List[ToolResult]:
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
    # List of dangerous tools requiring approval
DANGEROUS_TOOLS = ["delete_file", "send_email", "execute_sql"]
 
def approval_callback(context: ExecutionContext, tool_call: ToolCall):
    """Requests user approval before executing dangerous tools."""
    # Execute immediately if not a dangerous tool
    if tool_call.name not in DANGEROUS_TOOLS:
        return None
    
    print(f"\n Dangerous tool execution requested")
    print(f"Tool: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    
    response = input("Do you want to execute? (y/n): ").lower().strip()
    
    if response == 'y':
        print(" Approved. Executing...\n")
        return None  # Proceed with actual tool execution
    else:
        print(" Denied. Skipping execution.\n")
        return f"User denied execution of {tool_call.name}"
            

            

