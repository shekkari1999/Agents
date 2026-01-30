"""Test session manager to verify context persistence across conversations."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_framework import Agent, LlmClient, InMemorySessionManager, display_trace
from agent_tools import calculator


async def main():
    """Test session persistence."""
    
    print("=" * 60)
    print("Session Manager Test - Context Persistence")
    print("=" * 60)
    
    # Create a shared session manager
    session_manager = InMemorySessionManager()
    
    # Create agent with session support
    agent = Agent(
        model=LlmClient(model="gpt-4o-mini"),
        tools=[calculator],
        instructions="You are a helpful assistant with memory. Remember what users tell you.",
        max_steps=5,
        session_manager=session_manager
    )
    
    session_id = "test-user-123"
    
    # === Conversation 1: Introduce yourself ===
    print("\n" + "-" * 60)
    print("Conversation 1: User introduces themselves")
    print("-" * 60)
    
    result1 = await agent.run(
        "Hi! My name is Alice and I'm a software engineer. I love Python.",
        session_id=session_id
    )
    print(f"User: Hi! My name is Alice and I'm a software engineer. I love Python.")
    print(f"Agent: {result1.output}")
    print(f"Events in context: {len(result1.context.events)}")
    
    # === Conversation 2: Ask about something else ===
    print("\n" + "-" * 60)
    print("Conversation 2: Continue conversation")
    print("-" * 60)
    
    result2 = await agent.run(
        "What's 1234 * 5678?",
        session_id=session_id
    )
    print(f"User: What's 1234 * 5678?")
    print(f"Agent: {result2.output}")
    print(f"Events in context: {len(result2.context.events)}")
    
    # === Conversation 3: Test if it remembers ===
    print("\n" + "-" * 60)
    print("Conversation 3: Test memory - Does it remember?")
    print("-" * 60)
    
    result3 = await agent.run(
        "What's my name and what do I do for work?",
        session_id=session_id
    )
    print(f"User: What's my name and what do I do for work?")
    print(f"Agent: {result3.output}")
    print(f"Events in context: {len(result3.context.events)}")
    
    # === Test with a DIFFERENT session ===
    print("\n" + "-" * 60)
    print("Conversation 4: Different session (should NOT remember)")
    print("-" * 60)
    
    result4 = await agent.run(
        "What's my name?",
        session_id="different-user-456"  # Different session!
    )
    print(f"User: What's my name?")
    print(f"Agent: {result4.output}")
    print(f"Events in context: {len(result4.context.events)}")
    
    # === Show session storage ===
    print("\n" + "=" * 60)
    print("Session Storage Summary")
    print("=" * 60)
    
    # Access internal storage to show what's stored
    for sid, session in session_manager._sessions.items():
        print(f"\nSession ID: {sid}")
        print(f"  Events: {len(session.events)}")
        print(f"  State keys: {list(session.state.keys())}")
        print(f"  Created: {session.created_at}")
    
    # === Optional: Show full trace ===
    print("\n" + "=" * 60)
    print("Full Trace for Session 'test-user-123' (Last Conversation)")
    print("=" * 60)
    display_trace(result3.context)


if __name__ == "__main__":
    asyncio.run(main())

