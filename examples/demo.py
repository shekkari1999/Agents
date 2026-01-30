from pathlib import Path
import sys
import asyncio


# Add parent directory to path so we can import agent_framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_framework.llm import LlmClient, LlmRequest, Message
 
async def main():
    # Create client
    client = LlmClient(model="gpt-5-mini")
    
    # Build request
    request = LlmRequest(
        instructions=["You are a helpful assistant."],
        contents=[Message(role="user", content="What is 2 + 2?")],
        tool_choice = None
    )
 
# Generate response
    response = await client.generate(request)
  # Check for errors first!
    if response.error_message:
        print(f"Error: {response.error_message}")
        return
    
    # Response contains the answer
    if not response.content:
        print("No content in response")
        return
        
    for item in response.content:
        if isinstance(item, Message):
            print(item.content)  # "4"
        else:
            print(f"Got {type(item).__name__}: {item}")
if __name__ == "__main__":
    asyncio.run(main())