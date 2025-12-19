import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio
 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
 
from dotenv import load_dotenv
import google.genai as genai

# Import root_agent directly and alias as vm_recommender_agent
from backend.vm_recommender.agent import root_agent as vm_recommender_agent
 
async def call_vm_recommender_agent(data_summary: str):
    # Session setup
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="vm_recommender_app",
        user_id="test_user",
        session_id="session_1",
        state={"data_summary": data_summary}
    )
 
    # Runner: orchestrates agent execution
    runner = Runner(
        agent=vm_recommender_agent,
        app_name="vm_recommender_app",
        session_service=session_service
    )

    # Prepare the user message
    content = types.Content(role="user", parts=[types.Part(text=data_summary)])


    # Execute the agent and collect final response
    final_text = None
    async for event in runner.run_async(
        user_id="test_user",
        session_id="session_1",
        new_message=content
    ):
        if event.is_final_response():
            # Extract text from parts
            parts = event.content.parts if event.content and event.content.parts else []
            final_text = "".join(p.text or "" for p in parts)
            break
 
    return final_text
 
# # Run it
# if __name__ == "__main__":
#     load_dotenv()
#     # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     summary_text = (
#         "VM usage: CPU 12% avg, memory 30%. Running always. Disk idle."
#     )
#     response = asyncio.run(call_vm_recommender_agent(summary_text))
#     print("Agent final response:", response)