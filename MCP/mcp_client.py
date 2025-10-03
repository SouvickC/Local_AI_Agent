## this client.py will connect with servers of weather and math

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():

    client =MultiServerMCPClient(
       {"math":{"command":"Python",
                "args":["math_server.py"], ## full abosolute path needed
                "transport":"stdio"
                },
        "weather":{"url":"http://127.0.0.1:8000/mcp",
                   "transport":"streamable_http",}
       }
    )
    os.environ["GROQ_API_KEY"] =os.getenv("GROQ_API_KEY")
    
    tools = await client.get_tools()
    llm =ChatGroq(model="openai/gpt-oss-20b")
    agent =create_react_agent(llm, tools)


    user_message = input("Enter your question: ")

    weather_response = await agent.ainvoke(
        {"messages":[{"role":"user", "content": user_message}]}
        )
    print("response from weather server:", weather_response["messages"][-1].content)
    

asyncio.run(main())

