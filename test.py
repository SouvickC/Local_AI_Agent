import streamlit as st
import os
import operator
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv 
# --- GROQ IMPORT ---
from langchain_groq import ChatGroq

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- 0. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
# -----------------------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="LangGraph Web Search Agent (Groq)",
    page_icon="⚡", 
    layout="wide"
)

# 1. Environment Variables and Secrets
# Check for API Keys in environment variables
st.title("🌐 Agentic Chatbot ")
st.caption("This chatbot uses Groq for fast decision-making and LangGraph to manage search loops.")

# --- API KEYS ---
groq_api_key = os.environ.get("GROQ_API_KEY") 
tavily_api_key = os.environ.get("TAVILY_API_KEY")

if not groq_api_key or not tavily_api_key:
    st.error("API Keys not found. Please ensure you have a .env file with GROQ_API_KEY and TAVILY_API_KEY.")
    st.stop()
    



# --- 2. LANGGRAPH COMPONENTS (Encapsulated) ---

# Define the State (Shared Memory)
class AgentState(TypedDict):
    """The state of the graph, primarily holding the message history."""
    messages: Annotated[List[BaseMessage], add_messages]

# Define Tools
# We need to re-initialize the tool here as it uses the now-set environment variable
tavily_tool = TavilySearchResults(max_results=3) 
tools = [tavily_tool]
tools_by_name = {tool.name: tool for tool in tools}

# LLM Setup with Prompt Engineering
SYSTEM_PROMPT = (
    "You are a helpful and meticulous research assistant. Your goal is to answer "
    "user questions accurately. If a question is about current events, specific "
    "data, or requires external knowledge beyond your training data, you **MUST** "
    "use the `tavily_search_results_json` tool. "
    "Always cite the information you find."
)

# --- LLM INITIALIZATION (GROQ) ---
# FIX: Changed model from decommissioned 'mixtral-8x7b-32768' 
# to 'mixtral-8x7b-instruct-v0.1', a currently supported and capable tool-use model.
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
model_with_tools = llm.with_config({"system_instruction": SYSTEM_PROMPT}).bind_tools(tools)
# ---------------------------------

# Define Nodes
def call_model(state: AgentState) -> dict:
    """Invokes the LLM (Node 'llm')."""
    messages = state["messages"]

    # When looping back after a tool call, only send the last few messages for context
    if messages and isinstance(messages[-1], ToolMessage):
        # Last Human (query), AI (tool call), and Tool (result) messages
        input_messages = messages[-3:]
    else:
        # Pass all messages for normal chat flow
        input_messages = messages

    response = model_with_tools.invoke(input_messages)
    return {"messages": [response]}


def call_tool(state: AgentState) -> dict:
    """Executes the tool call requested by the LLM (Node 'tool')."""
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return {"messages": [AIMessage(content="Agent error: No tool call found.")]}

    # Execute the first tool call
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    try:
        tool_output = tools_by_name[tool_name].invoke(tool_args)
    except Exception as e:
        tool_output = f"Tool execution failed: {e}"

    # Return the ToolMessage to feed the result back to the LLM
    return {"messages": [ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])]}

# Define Conditional Edge
def should_continue(state: AgentState) -> Literal["tool", "end"]:
    """Decides whether to execute a tool call or end the graph."""
    last_message = state["messages"][-1]
    
    # Check if the LLM's last response includes a request to call a tool
    if last_message.tool_calls:
        return "tool"
    
    # Otherwise, the LLM has generated a final answer, so we end.
    return "end"

# Build and Compile the Graph (using st.cache_resource for single initialization)
# 
@st.cache_resource # cache - avoids re-compilation on every interaction
def compile_agent_graph(llm_tools_model, _tool_list):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("llm", call_model)
    workflow.add_node("tool", call_tool)
    
    workflow.set_entry_point("llm")
    
    # Conditional edge: LLM -> Tool or LLM -> END
    workflow.add_conditional_edges(
        "llm", should_continue, {"tool": "tool", "end": END}
    )
    
    # Normal edge: Tool -> LLM (to interpret tool result)
    workflow.add_edge("tool", "llm")
    
    return workflow.compile()

# Initialize the compiled app
# The call remains the same, but the function signature handles the unhashable 'tools' list correctly
app = compile_agent_graph(model_with_tools, tools)

# --- 3. STREAMLIT CHAT UI AND EXECUTION ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I'm a research agent, now powered by Groq and Tavily web search. Ask me a question that requires external knowledge, like 'What's the latest news on fusion energy?'")]

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and not message.tool_calls:
        with st.chat_message("assistant"):
            st.markdown(message.content)


if prompt := st.chat_input("Ask the agent a question..."):
    
    # 1. Add Human Message to history and display
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare for the assistant's response
    with st.chat_message("assistant"):
        
        # Create a container for the streamed response
        response_placeholder = st.empty()
        full_response = ""
        
        # Run the graph and collect all intermediate and final messages
        current_messages = st.session_state.messages.copy()
        final_ai_message = None
        # -------------------------------------
        # Use st.spinner for a nice loading indicator
        with st.spinner("Agent is working..."):
            
            for chunk in app.stream({"messages": current_messages}):
                # Extract the last message from the last node that executed in the chunk
                node_name = list(chunk.keys())[0]
                node_output = chunk[node_name]
                last_message = node_output['messages'][-1]
                
                # --- Intermediate Step Visibility ---
                if node_name == 'llm' and last_message.tool_calls:
                    tool_call = last_message.tool_calls[0]
                    response_placeholder.info(f"🔎 **Agent Search:** `{tool_call['args']['query']}`")
                    
                elif node_name == 'tool':
                    # Use a placeholder to indicate the fast processing
                    response_placeholder.info("⚡ **Groq Processing:** Integrating search results...")
                
                # --- Streaming Final Answer ---
                elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    # This is the final answer being generated by the LLM
                    
                    if final_ai_message is None:
                        # Initialize the final message object
                        final_ai_message = last_message.copy()
                        full_response = last_message.content
                    else:
                        # Append the streamed content
                        full_response += last_message.content or ""
                        final_ai_message.content = full_response

                    # Display the streamed content in the placeholder
                    response_placeholder.markdown(full_response)
            
            # 3. Finalize and update session state
            if final_ai_message:
                # Ensure the placeholder shows the final, complete result
                response_placeholder.markdown(final_ai_message.content)
                
                # Append the final assistant message to the session state for memory
                st.session_state.messages.append(final_ai_message)
