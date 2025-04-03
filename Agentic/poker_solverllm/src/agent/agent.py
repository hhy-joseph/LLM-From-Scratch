"""
Poker coaching agent using LangGraph and LangChain.
"""

import os
from typing import Dict, Any, List, Optional, Literal
import base64
# LangChain imports
from langchain_xai import ChatXAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, 
    ToolMessage, SystemMessage
)

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Local imports
from src.agent.state import AgentState
from src.tools.poker_tools import all_tools
from src.constants.config import (
    DEFAULT_XAI_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS, DEFAULT_PROVIDER, ENFORCE_TOOL_USE
)


def create_llm_client(provider: str = DEFAULT_PROVIDER):
    """
    Creates the LLM client for the specified provider.
    
    Args:
        provider: The provider to use ('xai' or 'openai')
    
    Returns:
        Configured LLM client
    
    Raises:
        ValueError: If API key is not set
    """
    if provider == "xai":
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY environment variable not set.")
        
        # Using XAI model which supports image input
        llm = ChatXAI(
            model=DEFAULT_XAI_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        # Using OpenAI model which supports image input
        llm = ChatOpenAI(
            model=DEFAULT_OPENAI_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'xai' or 'openai'.")
    
    return llm


def should_continue(state: AgentState) -> str:
    """
    Determines whether to call tools, respond directly, or end.
    
    Args:
        state: The current agent state
        
    Returns:
        Routing decision: "plan", "execute", "tool", "format" or "end"
    """
    messages = state['messages']
    last_message = messages[-1]
    
    # Check current stage from state
    stage = state.get("stage", "plan")
    
    # Log the transition decision
    print(f"[AGENT] Current stage: {stage}, deciding next transition...")
    
    # Stage-based routing
    if stage == "plan":
        print(f"[AGENT] Planning complete. Moving to execution stage.")
        # Move to the tool executor (execute)
        return "execute"
    elif stage == "execute":
        # If the LLM made tool calls, route to the tool node
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            print(f"[AGENT] Tool calls detected: {tool_calls}. Moving to tool stage.")
            return "tool"
        # If we're enforcing tool use but no tools were called, go back to planning
        elif ENFORCE_TOOL_USE:
            print(f"[AGENT] No tool calls but tools required. Moving back to planning stage.")
            return "plan"
        # Otherwise, move to formatting the response
        else:
            print(f"[AGENT] No tool calls and tools not required. Moving to formatting stage.")
            return "format"
    elif stage == "tool":
        # After tool execution, always move to format response
        print(f"[AGENT] Tool execution complete. Moving to formatting stage.")
        return "format"
    elif stage == "format":
        # After formatting, end the interaction
        print(f"[AGENT] Formatting complete. Ending interaction.")
        return "end"
    elif stage == "image":
        # After image processing, go to the planning stage
        print(f"[AGENT] Image processing complete. Moving to planning stage.")
        return "plan"
    
    # Default end if unexpected state
    print(f"[AGENT] WARNING: Unexpected state '{stage}'. Ending interaction.")
    return "end"


def call_plan_agent(state: AgentState) -> Dict[str, Any]:
    """
    Planning agent to determine which tools to use.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with planning results
    """
    print("[AGENT] Starting planning agent...")
    messages = state['messages']
    provider = state.get('provider', DEFAULT_PROVIDER)
    
    # Create planning agent
    llm = create_llm_client(provider)
    print(f"[AGENT] Created LLM client with provider: {provider}")
    
    # Custom planning prompt
    planning_prompt = """
    You are a Poker Planning Agent. Your job is to analyze the user's query and determine which poker tools should be used.
    Available tools:
    - evaluate_poker_hand: For evaluating specific hole cards + board combinations
    - parse_hand_range: For understanding poker range notation
    - calculate_holdem_equity: For calculating win probabilities between ranges
    
    You must recommend at least one tool to use. For each tool, specify the exact parameters to pass.
    Format your response as a simple list with tool names and parameter values, like this example:
    
    Tool: evaluate_poker_hand
    Parameters: hole_cards_string="AcKc", board_string="QcJcTc"
    
    Do not execute the tools yourself. Just suggest which ones to use.
    """
    
    # Add planning prompt to conversation
    planning_messages = messages + [SystemMessage(content=planning_prompt)]
    print(f"[AGENT] Added planning prompt to messages. Total messages: {len(planning_messages)}")
    
    # Get planning recommendation
    print("[AGENT] Invoking planning LLM...")
    planning_response = llm.invoke(planning_messages)
    print(f"[AGENT] Planning complete. Response type: {type(planning_response)}")
    
    # Update state to move to execution
    print("[AGENT] Moving to execution stage")
    return {
        "messages": messages + [planning_response],
        "stage": "execute",
        "provider": provider
    }


def process_image(state: AgentState) -> Dict[str, Any]:
    """
    Image processing agent that extracts poker-related information from images.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with image processing results
    """
    print("[AGENT] Starting image processing agent...")
    messages = state['messages']
    provider = state.get('provider', DEFAULT_PROVIDER)
    
    # Find image content in messages
    image_content = None
    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, list):
            for content_part in msg.content:
                if content_part.get("type") == "image_url":
                    image_content = content_part
                    print("[AGENT] Found image content in message")
                    break
    
    if not image_content:
        # No image found, skip processing
        print("[AGENT] No image found. Skipping image processing, moving to planning stage.")
        return {
            "messages": messages,
            "stage": "plan",
            "provider": provider
        }
    
    # Create image processing agent
    print(f"[AGENT] Creating LLM client for image processing with provider: {provider}")
    llm = create_llm_client(provider)
    
    # Custom image analysis prompt
    image_prompt = """
    You are a Poker Image Analysis Agent. Your job is to look at the poker game image and extract all relevant details as text.
    Describe in detail:
    1. What cards are visible (player hands, community cards)
    2. Chip stacks and bet sizes visible
    3. Number of players
    4. Game type (cash game, tournament)
    5. Any relevant context visible in the image
    
    Format your response as a clear, detailed description that can be used by other agents.
    """
    
    # Add image analysis prompt to conversation
    image_analysis_messages = messages + [SystemMessage(content=image_prompt)]
    print(f"[AGENT] Added image analysis prompt to messages. Total messages: {len(image_analysis_messages)}")
    
    # Get image analysis
    print("[AGENT] Invoking image analysis LLM...")
    image_analysis = llm.invoke(image_analysis_messages)
    print(f"[AGENT] Image analysis complete. Response type: {type(image_analysis)}")
    
    # Update state to move to planning
    print("[AGENT] Image processing complete. Moving to planning stage.")
    return {
        "messages": messages + [image_analysis],
        "stage": "plan",
        "provider": provider
    }


def call_model(state: AgentState) -> Dict[str, Any]:
    """
    Invokes the executor LLM with tools.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with the model's response
    """
    print("[AGENT] Starting tool execution agent...")
    messages = state['messages']
    provider = state.get('provider', DEFAULT_PROVIDER)
    
    # Create and bind the LLM with tools
    print(f"[AGENT] Creating LLM client for tool execution with provider: {provider}")
    llm = create_llm_client(provider)
    llm_with_tools = llm.bind_tools(all_tools)
    print(f"[AGENT] Bound {len(all_tools)} tools to LLM")
    
    # Custom execution prompt if we're enforcing tool use
    if ENFORCE_TOOL_USE:
        tool_enforcement_prompt = """
        You must use at least one of the available tools to answer the user's question.
        Do not provide an answer without using a tool first.
        """
        messages = messages + [SystemMessage(content=tool_enforcement_prompt)]
        print("[AGENT] Added tool enforcement prompt - tools are required")
    
    # Invoke the LLM with bound tools
    print("[AGENT] Invoking LLM with tools...")
    response = llm_with_tools.invoke(messages)
    print(f"[AGENT] Tool execution complete. Response type: {type(response)}")
    
    # Check if tools were actually called
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        print(f"[AGENT] Tool calls detected in response: {tool_calls}")
        next_stage = "tool"
    else:
        print("[AGENT] No tool calls detected in response.")
        if ENFORCE_TOOL_USE:
            print("[AGENT] Tools required but none called. Will route back to planning.")
            next_stage = "plan"
        else:
            print("[AGENT] No tools required. Will move to formatting.")
            next_stage = "format"
    
    # Return updated state with the model's response
    print(f"[AGENT] Moving to {next_stage} stage")
    return {
        "messages": messages + [response],
        "stage": next_stage, 
        "provider": provider
    }


def format_response(state: AgentState) -> Dict[str, Any]:
    """
    Final formatting agent that creates a user-friendly response.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with the formatted response
    """
    print("[AGENT] Starting formatting agent...")
    messages = state['messages']
    provider = state.get('provider', DEFAULT_PROVIDER)
    
    # Create formatting agent
    print(f"[AGENT] Creating LLM client for formatting with provider: {provider}")
    llm = create_llm_client(provider)
    
    # Extract tool results from conversation
    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results.append(f"Tool: {msg.tool_call_id}\nResult: {msg.content}")
    
    print(f"[AGENT] Found {len(tool_results)} tool results to format")
    
    # If no tool results (shouldn't happen with ENFORCE_TOOL_USE), use the last agent response
    if not tool_results:
        print("[AGENT] WARNING: No tool results found. Skipping formatting, ending interaction.")
        # Just return the current state with the last message
        return {
            "messages": messages,
            "stage": "end",
            "provider": provider
        }
    
    # Custom formatting prompt
    formatting_prompt = """
    You are a Poker Response Formatting Agent. Your job is to create a clear, helpful response based on the results from tools.
    
    Tool results:
    {}
    
    Create a response that:
    1. Directly answers the user's question
    2. Provides clear strategic advice based on the tool results
    3. Is concise but thorough
    4. Uses proper poker terminology
    
    Do not show the raw tool results or mention that tools were used.
    """.format("\n\n".join(tool_results))
    
    # Add formatting prompt to conversation
    formatting_messages = messages + [SystemMessage(content=formatting_prompt)]
    print(f"[AGENT] Added formatting prompt to messages. Total messages: {len(formatting_messages)}")
    
    # Get formatted response
    print("[AGENT] Invoking formatting LLM...")
    formatted_response = llm.invoke(formatting_messages)
    print(f"[AGENT] Formatting complete. Response type: {type(formatted_response)}")
    
    # Update state for final output
    print("[AGENT] Formatting complete. Moving to end stage.")
    return {
        "messages": messages + [formatted_response],
        "stage": "end",
        "provider": provider
    }


def build_graph():
    """
    Builds the multi-agent graph with nodes for planning, tool execution, and response formatting.
    
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize graph with the agent state
    workflow = StateGraph(AgentState)

    # Set up nodes for each agent
    workflow.add_node("image", process_image)  # Image processing agent
    workflow.add_node("plan", call_plan_agent)  # Planning agent
    workflow.add_node("execute", call_model)   # Tool execution agent
    workflow.add_node("tool", ToolNode(all_tools))  # Tool execution node
    workflow.add_node("format", format_response)  # Response formatting agent

    # Set entry point - start with image processing
    workflow.set_entry_point("image")

    # Add conditional edges based on the should_continue routing
    workflow.add_conditional_edges(
        "image",
        should_continue,
        {
            "plan": "plan",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "plan",
        should_continue,
        {
            "execute": "execute",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "tool": "tool",
            "plan": "plan",
            "format": "format",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tool",
        should_continue,
        {
            "format": "format",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "format",
        should_continue,
        {
            "end": END
        }
    )

    # Compile the graph
    return workflow.compile()


def run_agent(graph, user_input: str, image_base64: Optional[str] = None, 
              current_messages: Optional[List[BaseMessage]] = None,
              provider: str = DEFAULT_PROVIDER):
    """
    Runs the agent graph with the user's input.
    
    Args:
        graph: The compiled agent graph
        user_input: Text input from the user
        image_base64: Optional base64-encoded image (without data URL prefix)
        current_messages: Optional existing conversation history
        provider: The LLM provider to use ('xai' or 'openai')
        
    Returns:
        Updated list of messages after agent processing
    """
    print(f"[AGENT RUNNER] Starting agent execution with provider: {provider}")
    print(f"[AGENT RUNNER] User input: '{user_input}'")
    print(f"[AGENT RUNNER] Image provided: {image_base64 is not None}")
    
    # Load system prompt
    with open("prompt.txt", "r") as file:
        system_prompt = file.read()
    print("[AGENT RUNNER] Loaded system prompt")

    # Initialize message list
    if current_messages is None:
        # First conversation turn - add system message
        initial_messages = [SystemMessage(content=system_prompt)]
        print("[AGENT RUNNER] First conversation turn - created initial messages with system prompt")
    else:
        # Check if we need to add/update the system message
        if len(current_messages) > 0 and isinstance(current_messages[0], SystemMessage):
            # System message exists, make a copy of current messages
            initial_messages = list(current_messages)
            print("[AGENT RUNNER] Using existing conversation with system message")
        else:
            # Add system message to the start
            initial_messages = [SystemMessage(content=system_prompt)] + list(current_messages)
            print("[AGENT RUNNER] Added system message to existing conversation")

    # Format the user input, with or without image data
    if image_base64:
        # Create a multimodal message with the image
        user_message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": user_input
            }
        ])
        print("[AGENT RUNNER] Created multimodal message with image and text")
    else:
        # Create a plain text message - no image content
        user_message = HumanMessage(content=user_input)
        print("[AGENT RUNNER] Created text-only message")

    # Combine system/history + new user message
    input_messages = initial_messages + [user_message]
    print(f"[AGENT RUNNER] Final input message count: {len(input_messages)}")
    
    # Run the agent graph with provider information
    print("[AGENT RUNNER] Invoking agent graph with initial stage: 'image'")
    try:
        final_state = graph.invoke({
            "messages": input_messages,
            "stage": "image",  # Start with image processing
            "provider": provider
        })
        print("[AGENT RUNNER] Agent graph execution completed successfully")
        print(f"[AGENT RUNNER] Final message count: {len(final_state['messages'])}")
        return final_state['messages']
    except Exception as e:
        print(f"[AGENT RUNNER] ERROR during agent graph execution: {e}")
        import traceback
        traceback.print_exc()
        # Return the input messages + an error message
        error_message = AIMessage(content=f"I encountered an error while processing your request: {str(e)}")
        return input_messages + [error_message]