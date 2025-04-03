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


def route_from_image(state: AgentState) -> Literal["plan", "end"]:
    """
    Routes from image processing node.
    Determines whether to continue to planning or skip directly to executing a response.
    
    Args:
        state: The current agent state
        
    Returns:
        Routing decision: "plan" or "end"
    """
    messages = state["messages"]
    
    # Check if the last 1-2 messages indicate non-poker content
    for msg in messages[-2:]:
        if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str) and "does not contain poker-related content" in content.lower():
                # We detected a non-poker image but will still continue to planning
                # since the user might have a text query we can help with
                print("[AGENT] Non-poker image detected, but continuing to planning for text query")
                return "plan"
    
    print("[AGENT] Image processing complete. Moving to planning stage.")
    return "plan"

def route_from_plan(state: AgentState) -> Literal["execute", "end"]:
    """
    Routes from planning node.
    Handles cases where no tools are needed for non-poker content.
    
    Args:
        state: The current agent state
        
    Returns:
        Routing decision: "execute" or "end"
    """
    messages = state["messages"]
    
    # Check if the planning stage decided no tools are needed
    for msg in messages[-2:]:
        if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                # Check for "no poker tools needed" message or direct response for non-poker content
                if "no poker tools needed" in content.lower() or "doesn't contain poker content" in content.lower():
                    print("[AGENT] Planning decided no tools needed for non-poker content. Ending workflow.")
                    return "end"
    
    print("[AGENT] Planning complete. Moving to execution stage.")
    return "execute"

def route_from_execute(state: AgentState) -> Literal["tool", "plan", "format", "end"]:
    """
    Routes from execution node.
    
    Args:
        state: The current agent state
        
    Returns:
        Routing decision: "tool", "plan", "format", or "end"
    """
    messages = state["messages"]
    last_message = messages[-1]
    
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


def route_from_tool(state: AgentState) -> Literal["format", "end"]:
    """
    Routes from tool execution node.
    
    Args:
        state: The current agent state
        
    Returns:
        Routing decision: "format" or "end"
    """
    print("[AGENT] Tool execution complete. Moving to formatting stage.")
    return "format"


def route_from_format(state: AgentState) -> Literal["end"]:
    """
    Routes from formatting node.
    
    Args:
        state: The current agent state
        
    Returns:
        Routing decision: "end"
    """
    print("[AGENT] Formatting complete. Ending interaction.")
    return "end"


def call_plan_agent(state: AgentState) -> Dict[str, Any]:
    """
    Planning agent to determine which tools to use.
    Handles cases where images don't contain poker content.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with planning results
    """
    print("[AGENT] Starting planning agent...")
    messages = state['messages']
    provider = state.get('provider', DEFAULT_PROVIDER)
    
    # Check if we previously detected a non-poker image
    non_poker_image = False
    for msg in messages[-3:]:  # Check last few messages
        if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str) and "does not contain poker-related content" in content.lower():
                non_poker_image = True
                print("[AGENT] Planning detected previous non-poker image message")
                break
    
    # Create planning agent
    llm = create_llm_client(provider)
    print(f"[AGENT] Created LLM client with provider: {provider}")
    
    # Custom planning prompt with adjustments for non-poker images
    planning_prompt = """
    You are a Poker Planning Agent. Your job is to analyze the user's query and determine which poker tools should be used.
    Available tools:
    - evaluate_poker_hand: For evaluating specific hole cards + board combinations
    - parse_hand_range: For understanding poker range notation
    - calculate_holdem_equity: For calculating win probabilities between ranges
    
    """
    
    # If we detected a non-poker image, adjust the planning instructions
    if non_poker_image:
        planning_prompt += """
        I notice the image provided does not contain poker content. For text-only queries:
        1. If the user's text query is about poker, recommend at least one appropriate tool.
        2. If the user's text query is not about poker, you can respond that no tools are needed.
        
        Format your response as a simple list with tool names and parameter values, or clearly state 
        "No poker tools needed" if the query is not poker-related.
        """
    else:
        planning_prompt += """
        You must recommend at least one tool to use. For each tool, specify the exact parameters to pass.
        Format your response as a simple list with tool names and parameter values, like this example:
        
        Tool: evaluate_poker_hand
        Parameters: hole_cards_string="AcKc", board_string="QcJcTc"
        """
    
    planning_prompt += """
    Do not execute the tools yourself. Just suggest which ones to use.
    """
    
    # Add planning prompt to conversation
    planning_messages = messages + [SystemMessage(content=planning_prompt)]
    print(f"[AGENT] Added planning prompt to messages. Total messages: {len(planning_messages)}")
    
    # Get planning recommendation
    print("[AGENT] Invoking planning LLM...")
    planning_response = llm.invoke(planning_messages)
    print(f"[AGENT] Planning complete. Response type: {type(planning_response)}")
    
    # If non-poker image was detected, check if planning says no tools needed
    if non_poker_image and hasattr(planning_response, 'content'):
        content = planning_response.content
        if isinstance(content, str) and "no poker tools needed" in content.lower():
            # Add a direct response to the user without using tools
            direct_response = AIMessage(content="""
            I notice the image you've provided doesn't contain poker content, and your query doesn't seem to be about poker strategy or hand analysis. 
            
            I'm specialized in poker coaching and analysis. If you have any poker-related questions or want to analyze hands, positions, or strategies, please let me know! 
            
            You can also upload screenshots of poker games, hand histories, or poker situations for detailed analysis.
            """)
            return {
                "messages": messages + [planning_response, direct_response],
                "provider": provider
            }
    
    # Return updated state with the model's response
    return {
        "messages": messages + [planning_response],
        "provider": provider
    }


def process_image(state: AgentState) -> Dict[str, Any]:
    """
    Image processing agent that extracts poker-related information from images.
    Handles cases where images don't contain poker content.
    
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
                if isinstance(content_part, dict) and content_part.get("type") == "image_url":
                    image_content = content_part
                    print("[AGENT] Found image content in message")
                    break
    
    if not image_content:
        # No image found, skip processing
        print("[AGENT] No image found. Skipping image processing, moving to planning stage.")
        return {
            "messages": messages,
            "provider": provider
        }
    
    # For debugging, print image details
    if image_content and "image_url" in image_content:
        url_data = image_content["image_url"]
        print(f"[AGENT] Image URL details: {url_data.get('detail')}")
        # Only print first 50 chars of URL to avoid flooding logs
        url_str = url_data.get('url', '')
        print(f"[AGENT] Image URL prefix: {url_str[:50]}...")
    
    # Create image processing agent
    print(f"[AGENT] Creating LLM client for image processing with provider: {provider}")
    llm = create_llm_client(provider)
    
    # Custom image analysis prompt with instructions for non-poker images
    image_prompt = """
    You are a Poker Image Analysis Agent. Your job is to look at the image and extract all relevant poker details.
    
    First, determine if the image contains a poker game or poker-related content.
    
    If the image DOES contain poker content:
        Describe in detail:
        1. What cards are visible (player hands, community cards)
        2. Chip stacks and bet sizes visible
        3. Number of players
        4. Game type (cash game, tournament)
        5. Any relevant context visible in the image
    
    If the image DOES NOT contain poker content:
        1. State clearly that "The image does not contain poker-related content."
        2. Provide a brief description of what is actually in the image
        3. DO NOT make up or hallucinate poker details that aren't present
    
    Format your response as a clear, detailed description.
    """
    
    # Add image analysis prompt to conversation
    image_analysis_messages = messages + [SystemMessage(content=image_prompt)]
    print(f"[AGENT] Added image analysis prompt to messages. Total messages: {len(image_analysis_messages)}")
    
    try:
        # Get image analysis
        print("[AGENT] Invoking image analysis LLM...")
        image_analysis = llm.invoke(image_analysis_messages)
        print(f"[AGENT] Image analysis complete. Response type: {type(image_analysis)}")
        
        # Check if the response indicates no poker content
        response_content = image_analysis.content if hasattr(image_analysis, 'content') else ""
        if isinstance(response_content, str) and "does not contain poker-related content" in response_content.lower():
            print("[AGENT] Image analysis detected non-poker content")
            
            # Add a follow-up message asking for poker-related images
            non_poker_message = AIMessage(content="I notice the image you've shared doesn't contain poker-related content. For the best poker analysis, please upload an image showing a poker game, hand, or poker-related scenario. In the meantime, I'll try to help with your text query.")
            
            # Update the messages with both the analysis and follow-up
            return {
                "messages": messages + [image_analysis, non_poker_message],
                "provider": provider
            }
        
        # If poker content detected, proceed normally
        return {
            "messages": messages + [image_analysis],
            "provider": provider
        }
    except Exception as e:
        # Handle image processing errors
        print(f"[AGENT] Error during image processing: {e}")
        error_message = AIMessage(content=f"I had trouble processing the image: {str(e)}")
        return {
            "messages": messages + [error_message],
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
    
    # Return updated state with the model's response
    return {
        "messages": messages + [response],
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

    # Add conditional edges based on routing functions
    workflow.add_conditional_edges(
        "image",
        route_from_image,
        {
            "plan": "plan",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "plan",
        route_from_plan,
        {
            "execute": "execute",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "execute",
        route_from_execute,
        {
            "tool": "tool",
            "plan": "plan",
            "format": "format",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tool",
        route_from_tool,
        {
            "format": "format",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "format",
        route_from_format,
        {
            "end": END
        }
    )

    # Print workflow structure for debugging (before compilation)
    print("Graph nodes:", workflow.nodes)
    
    # Compile the graph
    graph = workflow.compile()
    
    # Return the compiled graph
    return graph


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
        # Validate the image is not empty or too small
        if len(image_base64) < 100:
            print(f"[AGENT RUNNER] WARNING: Image data seems very small ({len(image_base64)} bytes)")
        
        # Create a multimodal message with the image - use JPEG format
        user_message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "auto"  # Use 'auto' instead of 'high' for better compatibility
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