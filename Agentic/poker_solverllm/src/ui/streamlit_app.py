"""
Streamlit UI for the poker coach application.
"""

import os
import base64
from io import BytesIO
import streamlit as st
from PIL import Image

# Import agent components
from src.agent.agent import build_graph, run_agent
from src.utils.image_utils import process_image
from src.constants.config import PAGE_TITLE, SUBTITLE, DEFAULT_PROVIDER
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


def setup_page():
    """Configure the Streamlit page layout and title."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(f"{PAGE_TITLE} ({SUBTITLE})")


def setup_sidebar_config():
    """
    Set up the sidebar with configuration options.
    
    Returns:
        Dictionary with configuration settings
    """
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Provider selection
        provider = st.radio(
            "Select LLM Provider:",
            options=["xai", "openai"],
            index=0 if DEFAULT_PROVIDER == "xai" else 1,
            help="Choose which LLM provider to use for the poker coach."
        )
        
        # API key inputs based on provider
        api_keys = {}
        
        if provider == "xai":
            # XAI (Grok) API Key input
            xai_api_key = st.secrets.get("GROK_API_KEY", os.getenv("GROK_API_KEY"))
            if not xai_api_key:
                xai_api_key = st.text_input("Enter your Grok API Key:", type="password")
                if xai_api_key:
                    os.environ['GROK_API_KEY'] = xai_api_key
                    st.success("Grok API Key provided.")
                else:
                    st.error("Grok API Key is required when using XAI provider.")
            else:
                os.environ['GROK_API_KEY'] = xai_api_key
                st.success("Grok API Key loaded from environment/secrets.")
            
            api_keys["xai"] = xai_api_key
            
        elif provider == "openai":
            # OpenAI API Key input
            openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            if not openai_api_key:
                openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
                if openai_api_key:
                    os.environ['OPENAI_API_KEY'] = openai_api_key
                    st.success("OpenAI API Key provided.")
                else:
                    st.error("OpenAI API Key is required when using OpenAI provider.")
            else:
                os.environ['OPENAI_API_KEY'] = openai_api_key
                st.success("OpenAI API Key loaded from environment/secrets.")
            
            api_keys["openai"] = openai_api_key
        
        # Debug options
        show_tool_messages = st.checkbox(
            "Show Tool Messages",
            value=False,
            help="Show internal tool calls and results in the conversation"
        )

        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Show detailed logs and agent execution steps"
        )
        
        # Return configuration
        return {
            "provider": provider,
            "api_keys": api_keys,
            "show_tool_messages": show_tool_messages,
            "debug_mode": debug_mode
        }


def initialize_agent():
    """
    Initialize the agent graph.
    
    Returns:
        The compiled agent graph
    """
    # Cache the graph object for efficiency across reruns
    @st.cache_resource
    def get_poker_graph():
        try:
            return build_graph()
        except ValueError as e:  # Catch API key error specifically
            st.error(f"Failed to initialize agent: {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during agent initialization: {e}")
            st.stop()
    
    return get_poker_graph()


def display_message(msg, show_tool_messages=False):
    """
    Display a message in the Streamlit chat interface.
    
    Args:
        msg: The message to display
        show_tool_messages: Whether to show tool-related messages
    """
    # Skip internal system messages that are used for directing agent behavior
    if isinstance(msg, SystemMessage) and not show_tool_messages:
        return
    
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            # Check if content is multimodal (list) or plain text
            if isinstance(msg.content, list):
                # Display text parts
                for content_part in msg.content:
                    if content_part.get("type") == "text":
                        st.markdown(content_part.get("text", ""))
                    elif content_part.get("type") == "image_url":
                        # Display image 
                        image_url = content_part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image/jpeg;base64,"):
                            # Extract base64 part and display
                            img_data = image_url.split("base64,")[1]
                            st.image(Image.open(BytesIO(base64.b64decode(img_data))))
            else:
                # Plain text content
                st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage) and show_tool_messages:
        # Only show tool messages if debug mode is enabled
        with st.chat_message("tool", avatar="🛠️"):  # Use a tool avatar
            st.markdown(f"**Tool Used:** `{msg.tool_call_id}`")  # Tool name
            st.text(msg.content)  # Display tool output more plainly
    elif isinstance(msg, SystemMessage) and show_tool_messages:
        # Only show system messages if debug mode is enabled
        with st.chat_message("system", avatar="⚙️"):
            st.markdown(f"**System Message:**")
            st.text(msg.content)


def filter_messages_for_display(messages, show_tool_messages=False):
    """
    Filter messages to only show relevant ones to the user.
    
    Args:
        messages: Full list of messages
        show_tool_messages: Whether to show tool-related messages
    
    Returns:
        Filtered list of messages for display
    """
    if show_tool_messages:
        # Show all messages in debug mode
        return messages
    
    # In normal mode, only show main conversation messages
    return [
        msg for msg in messages
        if (isinstance(msg, (HumanMessage, AIMessage)) or
            # Show the initial system prompt
            (isinstance(msg, SystemMessage) and messages.index(msg) == 0)) 
    ]


def main():
    """Main function for the Streamlit app."""
    # Setup page
    setup_page()
    
    # Get configuration from sidebar
    config = setup_sidebar_config()
    provider = config["provider"]
    show_tool_messages = config["show_tool_messages"]
    debug_mode = config["debug_mode"]
    
    # Check if required API key is available
    required_api_key = config["api_keys"].get(provider)
    if not required_api_key:
        st.error(f"API Key for {provider.upper()} is required.")
        st.stop()
    
    # Initialize agent graph
    poker_graph = initialize_agent()
    
    # Show agent workflow diagram in debug mode
    if debug_mode:
        with st.sidebar.expander("Agent Workflow", expanded=False):
            st.markdown("""
            ### Agent Workflow Diagram
            ```
            ┌───────────────┐
            │ Image Analysis │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │    Planning    │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   Execution    │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   Tool Node    │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │   Formatting   │
            └───────────────┘
            ```
            """)
            
            st.markdown("""
            ### Agent Transitions
            - **Image Analysis:** Extracts information from image if present
            - **Planning:** Determines which poker tools to use
            - **Execution:** Calls the LLM with available tools
            - **Tool Node:** Executes selected poker tools
            - **Formatting:** Creates final response based on tool results
            """)
    
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Stores conversation history
    
    # UI Layout
    col1, col2 = st.columns([2, 1])  # Main chat area, Input area
    
    with col2:
        st.subheader("Your Query")
        
        # LLM Provider indicator
        st.info(f"Using {provider.upper()} as the LLM provider")
        
        uploaded_image = st.file_uploader(
            "Upload Poker Situation Image (Optional)", 
            type=["png", "jpg", "jpeg"]
        )
        user_query = st.text_area(
            "Or describe your situation / ask a question:", 
            height=150,
            placeholder="I have AcKc. The board is QcJcTcAd2s. What's my best hand?"
        )
        submit_button = st.button("Get Advice")
    
    with col1:
        st.subheader("Conversation")
        chat_container = st.container(height=500)  # Make chat scrollable
        
        # Display past messages (filtered based on show_tool_messages)
        with chat_container:
            display_messages = filter_messages_for_display(
                st.session_state.messages,
                show_tool_messages
            )
            for msg in display_messages:
                display_message(msg, show_tool_messages)
    
    # Handle input and agent invocation
    if submit_button and (user_query or uploaded_image):
        if not poker_graph:
            st.error("Agent graph not initialized. Please ensure API key is set.")
            st.stop()
        
        image_b64 = None
        input_text = user_query if user_query else "Analyze the situation in the image."
        
        if uploaded_image:
            st.info("Processing uploaded image...")
            try:
                image_b64 = process_image(uploaded_image)
            except ValueError as e:
                st.error(f"Failed to process image: {e}")
                st.stop()  # Stop if image processing failed
        
        # Display the user's input and image in the chat interface
        with chat_container:
            with st.chat_message("user"):
                st.markdown(input_text)
                if image_b64:
                    # Display the actual image that will be sent to the model
                    img_data = base64.b64decode(image_b64)
                    st.image(Image.open(BytesIO(img_data)))
        
        # Create a status area for agent progress
        status_area = st.empty()
        progress_area = st.empty()
        
        # Initialize progress display
        with status_area:
            st.info(f"Starting advice generation using {provider.upper()}...")
        
        try:
            # Set up stages and progress indicators
            stages = ["Image Analysis", "Planning", "Execution", "Tool Use", "Formatting"]
            progress = 0
            
            # Display initial progress
            with progress_area:
                progress_bar = st.progress(progress)
                st.text("Starting agents...")
            
            # Custom logging callback to update progress
            def update_progress(message):
                nonlocal progress
                with status_area:
                    st.info(message)
                
                # Check for stage transitions in the log message
                if "[AGENT] Image processing complete" in message:
                    progress = 0.2
                    with progress_area:
                        progress_bar.progress(progress)
                        st.text(f"Working on: {stages[1]}")
                elif "[AGENT] Planning complete" in message:
                    progress = 0.4
                    with progress_area:
                        progress_bar.progress(progress)
                        st.text(f"Working on: {stages[2]}")
                elif "[AGENT] Tool execution complete" in message:
                    progress = 0.6
                    with progress_area:
                        progress_bar.progress(progress)
                        st.text(f"Working on: {stages[3]}")
                elif "[AGENT] Tool calls detected" in message:
                    progress = 0.7
                    with progress_area:
                        progress_bar.progress(progress)
                        st.text(f"Working on: {stages[3]} - Running tools")
                elif "[AGENT] Formatting complete" in message:
                    progress = 0.9
                    with progress_area:
                        progress_bar.progress(progress)
                        st.text(f"Working on: {stages[4]} - Finalizing")
                elif "[AGENT RUNNER] Agent graph execution completed successfully" in message:
                    progress = 1.0
                    with progress_area:
                        progress_bar.progress(progress)
                        st.text("Completed!")
            
            # Set up a capturing callback for print statements
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Redirect stdout to capture print statements
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                # Run the agent, passing the current message history and provider
                updated_messages = run_agent(
                    poker_graph, 
                    input_text, 
                    image_b64, 
                    st.session_state.messages,
                    provider=provider
                )
            
            # Process logs for progress updates
            logs = captured_output.getvalue().split("\n")
            for log_line in logs:
                if log_line.strip():
                    update_progress(log_line)
            
            # Show final completion
            with status_area:
                st.success("Advice generation complete!")
            with progress_area:
                progress_bar.progress(1.0)
                st.text("Done!")
            
            # Show detailed logs if debug mode is enabled
            if debug_mode:
                with st.expander("Debug Logs", expanded=True):
                    st.text_area("Agent Execution Logs", value=captured_output.getvalue(), height=400)
            
            # Update session state with the full history returned by the agent
            st.session_state.messages = updated_messages
            
            # Rerun the script to update the chat display
            st.rerun()
        
        except Exception as e:
            import traceback
            with status_area:
                st.error(f"An error occurred while getting advice: {e}")
            with progress_area:
                st.error(traceback.format_exc())
    
    elif submit_button:
        st.warning("Please enter a query or upload an image.")