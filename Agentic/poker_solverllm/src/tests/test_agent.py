#!/usr/bin/env python3
"""
Test script for poker agent to debug the hand evaluation issue.
"""

import os
import traceback
from dotenv import load_dotenv

from src.agent.agent import build_graph, run_agent

# Load API key
load_dotenv()


def main():
    """Main test function."""
    try:
        print("Building agent graph...")
        poker_graph = build_graph()
        
        # Test with a text-only query
        print("\nTesting hand evaluation with agent (text only)...")
        result_messages = run_agent(
            poker_graph, 
            "I have AcKc. The board is QcJcTcAd2s. What's my best hand?"
        )
        
        # Optional: Test with a multimodal message (mock image)
        # Uncomment to test with a dummy base64 image
        """
        print("\nTesting with multimodal message (image + text)...")
        # This is a tiny 1x1 transparent pixel
        dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        result_messages = run_agent(
            poker_graph,
            "What do you see in this image?",
            image_base64=dummy_image_b64
        )
        """
        
        print("\n--- Final Conversation History ---")
        for msg in result_messages:
            # Display the type and content of each message
            print(f"{type(msg).__name__}: {msg.content}")
            
            # If there are tool calls or other attributes, display them
            for attr_name in ['tool_calls', 'tool_call_id']:
                if hasattr(msg, attr_name):
                    attr_value = getattr(msg, attr_name)
                    if attr_value:
                        print(f"  {attr_name}: {attr_value}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()