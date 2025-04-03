"""
Agent state definitions for the poker coaching agent.
"""

import operator
from typing import TypedDict, Annotated, Sequence, Optional

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Defines the state of the poker agent.
    
    Attributes:
        messages: The conversation history, which accumulates using operator.add
        stage: The current stage in the agent workflow: "image", "plan", "execute", "tool", "format", "end"
        provider: The LLM provider to use ("xai" or "openai")
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    stage: Optional[str]  # "image", "plan", "execute", "tool", "format", "end"
    provider: Optional[str]  # "xai" or "openai"