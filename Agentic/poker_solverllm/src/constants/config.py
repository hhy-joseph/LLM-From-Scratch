"""
Configuration constants for the poker coach application.
"""

# LLM Model Configuration
DEFAULT_XAI_MODEL = "grok-2-vision-1212"
DEFAULT_OPENAI_MODEL = "gpt-4-vision-preview"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PROVIDER = "xai"  # "xai" or "openai"

# Image Processing
MAX_IMAGE_SIZE = (1024, 1024)
IMAGE_QUALITY = 70
IMAGE_FORMAT = "JPEG"

# UI Configuration
PAGE_TITLE = "Poker LLM Coach"
SUBTITLE = "Powered by LLMs & PokerKit"

# Equity Calculation
MAX_SIMULATIONS = 50000
DEFAULT_SIMULATIONS = 10000

# Agent Configuration
ENFORCE_TOOL_USE = True