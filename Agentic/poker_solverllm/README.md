# Poker LLM Coach

This project is an AI Poker Coach using multiple Large Language Models (Grok or OpenAI), LangGraph for agent orchestration, PokerKit for poker logic analysis, and Streamlit for the user interface.

## Features

* **Multi-Agent Architecture:** Uses a team of specialized agents:
  * Image Analysis Agent: Extracts poker game details from uploaded images
  * Planning Agent: Determines which tools to use for each query
  * Execution Agent: Calls poker tools with appropriate parameters
  * Formatting Agent: Creates user-friendly responses from tool results
* **Provider Selection:** Switch between XAI (Grok) and OpenAI models
* **Enforced Tool Use:** Ensures that analysis is backed by proper poker tools
* **PokerKit Integration:** Leverages PokerKit library for hand evaluation and equity calculation
* **Multimodal Input:** Accepts user queries via text or image uploads
* **Streamlit UI:** Provides an interactive web interface with provider selection
* **Debug Mode:** Optional tool message visibility for transparency

## Project Structure

```
poker_solverllm/
├── app.py                  # Main entry point
├── prompt.txt              # System prompt for the LLM
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── agent/              # Agent implementation
│   │   ├── agent.py        # Core agent functionality
│   │   └── state.py        # Agent state definition
│   ├── constants/          # Configuration values
│   │   └── config.py       # Constants and settings
│   ├── tests/              # Test modules
│   │   ├── test_agent.py   # Tests for the agent
│   │   └── test_poker_tools.py # Tests for poker tools
│   ├── tools/              # Tool implementations
│   │   └── poker_tools.py  # Poker evaluation tools
│   ├── ui/                 # User interface
│   │   └── streamlit_app.py # Streamlit application
│   └── utils/              # Utility functions
│       └── image_utils.py  # Image processing utilities
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd poker_solverllm
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys:**
   * **Recommended:** Create a `.env` file in the project root:
     ```env
     # Grok API key for XAI provider
     GROK_API_KEY="your_grok_api_key_here"
     
     # OpenAI API key for OpenAI provider
     OPENAI_API_KEY="your_openai_api_key_here"
     
     # Optional: If Grok uses a non-standard base URL
     # GROK_BASE_URL="https://api.groq.com/openai/v1"
     ```
   * **Alternatively (Streamlit Secrets):** If deploying on Streamlit Community Cloud, add `GROK_API_KEY` and/or `OPENAI_API_KEY` to your Streamlit secrets.
   * **Fallback:** The app will prompt for the appropriate key in the sidebar based on your provider selection.

## Running the App

1. Ensure your virtual environment is activated.
2. Make sure your API key is configured.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
5. Enter your poker query or upload an image of a poker situation and click "Get Advice".

## How it Works

1. **Input**: The Streamlit UI captures text input or uploads/processes an image into base64 format.
2. **Provider Selection**: User selects between XAI (Grok) or OpenAI as the LLM provider.
3. **Multi-Agent Processing**: The request flows through several specialized agents:
   * **Image Analysis Agent**: If an image is uploaded, extracts all poker-relevant details
   * **Planning Agent**: Determines which tools should be used and with what parameters
   * **Execution Agent**: Executes the suggested tools with appropriate parameters
   * **Tool Node**: Processes the actual tool calls and captures results
   * **Formatting Agent**: Creates a coherent, user-friendly response based on tool results
4. **Tools**: The agents have access to specialized poker tools:
   * `evaluate_poker_hand`: Analyzes a specific hand and board combination
   * `parse_hand_range`: Converts poker range notation into specific hands
   * `calculate_holdem_equity`: Calculates win probabilities between hands/ranges
5. **Response**: The formatted response is displayed to the user in the conversation interface.

## Running Tests

```bash
# Test the poker tools
python -m src.tests.test_poker_tools

# Test the agent functionality
python -m src.tests.test_agent
```

## Future Improvements

* **Enhanced Tool Set:** Add more specialized poker analysis tools
* **Game State Tracking:** Implement state management for multi-street scenarios
* **Performance Optimization:** Improve equity calculation performance
* **UI Enhancements:** Add visualizations for equity results and hand strength
* **Multi-Session Support:** Allow saving and comparing different poker sessions
* **Additional LLM Providers:** Add support for more LLM providers like Anthropic's Claude
* **Specialized Training:** Fine-tune models on poker strategy datasets
* **Persistent Agent Memory:** Implement session-aware agents that remember previous advice

## Credits

- [PokerKit](https://github.com/pokerkit/pokerkit) - Poker game simulation and analysis
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Streamlit](https://streamlit.io/) - Web interface