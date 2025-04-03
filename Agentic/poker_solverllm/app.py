"""
Main entry point for the Poker Coach application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the Streamlit app main function
from src.ui.streamlit_app import main

# Run the Streamlit app
if __name__ == "__main__":
    main()