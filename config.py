import os
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#One place to manage API keys, other files will just call get_api_key() function to get the API key.
def get_api_key() -> str:
    """Get and validate OpenAI API key"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        st.error("API key not found!")
        st.stop()

    if not api_key.startswith("sk-"):
        st.error("Invalid API key format!")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("API key loaded successfully")
    return api_key
