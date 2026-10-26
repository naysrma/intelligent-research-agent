import streamlit as st
from config import get_api_key
from tools import get_base_tools, get_rag_tool
from agent import create_agent, run_agent

# Page config
st.set_page_config(
    page_title="Intelligent Research Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize API key
get_api_key()

# Page header
st.title("🤖 Intelligent Research Agent")
st.write("""
Ask me anything! I can:

- 🔍 Search the web for current information
- 📚 Search Wikipedia for facts
- 🧮 Calculate mathematical expressions
- 📄 Answer questions from your uploaded PDF
""")

# Sidebar
st.sidebar.title("📄 Document Upload")
st.sidebar.write("Upload a PDF to enable document search")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# Initialize tools
tools = get_base_tools()

# Add RAG tool if PDF uploaded
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        rag_tool = get_rag_tool(uploaded_file)
        if rag_tool:
            tools.append(rag_tool)
            st.sidebar.success("✅ Document ready!")
        else:
            st.sidebar.error("❌ Failed to process PDF")

# Create agent
agent_executor = create_agent(tools)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# Rate limiting
if st.session_state.question_count >= 20:
    st.error("Session limit reached - please refresh the page")
    st.stop()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input with validation
question = st.chat_input("Ask me anything...")

if question:

    # Validate input
    if len(question.strip()) < 3:
        st.warning("Please ask a more detailed question")

    elif len(question) > 500:
        st.warning("Question too long - please keep under 500 characters")

    else:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                answer = run_agent(
                    agent_executor,
                    question,
                    st.session_state.chat_history
                )
                st.write(answer)

        # Update session state
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.question_count += 1

# Footer
st.sidebar.divider()
st.sidebar.caption(
    f"Questions asked: {st.session_state.question_count}/20"
)