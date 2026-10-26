import logging

from langchain.agents import create_agent as lc_create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def create_agent(tools):
    """
    Create and return a LangChain agent.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000,
    )

    system_prompt = """
You are a helpful AI assistant.

You have access to the following tools:

1. web_search
   - Search the web for current information.

2. wikipedia_search
   - Search Wikipedia for general knowledge.

3. calculator
   - Perform mathematical calculations.

4. document_search
   - Search information from uploaded PDF documents.

Instructions:
- Think step by step.
- Use the appropriate tool whenever needed.
- If the question is about an uploaded PDF, use document_search.
- For current events or web information, use web_search.
- For encyclopedic knowledge, use wikipedia_search.
- For calculations, use calculator.
- If multiple tools are needed, use them together.
- Always provide clear, well-structured answers.
"""

    agent = lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


def run_agent(agent, question: str, chat_history: list):
    """
    Run the agent safely.
    """

    try:
        logger.info("Agent question: %s", question)

        messages = []

        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

            elif isinstance(msg, tuple):
                role, content = msg
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=question))

        response = agent.invoke({"messages": messages})

        if isinstance(response, dict) and "messages" in response:
            answer = response["messages"][-1].content
        else:
            answer = str(response)

        logger.info("Agent responded successfully")
        return answer

    except Exception as e:
        logger.exception("Agent error")
        return f"Sorry, I encountered an error:\n{e}"