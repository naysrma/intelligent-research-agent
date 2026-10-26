from unittest import result

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import logging
import ast
import PyPDF2


logger = logging.getLogger(__name__)
@tool
def calculator (expression: str) -> str: 
    """Calculate mathematical expressions. Input should be a math expression like '2+2' ог 10*5"""
    try:
        #Only allow safe characters
        allowed = set('0123456789+-*/.()')
        if not all(c in allowed for c in expression):
            return "Invalid expression only numbers and operators allowed"
        result = eval(expression)
        logger.info(f"Calculator: {expression} = {result}")
        return str(result)

    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return "Could not calculate please check your expression"

def get_search_tool():
    """Web search using DuckDuckGo"""
    try:
       search = DuckDuckGoSearchRun()
       search.run("test")
       return search
    except Exception as e:
       logger.warning(f"Search tool error: {e}")
       try:
           from langchain_community.utilities import GoogleSearchAPIWrapper
           wrapper= GoogleSearchAPIWrapper(backend= "lite")
           return DuckDuckGoSearchRun(api_wrapper=wrapper)
       except Exception as e2:
              logger.error(f"Google search tool error: {e2}")
              st.warning("Search tool unavailable")
              return None
           

def get_wiki_tool():
    """Wikipedia search tool"""
    try:
        return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    except Exception as e:
        logger.error(f"Wikipedia tool error: {e}")
        st.warning("Wikipedia tool unavailable")
        return None

def get_rag_tool(_uploaded_file):
    print("Entered get_rag_tool")

    try:
        print("Reading PDF")
        pdf_reader = PyPDF2.PdfReader(_uploaded_file)

        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        print("Extracted characters:", len(text))

        if not text.strip():
            print("No text extracted")
            st.error("Could not extract text")
            return None

        print("Creating splitter")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(text)

        print("Chunks:", len(chunks))

        print("Creating embeddings")

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        print("Creating Chroma")

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings
        )

        print("Success!")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        return create_retriever_tool(
            retriever,
            "document_search",
            "Search uploaded PDF."
        )

    except Exception:
        import traceback
        traceback.print_exc()
        raise
    
def get_base_tools() -> list:
    """Return list of base tools filtering out any that failed"""   
    tools = [
    get_search_tool(),
    get_wiki_tool(),
    calculator
    ]

#Filter out None tools
    return [t for t in tools if t is not None]