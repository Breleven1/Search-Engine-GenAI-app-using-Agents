import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- LLM ---------------- #
from langchain_groq import ChatGroq

# ---------------- TOOLS ---------------- #
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun

# Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# DuckDuckGo Search Tool #A
search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="LangChain Tool Chatbot", page_icon="🔎")

st.title("🔎 AI Chatbot with Tools (LangChain + Groq)")
st.write("Ask anything — the model can use web, Wikipedia, and research papers.")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Hi! I can search the web, Wikipedia, and research papers. Ask me anything."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- MAIN CHAT ---------------- #
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append(
        {"role": "user", 
         "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API Key")
        st.stop()

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Invoke model
    with st.chat_message("assistant"):
        response = llm_with_tools.invoke([
            {"role": "user", 
             "content": prompt}
        ])

        answer = response.content

        # Save response
        st.session_state.messages.append(
            {"role": "assistant", 
             "content": answer})

        # Display response
        st.write(answer)
