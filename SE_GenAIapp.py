import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------- LLM ---------------- #
from langchain_groq import ChatGroq

# ---------------- TOOLS ---------------- #
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
search = DuckDuckGoSearchRun(name="Search")

tools = {
    "wikipedia": wiki,
    "arxiv": arxiv,
    "search": search
}

# ---------------- UI ---------------- #
st.title("🔎 Stable Tool Chatbot (Groq)")

api_key = st.text_input("Enter Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Ask me anything!"}
    ]

# Show chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- CHAT ---------------- #
if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Enter API key")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    with st.chat_message("assistant"):

        try:
            #STEP 1: Decide whether tool is needed
            decision_prompt = f"""
            Decide:
            - If the question needs external info → say TOOL
            - Else → say DIRECT

            Question: {prompt}
            """

            decision = llm.invoke(decision_prompt).content.strip().upper()

            #STEP 2: If tool needed → generate CLEAN query
            if "TOOL" in decision:

                query_prompt = f"""
                Convert this into a clean search query:
                Question: {prompt}

                Return ONLY the query.
                """

                clean_query = llm.invoke(query_prompt).content.strip()

                #Hard fallback if still bad
                if len(clean_query) < 5:
                    clean_query = prompt

                #Choose tool (simple routing)
                if "research" in prompt.lower() or "paper" in prompt.lower():
                    tool = tools["arxiv"]
                elif "who" in prompt.lower() or "history" in prompt.lower():
                    tool = tools["wikipedia"]
                else:
                    tool = tools["search"]

                try:
                    tool_result = tool.invoke(clean_query)
                except:
                    tool_result = "Tool failed. Using general knowledge."

                # 🔥 Final answer generation
                final_prompt = f"""
                Answer based on this information:

                {tool_result}

                Question: {prompt}
                """

                answer = llm.invoke(final_prompt).content

            else:
                #direct answer
                answer = llm.invoke(prompt).content

        except Exception as e:
            answer = f"Error: {str(e)}"

        st.session_state.messages.append(
            {"role": "assistant", 
             "content": answer}
        )

        st.write(answer)