import streamlit as st
from dotenv import load_dotenv
import math
import re

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(page_title="Math Problem Solver", page_icon="🧮")
st.title("🧮 Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Enter Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API Key to continue")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# ✅ Plain Python tools — no LLMMathChain or LLMChain needed

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Input should be a valid Python math expression string,
    e.g. '2 + 3 * 4', 'sqrt(16)', '100 / 5'.
    """
    try:
        # Allow safe math functions
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for general information about a topic.
    Useful for facts about people, places, events, science, history, etc.
    """
    wrapper = WikipediaAPIWrapper()
    return wrapper.run(query)

@tool
def reasoning(question: str) -> str:
    """
    Breaks down and reasons through a logical or word problem step by step.
    Use this for multi-step problems, unit conversions, or verbal reasoning.
    Input should be the full question or sub-problem to reason through.
    """
    # Just return the question — the agent itself does the reasoning
    return f"Reasoning through: {question}\nPlease work through this step by step."

tools = [calculator, wikipedia_search, reasoning]

assistant_agent = create_react_agent(model=llm, tools=tools)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I'm your Math AI Assistant. Ask me any math or reasoning question."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area(
    "Enter your question:",
    value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 blueberries. How many total fruits do I have?"
)

if st.button("Find My Answer"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            try:
                result = assistant_agent.invoke(
                    {"messages": [{"role": "user", "content": question}]}
                )
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
    else:
        st.warning("Please enter a question.")