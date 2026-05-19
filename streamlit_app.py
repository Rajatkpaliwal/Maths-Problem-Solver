import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# ✅ Correct imports for langchain 1.x + langgraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(page_title="Math Problem Solver", page_icon="🧮")
st.title("🧮 Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="Enter Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API Key to continue")
    st.stop()

llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for searching general information about people, places, history, science, etc."
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for solving mathematical calculations and numerical problems."
)

prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful math assistant.
Solve the following question step-by-step with proper logical reasoning.

Question:
{question}

Answer:
"""
)

reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Useful for solving logical, reasoning, word problems, and detailed explanations."
)

tools = [wikipedia_tool, calculator_tool, reasoning_tool]

# ✅ create_react_agent from langgraph returns a compiled graph directly
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
            # ✅ LangGraph agent uses .invoke() with a messages list
            result = assistant_agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            # ✅ Extract the last AI message from the graph output
            answer = result["messages"][-1].content

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
    else:
        st.warning("Please enter a question.")