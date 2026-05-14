import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_classic.chains import LLMMathChain
from langchain_classic.chains import LLMChain

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

st.set_page_config(
    page_title="Math Problem Solver",
    page_icon="🧮"
)

st.title("🧮 Text to Math Problem Solver")

groq_api_key = st.sidebar.text_input(
    label="Enter Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please enter your Groq API Key to continue")
    st.stop()

llm = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=groq_api_key
)

wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description=(
        "Useful for searching general information "
        "about people, places, history, science, etc."
    )
)

math_chain = LLMMathChain.from_llm(llm=llm)

calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description=(
        "Useful for solving mathematical calculations "
        "and numerical problems."
    )
)

prompt = """
You are a helpful math assistant.

Solve the following question step-by-step
with proper logical reasoning.

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

reasoning_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description=(
        "Useful for solving logical, reasoning, "
        "word problems, and detailed explanations."
    )
)

assistant_agent = initialize_agent(
    tools=[
        wikipedia_tool,
        calculator_tool,
        reasoning_tool
    ],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 I'm your Math AI Assistant. Ask me any math or reasoning question."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


question = st.text_area(
    "Enter your question:",
    value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 blueberries. How many total fruits do I have?"
)


if st.button("Find My Answer"):

    if question:

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):

            # Streamlit callback
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

            # Agent response
            response = assistant_agent.run(
                question,
                callbacks=[st_cb]
            )

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            # Display response
            st.chat_message("assistant").write(response)

    else:
        st.warning("Please enter a question.")