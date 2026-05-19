import streamlit as st
import math

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Math Problem Solver", page_icon="🧮")
st.title("🧮 Text to Math Problem Solver")
st.markdown("Ask any maths or reasoning question and get a step-by-step answer.")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
groq_api_key = st.sidebar.text_input(
    label="Groq API Key",
    type="password",
    placeholder="gsk_...",
)

if not groq_api_key:
    st.info("👈 Please enter your **Groq API Key** in the sidebar to get started.")
    st.stop()

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the numeric result.
    Input must be a valid Python math expression, e.g. '2 + 3 * 4',
    'sqrt(16)', 'log(100, 10)', '100 / 5'.
    Use only when you need an exact numeric calculation.
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for factual information about a topic.
    Useful for constants, unit definitions, historical facts, or scientific data.
    Input should be a concise search phrase.
    """
    try:
        wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)
        return wrapper.run(query)
    except Exception as e:
        return f"Wikipedia search error: {e}"


@tool
def reasoning(problem: str) -> str:
    """
    Helps reason through a multi-step word problem or logical puzzle.
    Returns a structured thinking prompt. Use this before calling the
    calculator when the problem involves multiple steps or unit conversions.
    Input should be the full problem statement.
    """
    return (
        f"Problem to reason through:\n{problem}\n\n"
        "Please identify all given values, what is asked, and solve step by step."
    )


tools = [calculator, wikipedia_search, reasoning]

# ── Agent (cached per API key so it isn't rebuilt on every rerun) ─────────────

@st.cache_resource(show_spinner=False)
def build_agent(api_key: str):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0,
    )
    system_prompt = (
        "You are an expert mathematics tutor. "
        "Always show your reasoning step by step. "
        "Use the calculator tool for any arithmetic to guarantee accuracy. "
        "Use the wikipedia_search tool when you need factual constants or definitions. "
        "Use the reasoning tool to plan your approach for multi-step word problems. "
        "End every answer with a clear final answer highlighted with '**Answer:**'."
    )
    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)


agent = build_agent(groq_api_key)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi 👋 I'm your Maths AI Assistant!\n\n"
                "Ask me any maths or reasoning problem and I'll solve it step by step."
            ),
        }
    ]

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Input area ────────────────────────────────────────────────────────────────

default_question = (
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. "
    "Each pack contains 25 blueberries. How many total fruits do I have?"
)

question = st.text_area(
    "Enter your question:",
    value=default_question,
    height=120,
)

if st.button("🔍 Find My Answer", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Append user message and show it
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Thinking…"):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            # The last message in the result is the assistant's final reply
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

# ── Sidebar extras ────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear chat history"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chat cleared. Ask me a new question!",
        }
    ]
    st.rerun()

st.sidebar.markdown(
    "**Model:** llama-3.3-70b-versatile  \n"
    "**Tools:** Calculator · Wikipedia · Reasoning"
)