from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import os

load_dotenv()
app = Flask(__name__)

# Predefine the prompt template globally
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a math assistant. Logically solve the problem below and explain step-by-step:
    Question: {question}
    Answer:
    """
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    question = data.get("question", "")
    user_api_key = data.get("api_key", "")

    if not question or not user_api_key:
        return jsonify({"error": "Both question and Groq API key are required"}), 400

    try:
        # Instantiate the LLM with user key
        user_llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=user_api_key)

        # Recreate tools per request
        wikipedia_tool = Tool(
            name="Wikipedia",
            func=WikipediaAPIWrapper().run,
            description="Search info on Wikipedia"
        )

        math_chain = LLMMathChain.from_llm(llm=user_llm)
        calculator = Tool(
            name="Calculator",
            func=math_chain.run,
            description="Solve math expressions"
        )

        chain = LLMChain(llm=user_llm, prompt=prompt_template)
        reasoning_tool = Tool(
            name="Reasoning Tool",
            func=chain.run,
            description="Answer logic-based questions"
        )

        assistant_agent = initialize_agent(
            tools=[wikipedia_tool, calculator, reasoning_tool],
            llm=user_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

        response = assistant_agent.run(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)