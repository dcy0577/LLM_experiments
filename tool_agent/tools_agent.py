#https://github.com/hwchase17/langchain/issues/832
#
# tools_agent.py
#
# zero-shot react agent that reply questions using available tools

#
# The agent gets the question as a command line argument (a quoted sentence).
# $ py tools_agent.py What about the weather today in Genova, Italy
#
import os
import sys
import dotenv

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import openai

# import custom tools
from tool_agent.create_element_tool import Weather

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0, verbose=True)

template = '''\
Please respond to the questions accurately and succinctly. \
If you are unable to obtain the necessary data after seeking help, \
indicate that you do not know.
'''

prompt = PromptTemplate(input_variables=[], template=template)

# debug
# print(prompt.format())

# Load the tool configs that are needed.
llm_weather_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

tools = [
    Weather,
]

# Construct the react agent type.
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# DEBUG
# https://github.com/hwchase17/langchain/issues/912#issuecomment-1426646112
agent.agent.llm_chain.verbose=True

if __name__ == '__main__':
    
    print('Agent answers questions using custom tools')
    while True:
        question = input('Ask a question: ')

        # run the agent
        answer = agent.run(question)
        print(answer)


