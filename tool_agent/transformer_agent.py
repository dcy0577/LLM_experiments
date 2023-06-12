import os
from transformers.tools import OpenAiAgent
import dotenv
from transformers import Tool

dotenv.load_dotenv()

class test_tool(Tool):
    name = "test_tool"
    description = "this tool is use to create a element in vetorworks"
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, query: str):
        return "test"


def run():
    agent = OpenAiAgent(
        model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY")
    )

    agent.run("generate an image of a boat in the water")

if __name__ == "__main__":
    run()
