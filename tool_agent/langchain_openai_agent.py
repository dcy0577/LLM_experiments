from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import format_tool_to_openai_function
import os
import dotenv
import chainlit as cl
dotenv.load_dotenv(dotenv_path=r"C:\Users\ge25yak\Desktop\LLM_experiments\.env")

def get_flight_in_period(fly_from, fly_to, date_from, date_to, sort):
    print(f'function calling: {fly_from}, {fly_to}, {date_from}, {date_to}, {sort}' )
    
    flights = []
    detail=''
    flight_info = {"price":100, "duration": 100, "availability": 100, "route": detail}  
    flights.append(flight_info)
    return flights

class GetflightInPeriodCheckInput(BaseModel):

    fly_from: str = Field(..., description="the 3-digit code for departure airport")
    fly_to: str = Field(..., description="the 3-digit code for arrival airport")
    date_from: str = Field(..., description="the dd/mm/yyyy format of start date for the range of search")
    date_to: str = Field(..., description="the dd/mm/yyyy format of end date for the range of search")
    sort: str = Field(..., description="the category for low-to-high sorting, only support 'price', 'duration', 'date'")


class GetflightInPeriodTool(BaseTool):
    name = "get_flight_in_period"
    description = """Useful when you need to search the flights info. You can sort the result by "sort" argument.
                    You need to consider 2023 for default year of search.
                    Try to understand the parameters of every flight

                  """
    def _run(self, fly_from: str, fly_to: str, date_from: str, date_to: str, sort: str):
        get_flight_in_period_response = get_flight_in_period(fly_from, fly_to, date_from, date_to, sort)

        return get_flight_in_period_response

    def _arun(self, fly_from: str, fly_to: str, date_from: str, date_to: str, sort: str):
        raise NotImplementedError("This tool does not support async")


    args_schema: Optional[Type[BaseModel]] = GetflightInPeriodCheckInput

class MeaningOfLife(BaseTool):
    name = "meaning_of_life"
    description = """Useful tool to explain what is meaning of life, it takes no input parameter.

                  """
    def _run(self, test=None):
        return {"meaning of life": 3.1415926}  
    def _arun(self, test=None):
        raise NotImplementedError("This tool does not support async")
    
@cl.langchain_factory(use_async=False)
def agent():
    tools = [GetflightInPeriodTool(), MeaningOfLife()]
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",openai_api_key=os.getenv("OPENAI_API_KEY"))
    open_ai_agent = initialize_agent(tools,
                            llm,
                            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                            verbose=True)

    return open_ai_agent