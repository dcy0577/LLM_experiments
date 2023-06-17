import os
import uuid
from transformers.tools import OpenAiAgent
from transformers.tools.agents import PreTool
import dotenv
from transformers import Tool
import vs
from ast import literal_eval 

dotenv.load_dotenv(dotenv_path=".env")

# def run():
#     agent = OpenAiAgent(
#         model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY")
#     )
#     result = agent.run("generate an image of a boat in the water")
#     result.save("img.png")

# def create_wall(height: str, st_pt: tuple, ed_pt: tuple, thickness: str):
#     return print(f"wall created with height={height}, start point ={st_pt}, end point = {ed_pt} and thickness = {thickness}!")


class CreateWallTool(Tool):
    name = "create_wall"
    description = ("this tool is use to create a wall in Vetorworks. It takes two inputs: 'st_pt', which should be the start point of the wall, 'ed_pt', which should be the end point of the wall. Both 'st_pt' and 'ed_pt' are 2D coordinates string, such as '(154, 56)' or '(0.5, 546)'")
    inputs = ["text", "text"]
    outputs = ["text"]

    def __call__(self, st_pt: str, ed_pt: str):
        id = str(uuid.uuid4())
        print(f"wall {id} created with, start point ={st_pt}, end point = {ed_pt}!")
        vs.AlrtDialog(f"wall {id} created with, start point ={st_pt}, end point = {ed_pt}!")
        vs.Wall(literal_eval(st_pt), literal_eval(ed_pt))
        return id
    
class CreateSphere(Tool):
    name="create_sphere"
    description = ("this tool is use to create a sphere object in Vetorworks. It takes two inputs: 'center', which should be the center point of the sphere, 'radiusDistance', which should be the radius of sphere. 'center' is 3D coordinates string, such as '(154,56,12)' or '(0.5,546,0)'. 'radiusDistance' should be float value, such as 50. The function returns a handle to the new sphere object.")
    inputs = ["text", "text"]
    outputs = ["text"]

    def __call__(self, center: str, radiusDistance: float):
        handle = vs.CreateSphere(literal_eval(center), radiusDistance)
        vs.AlrtDialog(f"sphere {handle} created with, center ={center}, radiusDistance = {radiusDistance}!")
        
        return handle
    
class UpdateWallTool(Tool):
    name = "update_wall"
    description = ("This tool is use to update a wall's parameters in Vetorworks. It takes one requierd input which is the wall's unique id, and up to four optional inputs: 'height', which should be the the height of the wall,",
                   " 'st_pt', which should be the start point of the wall, 'ed_pt', which should be the end point of the wall and 'thickness', which ",
                   "should be the thickness of the wall. Both 'st_pt' and 'ed_pt' are 3D coordinates string, such as '(154, 56, 54)' or '(0.5, 546, 100.5)'. Both 'height' ",
                   "and 'thickness' are integer or float value, such as 200 or 200.5. It should be up to the context to decide which optional parameter needs to be updated.")
    inputs = ["text", "text", "text", "text", "text"]
    outputs = ["text"]

    def __call__(self, id: str, **kwrg):
        print(f"wall {id} is updated with the information {kwrg}!")
    
class DeleteTool(Tool):
    name = "delete_element"
    description = ("This tool is use to delete an elemnt in Vectorworks. It takes the element's unique handle as input and then delete the element.")
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, handle: str):
        print(f"element {handle} deleted!")
        vs.DelObject(handle)
        vs.AlrtDialog(f"element {handle} deleted!")

class RetrieveTool(Tool):
    name = "retrieve_element"
    description = ("This tool is use to get an elemnt and its information in Vectorworks. It takes the element's unique id as input and then returns the elemets information.")
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, id: str):
        # some dummy dimensions
        height = "..."
        width = "..."
        print(f"information of element {id} retrieved! Height: {height}, width: {width}")


def run_custom_tool(query: str):
    # init tools
    create_wall_tool = CreateWallTool()
    update_wall_tool = UpdateWallTool()
    retrieve_tool = RetrieveTool()
    delete_tool = DeleteTool()
    create_sphere_tool = CreateSphere()

    agent = OpenAiAgent(
        model="gpt-3.5-turbo",
        # api_key=os.getenv("OPENAI_API_KEY"),
        api_key="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",
        additional_tools=[create_wall_tool, update_wall_tool, retrieve_tool, delete_tool, create_sphere_tool]
    )

    # delete pretools to optimize performance
    del_pretools(agent)

    result = agent.run(query) # return_code=True


def del_pretools(agent):
    del_list = []
    for name, tool in agent.toolbox.items():
        if type(tool) is PreTool:
            del_list.append(name)
    # pop the tools identified
    for name in del_list:
        del agent.toolbox[name]
    # now lets see how much added text will be added to our prompts
    print(agent.toolbox)



if __name__ == "__main__":
    # run()
    run_custom_tool("create a sphere at point 0,0,0 and with raious 30")
    # run_custom_tool("create a wall with height=855, thickness=60, starting at point (15,76,0) and ending at (65,85,0).\
    #                  And then change the height to 200, thickness to 80. \
    #                 After that give me information about the wall. Finally delete the wall")


