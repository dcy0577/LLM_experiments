import os
import uuid
import ast
from typing import Union, List
from transformers.tools import OpenAiAgent, python_interpreter
from transformers.tools.agents import PreTool, resolve_tools
from transformers.tools.python_interpreter import InterpretorError, evaluate_ast
from astinterp import Interpreter


import dotenv
from transformers import Tool
import vs
from ast import literal_eval 

dotenv.load_dotenv(dotenv_path=r"C:\Users\ge25yak\Desktop\LLM_experiments\.env")

output_sum = ''
def streamer(output):
    global output_sum
    output_sum = output_sum + output


def _evaluate_condition(condition, state, tools):

    if hasattr(condition, "opt"):
        if len(condition.ops) > 1:
            raise InterpretorError("Cannot evaluate conditions with multiple operators")

        left = evaluate_ast(condition.left, state, tools)
        comparator = condition.ops[0]
        right = evaluate_ast(condition.comparators[0], state, tools)

        if isinstance(comparator, ast.Eq):
            return left == right
        elif isinstance(comparator, ast.NotEq):
            return left != right
        elif isinstance(comparator, ast.Lt):
            return left < right
        elif isinstance(comparator, ast.LtE):
            return left <= right
        elif isinstance(comparator, ast.Gt):
            return left > right
        elif isinstance(comparator, ast.GtE):
            return left >= right
        elif isinstance(comparator, ast.Is):
            return left is right
        elif isinstance(comparator, ast.IsNot):
            return left is not right
        elif isinstance(comparator, ast.In):
            return left in right
        elif isinstance(comparator, ast.NotIn):
            return left not in right
        else:
            raise InterpretorError(f"Operator not supported: {comparator}")
    else:
        return True



class CreateWallTool(Tool):
    name = "create_wall"
    description = ("This tool is use to create a wall in Vetorworks. It takes two inputs: 'st_pt', which should be the start point of the wall, ",
                   "'ed_pt', which should be the end point of the wall. Both 'st_pt' and 'ed_pt' are 2D coordinates string, such as '(154, 56)' or '(0.5, 546)'. ",
                   "This tool will return nothing!")
    inputs = ["text", "text"]
    outputs = []

    def __call__(self, st_pt: str, ed_pt: str):
        print(f"wall created with, start point ={st_pt}, end point = {ed_pt}!")
        vs.AlrtDialog(f"wall created with, start point ={st_pt}, end point = {ed_pt}!")
        vs.Wall(literal_eval(st_pt), literal_eval(ed_pt))
    
class CreateSphere(Tool):
    name="create_sphere"
    description = ("this tool is use to create a sphere object in Vetorworks. It takes two inputs: 'center', which should be the center point of the sphere, ",
                   "'radiusDistance', which should be the radius of sphere. 'center' is 3D coordinates string, such as '(154,56,12)' or '(0.5,546,0)'. ",
                   "'radiusDistance' should be float value, such as 50. The function returns a handle to the new sphere object.")
    inputs = ["text", "text"]
    outputs = ["text"]

    def __call__(self, center: str, radiusDistance: float) -> vs.Handle:
        handle = vs.CreateSphere(literal_eval(center), radiusDistance)
        vs.AlrtDialog(f"sphere {str(handle)} created with, center ={center}, radiusDistance = {radiusDistance}!")
        
        return handle
    
class Move(Tool):
    name = "move_obj"
    description = ("This tool is use to move an element of a list of elements in Vetorworks. ",
                   "It takes four requierd input which are the 'xDistance','yDistance','zDistance' and 'handle'. These represent moving distance in x, y, z direction, ",
                   "as well as the element's unique handle. ",
                   "The moving distances in each direction should be either integer or float value. The optional 'handle' can be a list or a single value. ")
    inputs = ["text", "text", "text", "text"]
    outputs = ["text"]

    def __call__(self, xDistance, yDistance, zDistance, handle: Union[vs.Handle, List[str]] ):
        if isinstance(handle, vs.Handle):
            vs.Move3DObj(handle, xDistance, yDistance, zDistance)
            vs.AlrtDialog(f"element {handle} moved!")
        elif isinstance(handle, List):
            for h in handle:
                vs.Move3DObj(h, xDistance, yDistance, zDistance)
            vs.AlrtDialog("all selected elements moved!")
        # elif not handle:
        #     vs.Move3D(xDistance, yDistance, zDistance)
        #     vs.AlrtDialog("Handle not given, moves the most recently created 3D object a relative distance from it's original location.")
        else:
            vs.AlrtDialog("handle type not supported")

    
class DeleteTool(Tool):
    name = "delete_element"
    description = ("This tool is use to delete an elemnt or a list of elements in Vectorworks. ",
                   "It takes the a element's unique handle or a list of handles as input and then delete the elements. ",
                   "The handle can be a value or a list of strings.")
    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, handle: Union[vs.Handle, List[str]]):
        if isinstance(handle, vs.Handle):
            print(f"element {handle} deleted!")
            vs.DelObject(handle)
            vs.AlrtDialog(f"element {handle} deleted!")
        elif isinstance(handle, List):
            for h in handle:
                vs.DelObject(h)
            vs.AlrtDialog(f"elements deleted!")
        else:
            vs.AlrtDialog("handle type not supported")

class FindSelect(Tool):
    name = "find_selected_element"
    description = ("This tool is use to get selected elements or element in Vectorworks. It takes no input, but returns the elemets handles in list. ", 
                   "If there are no selected elements found, it will return an empty list.")
    inputs = []
    outputs = ["text"]

    def __call__(self):
        selected_obj_list =[]
        current_layer = vs.ActLayer()
        if current_layer != None:
            current_obj = vs.FSObject(current_layer)
            while current_obj != None:
                selected_obj_list.append(current_obj)
                current_obj = vs.NextSObj(current_obj)
        else:
            vs.AlrtDialog("No layer exsists!")
        
        vs.AlrtDialog(f"selected objs' handles: {str(selected_obj_list)}")
        return selected_obj_list


def run_custom_tool(query: str, agent_interpreter = True):
    # init tools
    create_wall_tool = CreateWallTool()
    move_tool = Move()
    get_selected_tool = FindSelect()
    delete_tool = DeleteTool()
    create_sphere_tool = CreateSphere()

    prompt_path = r"C:\Users\ge25yak\Desktop\LLM_experiments\tool_agent\run_prompt_template.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_str = f.read()

    agent = OpenAiAgent(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        run_prompt_template=prompt_str,
        additional_tools=[create_wall_tool, move_tool, delete_tool, create_sphere_tool, get_selected_tool]
    )

    agent.set_stream(streamer)
    # delete pretools to optimize performance
    del_pretools(agent)

    # modify the interpreter
    python_interpreter.evaluate_condition = _evaluate_condition

    if agent_interpreter:
        agent.run(query) # return_code=True
    
    # use custom interpreter, not working currently
    else:
        code_string = agent.run(query, return_code=True)
        tools = resolve_tools(code_string, agent.toolbox)
        tree = ast.parse(code_string)
        interp = Interpreter('test')
        interp.visit(tree)

    vs.AlrtDialog(output_sum)


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


