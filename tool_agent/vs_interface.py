import vs
from tool_agent.transformer_agent import run_custom_tool

def excute():
    request = "User prompt string"
    default = "create a sphere at point (0,0,0), with radious 30. After that create a wall from point (10,10) to point (100,100)"
    result = vs.StrDialog(request, default)
    run_custom_tool(str(result))