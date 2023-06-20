import vs
import importlib
import tool_agent.transformer_agent
importlib.reload(tool_agent.transformer_agent)
from tool_agent.transformer_agent import run_custom_tool
import ptvsd

def excute():
    # debug attach
    DEBUG = True
    if DEBUG:
        svr_addr = str(ptvsd.options.host) + ":" + str(ptvsd.options.port)
        print(" -> Hosting debug server ... (" + svr_addr + ")")
        ptvsd.enable_attach()
        ptvsd.debug_this_thread()
        ptvsd.wait_for_attach(0.3)


    request = "User prompt string"
    default = "create a sphere at center 0,0,0, and with radious 50"
    result = vs.StrDialog(request, default)
    if not vs.DidCancel():
        run_custom_tool(str(result), agent_interpreter = True)
