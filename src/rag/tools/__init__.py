from rag.tools.code_sandbox import code_interpreter_tool
from rag.tools.files_generation import generate_image_tool
from rag.tools.satellites import get_satellite_position_tool, get_tle_tool
from rag.tools.web_search import web_search_tool

ALL_TOOLS = [
    web_search_tool,
    get_tle_tool,
    get_satellite_position_tool,
    code_interpreter_tool,
    generate_image_tool,
]

__all__ = ["ALL_TOOLS"]
