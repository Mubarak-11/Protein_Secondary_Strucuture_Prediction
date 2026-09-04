import os
import sys
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp.client.stdio import StdioServerParameters
from .config import get_agent_model
from .tools import predict_q3, predict_q8, batch_predict_q3, batch_predict_q8
from .uniprot_tools import search_uniprot, get_uniprot_entry
from .structure_tools import create_structure_view_link

script_dir = os.path.dirname(os.path.abspath(__file__))
instruction_file_path = os.path.join(script_dir, "agent-prompt.md")
project_root = os.path.dirname(script_dir)

with open(instruction_file_path, "r") as f:
    instruction = f.read()

def _stdio_mcp_toolset(module: str) -> McpToolset:
    """Create an MCP toolset that runs inside the active Python environment."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["-m", module],
                cwd=project_root,
            ),
        ),
    )

bq_toolset = _stdio_mcp_toolset("protein_bq_mcp_server.server")
retrieval_toolset = _stdio_mcp_toolset("protein_retrieval_mcp_server.server")

tools = [
    predict_q3,
    predict_q8,
    batch_predict_q3,
    batch_predict_q8,
    search_uniprot,
    get_uniprot_entry,
    create_structure_view_link,
    bq_toolset,
    retrieval_toolset,
]
root_agent = Agent(
    name = "ProteinResearchAgent",
    description = "Protein Research Assistant that helps with Protein secondary structure from Amino Acids",
    instruction = instruction,
    model = get_agent_model(),
    tools=tools,
)
