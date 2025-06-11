from json import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

client = MultiServerMCPClient(
    {
        "add": {
            "command": "uv",
            "args": ["run","src/mcp_service/mcp_server.py"],
            "transport": "stdio",
            "env": {
                "PYTHONPATH": str(project_root)
            }
        }
    }
)


async def get_client_tools():
    tools = await client.get_tools()
    return tools
