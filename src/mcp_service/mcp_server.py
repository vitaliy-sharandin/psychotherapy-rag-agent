import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import InjectedToolCallId
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from mcp.server.fastmcp import FastMCP

from src.tools.search_tools import SearchTools

load_dotenv()
TEXT_GENERATION_MODEL_NAME = os.getenv("TEXT_GENERATION_MODEL_NAME", "ollama")

mcp = FastMCP("tools")

embedding_model = OllamaEmbedding(model_name="mxbai-embed-large")
rag_model = Ollama(model=TEXT_GENERATION_MODEL_NAME)

search_tools = SearchTools(
    rag_model=rag_model,
    embedding_model=embedding_model,
)


@mcp.tool()
def rag_search(tool_call_id: Annotated[str, InjectedToolCallId], request: str):
    """RAG search through local documents vector database based on user request.

    Args:
        request (str): User request.

    """
    return search_tools.rag_search(tool_call_id, request)


@mcp.tool()
def web_search(tool_call_id: Annotated[str, InjectedToolCallId], request: str):
    """Searches web for information based on user's request.

    Args:
        request (str): User request.

    """
    return search_tools.web_search(tool_call_id, request)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
