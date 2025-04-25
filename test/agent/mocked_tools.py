from typing import Annotated
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage


class MockedTool:
    def rag_search(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        request: str,
    ):
        """
        RAG search through local documents vector database based on user request.

        Args:
            request (str): User request.
        """
        return Command(
            update={
                "rag_search_results": "Founders personality trait of malevolence is a key factor in the success of the company.",
                "messages": [
                    ToolMessage(
                        "Founders personality trait of malevolence is a key factor in the success of the company.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    def web_search(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        request: str,
    ):
        """Searches web for information based on user's request.

        Args:
            request (str): User request.
        """

        return Command(
            update={
                "web_search_results": "Founders personality trait of malevolence is a key factor in the success of the company.",
                "messages": [
                    ToolMessage(
                        "Founders personality trait of malevolence is a key factor in the success of the company.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
