import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.tools.base import StructuredTool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.mcp_service.mcp_client import get_client_tools
from src.metrics.metrics import AGENT_RESPONSE_TIME, timer
from src.prompts import prompts
from src.tools.search_tools import SearchTools

load_dotenv()

PROFILE = os.getenv("PROFILE", "dev")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

TEXT_GENERATION_MODEL_NAME = os.getenv("TEXT_GENERATION_MODEL_NAME", "ollama")

LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_ADDRESS = os.getenv("LLM_ADDRESS", "http://localhost:11434")

EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY", "ollama")
EMBEDDING_MODEL_ADDRESS = os.getenv("EMBEDDING_MODEL_ADDRESS", "http://localhost:11434")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class AgentState(TypedDict):
    request: str

    action: str
    last_node: str

    tool_usage_counter: int

    messages: Annotated[list[AnyMessage], add_messages]


class PsyAgent:
    text_generation_model_vllm = ChatOpenAI(
        model=TEXT_GENERATION_MODEL_NAME,
        api_key=LLM_API_KEY,
        base_url=LLM_ADDRESS,
        temperature=0,
    )
    text_generation_model = ChatOllama(model=TEXT_GENERATION_MODEL_NAME)
    embedding_model = OllamaEmbedding(model_name="mxbai-embed-large")
    rag_model = Ollama(model=TEXT_GENERATION_MODEL_NAME)

    MAX_TOOL_LOOPS = 3

    def __init__(
        self,
        text_generation_model=None,
        vllm_model=False,
        embedding_model=None,
        rag_model=None,
        knowledge_base_folder=f"{SCRIPT_DIR}/resources/pdf",
        web_search_enabled=True,
        rag_search_enabled=True,
        tools=[],
        mcp=False,
        debug=False,
    ):
        self.prompts = prompts

        if vllm_model:
            self.text_generation_model = text_generation_model or PsyAgent.text_generation_model_vllm
        else:
            self.text_generation_model = text_generation_model or PsyAgent.text_generation_model

        self.embedding_model = embedding_model or PsyAgent.embedding_model
        self.rag_model = rag_model or PsyAgent.rag_model

        builder = StateGraph(AgentState)

        enabled_tools = []

        if not mcp:
            if web_search_enabled:
                search_tools = SearchTools(
                    rag_model=self.rag_model,
                    embedding_model=self.embedding_model,
                    knowledge_base_folder=knowledge_base_folder,
                )
                enabled_tools += [StructuredTool.from_function(search_tools.web_search)]

            if rag_search_enabled:
                search_tools = SearchTools(
                    rag_model=self.rag_model,
                    embedding_model=self.embedding_model,
                    knowledge_base_folder=knowledge_base_folder,
                )
                enabled_tools += [StructuredTool.from_function(search_tools.rag_search)]

            if tools:
                enabled_tools = [StructuredTool.from_function(tool) for tool in tools]

            builder.add_node("action_selector", self.action_selector_node)
            builder.add_node("clarify", self.clarify_node)
            builder.add_node("question_answering", self.question_answering_node)

        else:
            builder.add_node("action_selector", self.async_action_selector_node)
            builder.add_node("clarify", self.async_clarify_node)
            builder.add_node("question_answering", self.async_question_answering_node)

            enabled_tools = list(tools)

        self.text_generation_model = self.text_generation_model.bind_tools(enabled_tools)

        self.tool_node = ToolNode(enabled_tools)
        builder.add_node("tools", self.tool_node)

        builder.set_entry_point("action_selector")

        builder.add_conditional_edges("action_selector", self.select_action, self.prompts.ACTION_DETECTION_OPTIONS)

        builder.add_edge("clarify", "action_selector")
        builder.add_edge("tools", "action_selector")
        builder.add_edge("question_answering", "action_selector")

        if mcp:
            self.builder = builder
        else:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            self.graph = builder.compile(
                checkpointer=checkpointer,
                interrupt_after=["clarify", "question_answering"],
                debug=debug,
            )

    @timer(name="action_selector_node", metric=AGENT_RESPONSE_TIME)
    def action_selector_node(self, state: AgentState):
        user_request = self.get_last_human_message(state)

        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(content=self.prompts.ACTION_DETECTION_PROMPT.format(prompt=user_request)),
        ]
        response = self.text_generation_model.invoke(messages)

        return {
            "request": user_request,
            "action": response.content,
            "last_node": "action_selector",
            "messages": [response],
        }

    async def async_action_selector_node(self, state: AgentState):
        user_request = self.get_last_human_message(state)

        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(content=self.prompts.ACTION_DETECTION_PROMPT.format(prompt=user_request)),
        ]
        response = await self.text_generation_model.ainvoke(messages)

        return {
            "request": user_request,
            "action": response.content,
            "last_node": "action_selector",
            "messages": [response],
        }

    def select_action(self, state: AgentState):
        if state["messages"][-1].tool_calls and state["tool_usage_counter"] < PsyAgent.MAX_TOOL_LOOPS:
            return "tools"
        if state["messages"][-1].tool_calls and state["tool_usage_counter"] >= PsyAgent.MAX_TOOL_LOOPS:
            return "question_answering"
        return state["action"]

    @timer(name="clarify_node", metric=AGENT_RESPONSE_TIME)
    def clarify_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.prompts.THERAPIST_POLICY_PROMPT),
            HumanMessage(content=self.prompts.CLARIFICATION_PROMPT.format(prompt=state["request"])),
        ]

        response = self.text_generation_model.invoke(messages)
        return {"messages": [response], "last_node": "clarify"}

    async def async_clarify_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.prompts.THERAPIST_POLICY_PROMPT),
            HumanMessage(content=self.prompts.CLARIFICATION_PROMPT.format(prompt=state["request"])),
        ]

        response = await self.text_generation_model.ainvoke(messages)
        return {"messages": [response], "last_node": "clarify"}

    @timer(name="knowledge_evaluation_node", metric=AGENT_RESPONSE_TIME)
    def knowledge_evaluation_node(self, state: AgentState):
        # Evaluate the knowledge search results and determine if they are relevant
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(
                content=self.prompts.KNOWLEDGE_RELEVANCY_EVALUATION_PROMPT.format(
                    rag=state["rag_search_results"],
                    web=state["web_search_results"],
                    request=state["request"],
                ),
            ),
        ]

        rag_results = state["rag_search_results"] if state["rag_search_results"] else ""
        web_results = state["web_search_results"] if state["web_search_results"] else ""

        # Call ToolNode to evaluate the knowledge search results
        # response = self.tool_node.invoke(messages)

        # Repeat 3 times if the evaluation fails
        return {
            "web_search_results": web_results,
            "rag_search_results": rag_results,
            "tool_usage_counter": state["tool_usage_counter"] + 1,
            "last_node": "knowledge_evaluation",
        }

    @timer(name="question_answering_node", metric=AGENT_RESPONSE_TIME)
    def question_answering_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.prompts.THERAPIST_POLICY_PROMPT),
            HumanMessage(content=self.prompts.QUESTION_ANSWERING_PROMPT.format(request=state["request"])),
        ]

        response = self.text_generation_model.invoke(messages)
        return {"messages": [response], "last_node": "question_answering"}

    async def async_question_answering_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.prompts.THERAPIST_POLICY_PROMPT),
            HumanMessage(content=self.prompts.QUESTION_ANSWERING_PROMPT.format(request=state["request"])),
        ]

        response = await self.text_generation_model.ainvoke(messages)
        return {"messages": [response], "last_node": "question_answering"}

    def get_last_human_message(self, state: AgentState):
        """Get the last human message from the state."""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return message.content
        return ""

    def draw_graph(self):
        from langchain_core.runnables.graph import CurveStyle

        with open("graph.png", "wb") as f:
            f.write(self.graph.get_graph().draw_mermaid_png(curve_style=CurveStyle.NATURAL))


def run_agent_with_messages(input_messages: list, config=None):
    agent = PsyAgent(
        debug=True,
    )
    for user_message in input_messages:
        if (
            "last_node" not in agent.graph.get_state(config).values
            or not agent.graph.get_state(config).values["last_node"]
        ):
            agent.graph.update_state(
                config,
                {
                    "request": "",
                    "action": "",
                    "last_node": "",
                    "tool_usage_counter": 0,
                },
            )
            user_input = HumanMessage(content=user_message)
            for event in agent.graph.stream({"messages": [user_input]}, config, stream_mode="values"):
                print(f"\n {event}")

        else:
            user_response = HumanMessage(content=user_message)

            last_node = agent.graph.get_state(config).values["last_node"]
            agent.graph.update_state(config, {"messages": [user_response]}, as_node=last_node)

            for event in agent.graph.stream(None, config, stream_mode="values"):
                print(f"\n {event}")


async def async_run_mcp_agent_with_messages(input_messages: list):
    config = {"configurable": {"thread_id": "1"}}

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        tools = await get_client_tools()

        agent = PsyAgent(
            mcp=True,
            tools=tools,
            debug=True,
        )

        agent.graph = agent.builder.compile(
            checkpointer=checkpointer,
            interrupt_after=["clarify", "question_answering"],
            debug=True,
        )

        for user_message in input_messages:
            values_list = await agent.graph.aget_state(config)
            nodes = await agent.graph.aget_state(config)
            if "last_node" not in values_list.values or not nodes.values["last_node"]:
                await agent.graph.aupdate_state(
                    config,
                    {
                        "request": "",
                        "action": "",
                        "last_node": "",
                        "tool_usage_counter": 0,
                    },
                )
                user_input = HumanMessage(content=user_message)
                async for event in agent.graph.astream({"messages": [user_input]}, config, stream_mode="values"):
                    print(f"\n {event}")

            else:
                user_response = HumanMessage(content=user_message)

                nodes = await agent.graph.aget_state(config)

                last_node = nodes.values["last_node"]
                await agent.graph.aupdate_state(config, {"messages": [user_response]}, as_node=last_node)

                async for event in agent.graph.astream(None, config, stream_mode="values"):
                    print(f"\n {event}")


if __name__ == "__main__":
    run_agent_with_messages(
        ["Search info on founders personality correlation with success in rag db. Don't clarify."],
        {"configurable": {"thread_id": "1"}},
    )

    # asyncio.run(async_run_mcp_agent_with_messages(
    #     ["Search info on founders personality correlation with success in rag db. Don't clarify."]
    # ))
