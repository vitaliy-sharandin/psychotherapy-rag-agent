import ast
import logging
import os
import sqlite3
from typing import Annotated, List, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain.tools.base import StructuredTool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.chroma import ChromaVectorStore
from tavily import TavilyClient

import prompts
from exception_handling import graceful_exceptions
from metrics import AGENT_RESPONSE_TIME, REQUEST_COUNT, REQUEST_LATENCY, timer

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


class Queries(BaseModel):
    """Queries for RAG and web search based on user request"""

    rag_queries: str = Field(description="A list of queries for RAG search")
    web_queries: str = Field(description="A list of queries for web search")


class RagQueries(BaseModel):
    """Queries for RAG search based on user request"""

    rag_queries: str = Field(description="A list of queries for RAG search")


class WebQueries(BaseModel):
    """Queries for web search based on user request"""

    web_queries: str = Field(description="A list of queries for web search")


class PsyAgent:
    text_generation_model_vllm = ChatOpenAI(
        model=TEXT_GENERATION_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_ADDRESS, temperature=0
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
        knowledge_retrieval=True,
        web_search_enabled=True,
        rag_search_enabled=True,
        tools=[],
        debug=False,
    ):
        self.prompts = prompts
        self.knowledge_base_folder = knowledge_base_folder

        if vllm_model:
            self.text_generation_model = text_generation_model or PsyAgent.text_generation_model_vllm
        else:
            self.text_generation_model = text_generation_model or PsyAgent.text_generation_model

        self.embedding_model = embedding_model or PsyAgent.embedding_model
        self.rag_model = rag_model or PsyAgent.rag_model

        builder = StateGraph(AgentState)

        enabled_tools = []

        if knowledge_retrieval:
            if web_search_enabled:
                enabled_tools += [StructuredTool.from_function(self.web_search)]
                self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

            if rag_search_enabled:
                enabled_tools += [StructuredTool.from_function(self.rag_search)]
                self._initialize_vector_store()
                # self._initialize_vector_store_rerank()
                # self._initialize_automerging_store()

            if tools:
                enabled_tools = [StructuredTool.from_function(tool) for tool in tools]

            self.tool_node = ToolNode(enabled_tools)
            builder.add_node("tools", self.tool_node)

        else:
            self.tool_node = ToolNode(enabled_tools)
            builder.add_node("tools", self.tool_node)

        self.text_generation_model = self.text_generation_model.bind_tools(enabled_tools)

        builder.add_node("action_selector", self.action_selector_node)
        builder.add_node("clarify", self.clarify_node)
        builder.add_node("question_answering", self.question_answering_node)

        builder.set_entry_point("action_selector")

        builder.add_conditional_edges("action_selector", self.select_action, self.prompts.ACTION_DETECTION_OPTIONS)

        builder.add_edge("clarify", "action_selector")
        builder.add_edge("tools", "action_selector")
        builder.add_edge("question_answering", "action_selector")

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        self.graph = builder.compile(
            checkpointer=checkpointer, interrupt_after=["clarify", "question_answering"], debug=debug
        )

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

        # TODO break down request into multiple queries using QUERIES_GENERATION_PROMPT
        # rag_results = []
        # for rag_search_query in state["rag_queries"]:
        #     result = state["query_engine"].query(rag_search_query)
        #     rag_results.append(result.response)

        result = self.query_engine.query(request)

        return Command(
            update={
                "rag_search_results": str(result.response),
                "messages": [ToolMessage(result.response, tool_call_id=tool_call_id)],
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

        # TODO break down request into multiple queries using QUERIES_GENERATION_PROMPT
        # search_results = []
        # for q in state["web_queries"]:
        #     response = state["tavily"].search(query=q, max_results=2)
        #     for r in response["results"]:
        #         search_results.append(r["content"])

        response = self.tavily.search(query=request, max_results=2)
        search_results = [r["content"] for r in response["results"]]

        return Command(
            update={
                "web_search_results": str(search_results),
                "messages": [ToolMessage(search_results, tool_call_id=tool_call_id)],
            }
        )

    @timer(name="vector_db_creation", metric=AGENT_RESPONSE_TIME)
    def _initialize_vector_store(self):
        Settings.llm = self.rag_model
        documents = SimpleDirectoryReader(f"{self.knowledge_base_folder}", filename_as_id=True).load_data()

        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if chroma_collection.count() == 0:
            vector_store_index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embedding_model
            )
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embedding_model)
            # TODO Create a proper refresh of db
            # vector_store_index.refresh_ref_docs(documents)

        self.query_engine = vector_store_index.as_query_engine()

    def _initialize_vector_store_rerank(self):
        Settings.llm = self.rag_model

        documents = SimpleDirectoryReader(f"{self.knowledge_base_folder}", filename_as_id=True).load_data()

        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if chroma_collection.count() == 0:
            vector_store_index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embedding_model
            )
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embedding_model)
            # TODO Create a proper refresh of db
            # vector_store_index.refresh_ref_docs(documents)

        rerank = FlagEmbeddingReranker(top_n=2, model="BAAI/bge-reranker-base")

        self.rerank_query_engine = vector_store_index.as_query_engine(similarity_top_k=6, node_postprocessors=[rerank])

    def _initialize_automerging_store(self):
        Settings.llm = self.rag_model
        Settings.embed_model = self.embedding_model
        documents = SimpleDirectoryReader(f"{self.knowledge_base_folder}", filename_as_id=True).load_data()

        chunk_sizes = [2048, 512, 128]
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(documents)

        leaf_nodes = get_leaf_nodes(nodes)

        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)

        if not os.path.exists("merging_index"):
            storage_context = StorageContext.from_defaults(docstore=docstore)
            automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, store_nodes_override=True)
            automerging_index.storage_context.persist(persist_dir="merging_index")
        else:
            storage_context = StorageContext.from_defaults(persist_dir="merging_index")
            automerging_index = load_index_from_storage(storage_context)
            # TODO Create a proper refresh of db

        base_retriever = automerging_index.as_retriever(similarity_top_k=6)
        retriever = AutoMergingRetriever(base_retriever, automerging_index.storage_context, verbose=True)

        rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

        self.automerging_query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])

    @timer(name="action_selector_node", metric=AGENT_RESPONSE_TIME)
    def action_selector_node(self, state: AgentState):
        user_request = state["messages"][-1].content

        try:
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
        except Exception as e:
            logging.error(f"Error in action_selector_node: {e}")
            return {
                "request": user_request,
                "action": "clarify",
                "last_node": "action_selector",
            }

    def select_action(self, state: AgentState):
        if state["messages"][-1].tool_calls and state["tool_usage_counter"] < PsyAgent.MAX_TOOL_LOOPS:
            return "tools"
        elif state["messages"][-1].tool_calls and state["tool_usage_counter"] >= PsyAgent.MAX_TOOL_LOOPS:
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

    @timer(name="knowledge_evaluation_node", metric=AGENT_RESPONSE_TIME)
    def knowledge_evaluation_node(self, state: AgentState):
        # Evaluate the knowledge search results and determine if they are relevant
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(
                content=self.prompts.KNOWLEDGE_RELEVANCY_EVALUATION_PROMPT.format(
                    rag=state["rag_search_results"], web=state["web_search_results"], request=state["request"]
                )
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

    def draw_graph(self):
        from langchain_core.runnables.graph import CurveStyle

        with open("graph.png", "wb") as f:
            f.write(self.graph.get_graph().draw_mermaid_png(curve_style=CurveStyle.NATURAL))

    def run_agent_with_messages(self, input_messages: list, config):
        for user_message in input_messages:
            if (
                "last_node" not in self.graph.get_state(config).values
                or not self.graph.get_state(config).values["last_node"]
            ):
                self.graph.update_state(
                    config,
                    {
                        "request": "",
                        "action": "",
                        "last_node": "",
                        "tool_usage_counter": 0,
                    },
                )
                user_input = HumanMessage(content=user_message)
                for event in self.graph.stream({"messages": [user_input]}, config, stream_mode="values"):
                    print(f"\n {event}")
                    # event["messages"][-1].pretty_print()
            else:
                user_response = HumanMessage(content=user_message)

                last_node = self.graph.get_state(config).values["last_node"]
                self.graph.update_state(config, {"messages": [user_response]}, as_node=last_node)

                for event in self.graph.stream(None, config, stream_mode="values"):
                    print(f"\n {event}")
                    # event["messages"][-1].pretty_print()


if __name__ == "__main__":
    agent = PsyAgent(
        debug=True,
    )
    # agent.run_agent_with_messages(
    #     ["Search info on founders personality correlation with success in rag db. Don't clarify, just do it."],
    #     {"configurable": {"thread_id": "1"}},
    # )
    agent.draw_graph()
