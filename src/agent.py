import ast
import os
import sqlite3
from typing import Annotated, List, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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
from metrics import REQUEST_COUNT, REQUEST_LATENCY, AGENT_RESPONSE_TIME, timer

load_dotenv()

PROFILE = os.getenv("PROFILE", "dev")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

INSTRUCTIONS_MODEL_NAME = os.getenv("INSTRUCTION_MODEL_NAME", "ollama")
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

    rag_queries: List[str]
    web_queries: List[str]

    rag_search_results: str
    web_search_results: str

    knowledge_search_summary: str
    knowledge_search_failure_point: str
    knowledge_reevaluation_counter: int

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
    text_generation_model = ChatOpenAI(
        model=TEXT_GENERATION_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_ADDRESS, temperature=0
    )
    instructions_model = ChatOpenAI(
        model=INSTRUCTIONS_MODEL_NAME, api_key=LLM_API_KEY, base_url=LLM_ADDRESS, temperature=0
    )

    def __init__(
        self,
        config,
        text_generation_model=text_generation_model,
        instructions_model=instructions_model,
        knowledge_base_folder=f"{SCRIPT_DIR}/resources/pdf",
        knowledge_retrieval=True,
        web_search_enabled=True,
        rag_search_enabled=True,
        debug=False,
    ):
        self.prompts = prompts
        self.knowledge_base_folder = knowledge_base_folder

        self.text_generation_model = text_generation_model
        self.instructions_model = instructions_model

        builder = StateGraph(AgentState)

        builder.add_node("action_selector", self.action_selector_node)
        builder.add_node("clarify", self.clarify_node)
        builder.add_node("question_answering", self.question_answering_node)

        builder.set_entry_point("action_selector")

        if knowledge_retrieval:
            self.prompts.ACTION_DETECTION_PROMPT = self.prompts.ACTION_DETECTION_PROMPT.format(
                knowledge_retrieval_prompt=self.prompts.KNOWLEDGE_RETRIEVAL_PROMPT
            )
            action_detection_options = self.prompts.ACTION_DETECTION_OPTIONS_WITH_KNOWLEDGE_RETRIEVAL

            builder.add_node("knowledge_retrieval", self.knowledge_retrieval_node)
            builder.add_node("knowledge_evaluation", self.knowledge_evaluation_node)
            builder.add_node("knowledge_summary", self.knowledge_summary_node)

            if web_search_enabled:
                builder.add_node("web", self.web_search_node)
                builder.add_edge("knowledge_retrieval", "web")
                builder.add_edge("web", "knowledge_evaluation")
                self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

            if rag_search_enabled:
                builder.add_node("rag", self.rag_search_node)
                builder.add_edge("knowledge_retrieval", "rag")
                builder.add_edge("rag", "knowledge_evaluation")

                self._initialize_vector_store()
                # self._initialize_vector_store_rerank()
                # self._initialize_automerging_store()

            builder.add_edge("knowledge_summary", "question_answering")
            builder.add_conditional_edges(
                "knowledge_evaluation",
                self.knowledge_relevancy_evaluation,
                self.prompts.KNOWLEDGE_RELEVANCY_EVALUATION_OPTIONS,
            )
        else:
            self.prompts.ACTION_DETECTION_PROMPT = self.prompts.ACTION_DETECTION_PROMPT.format(
                knowledge_retrieval_prompt=""
            )
            action_detection_options = self.prompts.ACTION_DETECTION_OPTIONS_NO_KNOWLEDGE_RETRIEVAL

        builder.add_conditional_edges("action_selector", self.select_action, action_detection_options)

        builder.add_edge("clarify", "action_selector")
        builder.add_edge("question_answering", "action_selector")

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        self.graph = builder.compile(
            checkpointer=checkpointer, interrupt_after=["clarify", "question_answering"], debug=debug
        )
        self.graph.update_state(config, {"knowledge_search_summary": "", "knowledge_reevaluation_counter": 0})

    @timer(name="vector_db_creation", metric=AGENT_RESPONSE_TIME)
    def _initialize_vector_store(self):
        Settings.llm = Ollama(model=TEXT_GENERATION_MODEL_NAME)
        documents = SimpleDirectoryReader(f"{self.knowledge_base_folder}", filename_as_id=True).load_data()

        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embedding_model = OllamaEmbedding(model_name="mxbai-embed-large", base_url=EMBEDDING_MODEL_ADDRESS)

        if chroma_collection.count() == 0:
            vector_store_index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=embedding_model
            )
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)
            # TODO Create a proper refresh of db
            # vector_store_index.refresh_ref_docs(documents)

        self.query_engine = vector_store_index.as_query_engine()

    def _initialize_vector_store_rerank(self):
        Settings.llm = Ollama(model=TEXT_GENERATION_MODEL_NAME)

        documents = SimpleDirectoryReader(f"{self.knowledge_base_folder}", filename_as_id=True).load_data()

        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embedding_model = OllamaEmbedding(model_name="mxbai-embed-large", base_url=EMBEDDING_MODEL_ADDRESS)

        if chroma_collection.count() == 0:
            vector_store_index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=embedding_model
            )
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)
            # TODO Create a proper refresh of db
            # vector_store_index.refresh_ref_docs(documents)

        rerank = FlagEmbeddingReranker(top_n=2, model="BAAI/bge-reranker-base")

        self.rerank_query_engine = vector_store_index.as_query_engine(similarity_top_k=6, node_postprocessors=[rerank])

    def _initialize_automerging_store(self):
        Settings.llm = Ollama(model=TEXT_GENERATION_MODEL_NAME)
        Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large", base_url=EMBEDDING_MODEL_ADDRESS)
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

        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(
                content=self.prompts.ACTION_DETECTION_PROMPT.format(
                    knowledge=state["knowledge_search_summary"], prompt=user_request
                )
            ),
        ]
        response = self.instructions_model.invoke(messages)

        return {"request": user_request, "action": response.content, "last_node": "action_selector"}

    def select_action(self, state: AgentState):
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

    @timer(name="knowledge_retrieval_node", metric=AGENT_RESPONSE_TIME)
    def knowledge_retrieval_node(self, state: AgentState):
        rag_queries = state["rag_queries"]
        web_queries = state["web_queries"]

        if state["knowledge_search_failure_point"] == "both":
            queries = self.instructions_model.with_structured_output(Queries).invoke(
                [
                    SystemMessage(
                        content=self.prompts.QUERIES_REGENERATION_PROMPT.format(
                            rag=state["rag_queries"], web=state["web_queries"], request=state["request"]
                        )
                    ),
                    HumanMessage(content=state["request"]),
                ]
            )
            rag_queries = ast.literal_eval(queries.rag_queries) if queries.rag_queries else []
            web_queries = ast.literal_eval(queries.web_queries) if queries.web_queries else []
        elif state["knowledge_search_failure_point"] == "rag":
            queries = self.instructions_model.with_structured_output(RagQueries).invoke(
                [
                    SystemMessage(
                        content=self.prompts.RAG_RENENERATION_PROMPT.format(
                            rag=state["rag_queries"], request=state["request"]
                        )
                    ),
                    HumanMessage(content=state["request"]),
                ]
            )
            rag_queries = ast.literal_eval(queries.rag_queries) if queries.rag_queries else []
        elif state["knowledge_search_failure_point"] == "web":
            queries = self.instructions_model.with_structured_output(WebQueries).invoke(
                [
                    SystemMessage(
                        content=self.prompts.WEB_REGENERATION_PROMPT.format(
                            web=state["web_queries"], request=state["request"]
                        )
                    ),
                    HumanMessage(content=state["request"]),
                ]
            )
            web_queries = ast.literal_eval(queries.web_queries) if queries.web_queries else []
        else:
            queries = self.instructions_model.with_structured_output(Queries).invoke(
                [SystemMessage(content=self.prompts.QUERIES_GENERATION_PROMPT), HumanMessage(content=state["request"])]
            )
            rag_queries = ast.literal_eval(queries.rag_queries) if queries.rag_queries else []
            web_queries = ast.literal_eval(queries.web_queries) if queries.web_queries else []
        return {"rag_queries": rag_queries, "web_queries": web_queries, "last_node": "knowledge_retrieval"}

    @timer(name="rag_search_node", metric=AGENT_RESPONSE_TIME)
    def rag_search_node(self, state: AgentState):
        """RAG search through local documents vector database based on user request"""

        if not state["knowledge_search_failure_point"] or not state["knowledge_search_failure_point"] == "web":
            rag_results = []
            for rag_search_query in state["rag_queries"]:
                result = self.query_engine.query(rag_search_query)
                rag_results.append(result.response)
            return {"rag_search_results": rag_results}
        return {"rag_search_results": []}

    @timer(name="web_search_node", metric=AGENT_RESPONSE_TIME)
    def web_search_node(self, state: AgentState):
        """Searches web for information based on user's request"""

        if not state["knowledge_search_failure_point"] or not state["knowledge_search_failure_point"] == "rag":
            search_results = []
            for q in state["web_queries"]:
                response = self.tavily.search(query=q, max_results=2)
                for r in response["results"]:
                    search_results.append(r["content"])
            return {"web_search_results": search_results}
        return {"web_search_results": []}

    @timer(name="knowledge_evaluation_node", metric=AGENT_RESPONSE_TIME)
    def knowledge_evaluation_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(
                content=self.prompts.KNOWLEDGE_RELEVANCY_EVALUATION_PROMPT.format(
                    rag=state["rag_search_results"], web=state["web_search_results"], request=state["request"]
                )
            ),
        ]

        response = self.instructions_model.invoke(messages)

        failure_point = response.content
        counter = state["knowledge_reevaluation_counter"]
        counter += 1
        web_search_results = state["web_search_results"]
        rag_search_results = state["rag_search_results"]

        if failure_point == "web":
            web_search_results = []
        elif failure_point == "rag":
            rag_search_results = []
        elif failure_point == "both":
            web_search_results = []
            rag_search_results = []

        return {
            "knowledge_search_failure_point": failure_point,
            "knowledge_reevaluation_counter": counter,
            "web_search_results": web_search_results,
            "rag_search_results": rag_search_results,
            "last_node": "knowledge_evaluation",
        }

    @timer(name="knowledge_relevancy_evaluation", metric=AGENT_RESPONSE_TIME)
    def knowledge_relevancy_evaluation(self, state: AgentState):
        failure_point = state["knowledge_search_failure_point"]
        counter = state["knowledge_reevaluation_counter"]

        if not failure_point == "none" and counter <= 3:
            return "knowledge_retrieval"
        elif failure_point == "both":
            return "question_answering"
        else:
            return "knowledge_summary"

    @timer(name="knowledge_summary_node", metric=AGENT_RESPONSE_TIME)
    def knowledge_summary_node(self, state: AgentState):
        rag_search_results = "\n".join(state["rag_search_results"])
        web_search_results = "\n".join(state["web_search_results"])

        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(
                content=self.prompts.KNOWLEDGE_SUMMARY_PROMPT.format(
                    rag=rag_search_results, web=web_search_results, request=state["request"]
                )
            ),
        ]

        response = self.instructions_model.invoke(messages)

        return {
            "knowledge_search_summary": response.content,
            "knowledge_reevaluation_counter": 0,
            "knowledge_search_failure_point": None,
            "rag_search_results": [],
            "web_search_results": [],
            "last_node": "knowledge_summary",
        }

    @timer(name="question_answering_node", metric=AGENT_RESPONSE_TIME)
    def question_answering_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.prompts.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.prompts.THERAPIST_POLICY_PROMPT),
            HumanMessage(
                content=self.prompts.QUESTION_ANSWERING_PROMPT.format(
                    knowledge_summary=state["knowledge_search_summary"], request=state["request"]
                )
            ),
        ]

        response = self.text_generation_model.invoke(messages)
        return {"messages": [response], "last_node": "question_answering"}

    def draw_graph(self):
        from IPython.display import Image

        return Image(self.graph.get_graph().draw_mermaid_png(output_file_path="./graph.png"))

    def initial_invocation(self, input, config):
        user_input = HumanMessage(content=input)
        for event in self.graph.stream({"messages": [user_input]}, config, stream_mode="values"):
            if event["messages"]:
                event["messages"][-1].pretty_print()

    def human_assisted_input_loop(self, input, config):
        while True:
            user_response = HumanMessage(content=input)

            last_node = self.graph.get_state(config).values["last_node"]
            self.graph.update_state(config, {"messages": [user_response]}, as_node=last_node)

            for event in self.graph.stream(None, config, stream_mode="values"):
                if event["messages"]:
                    event["messages"][-1].pretty_print()
