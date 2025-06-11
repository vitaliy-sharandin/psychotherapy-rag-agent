import os
from typing import Annotated

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.chroma import ChromaVectorStore
from tavily import TavilyClient

from src.metrics.metrics import AGENT_RESPONSE_TIME, timer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


class SearchTools:
    def __init__(self, rag_model=None, embedding_model=None, knowledge_base_folder=f"{SCRIPT_DIR}/../resources/pdf"):
        self.rag_model = rag_model
        self.embedding_model = embedding_model
        self.knowledge_base_folder = knowledge_base_folder

        self.query_engine = self._initialize_vector_store()
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

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
                documents,
                storage_context=storage_context,
                embed_model=self.embedding_model,
            )
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embedding_model)
            # TODO Create a proper refresh of db
            # vector_store_index.refresh_ref_docs(documents)

        return vector_store_index.as_query_engine()

    def _initialize_vector_store_rerank(self):
        Settings.llm = self.rag_model

        documents = SimpleDirectoryReader(f"{self.knowledge_base_folder}", filename_as_id=True).load_data()

        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if chroma_collection.count() == 0:
            vector_store_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embedding_model,
            )
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embedding_model)
            # TODO Create a proper refresh of db
            vector_store_index.refresh_ref_docs(documents)

        rerank = FlagEmbeddingReranker(top_n=2, model="BAAI/bge-reranker-base")

        return vector_store_index.as_query_engine(similarity_top_k=6, node_postprocessors=[rerank])

    def rag_search(
        self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        request: str,
    ):
        """RAG search through local documents vector database based on user request.

        Args:
            request (str): User request.

        """
        result = self.query_engine.query(request)

        return Command(
            update={
                "rag_search_results": str(result.response),
                "messages": [ToolMessage(str(result.response), tool_call_id=tool_call_id)],
            },
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
        response = self.tavily.search(query=request, max_results=2)
        search_results = [r["content"] for r in response["results"]]

        return Command(
            update={
                "web_search_results": str(search_results),
                "messages": [ToolMessage(str(search_results), tool_call_id=tool_call_id)],
            },
        )
