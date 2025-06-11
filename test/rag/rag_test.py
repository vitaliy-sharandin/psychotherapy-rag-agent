import os
import uuid

import pytest
from langfuse import Langfuse
from langfuse.client import DatasetStatus
from langfuse.llama_index import LlamaIndexInstrumentor
from ragas import EvaluationDataset
from ragas.integrations.llama_index import evaluate
from ragas.metrics import ContextUtilization, answer_relevancy, faithfulness

from src.agent import PsyAgent
from src.tools.search_tools import SearchTools

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/rag"))


@pytest.fixture(scope="module")
def search_tools():
    agent = PsyAgent(
        knowledge_base_folder=f"{RESOURCE_FOLDER}/pdf", web_search_enabled=False, rag_search_enabled=False, debug=True
    )
    return SearchTools(
        rag_model=agent.rag_model,
        embedding_model=agent.embedding_model,
        knowledge_base_folder=f"{RESOURCE_FOLDER}/pdf",
    )


@pytest.fixture(scope="module")
def query_engines(search_tools):
    return {
        "basic_engine": search_tools.query_engine,
    }


@pytest.fixture(scope="module")
def langfuse_client():
    return Langfuse()


def test_query_engine(search_tools, query_engines, langfuse_client):
    dataset = langfuse_client.get_dataset("agent_rag_test_dataset")
    instrumentor = LlamaIndexInstrumentor()

    for item in filter(lambda item: item.status == DatasetStatus.ACTIVE, dataset.items):
        trace_id = str(uuid.uuid4())

        # TODO substitute everything with instrumentor
        with instrumentor.observe(trace_id=trace_id), item.observe(run_name="rag_eval", trace_id=trace_id):
            metrics = [
                faithfulness,
                answer_relevancy,
                ContextUtilization(),
            ]

            result = evaluate(
                query_engine=query_engines["basic_engine"],
                metrics=metrics,
                dataset=EvaluationDataset.from_list([{"user_input": item.input}]),
                llm=search_tools.rag_model,
                embeddings=search_tools.embedding_model,
                show_progress=True,
            )

            langfuse_client.score(trace_id=trace_id, name="faithfulness", value=result.scores[0]["faithfulness"])

            langfuse_client.score(
                trace_id=trace_id,
                name="answer_relevancy",
                value=result.scores[0]["answer_relevancy"],
            )

            langfuse_client.score(
                trace_id=trace_id,
                name="context_precision_no_ref",
                value=result.scores[0]["context_utilization"],
            )
    langfuse_client.flush()
