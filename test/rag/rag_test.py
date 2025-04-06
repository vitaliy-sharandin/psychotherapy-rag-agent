import os
import sys

import pytest
import uuid
from ragas import EvaluationDataset
import yaml
from langfuse.llama_index import LlamaIndexInstrumentor
from langfuse import Langfuse
from src.agent import PsyAgent
from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization, context_recall
from ragas.integrations.llama_index import evaluate
from langfuse.llama_index import LlamaIndexInstrumentor

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/rag"))


@pytest.fixture(scope="module")
def agent():
    config = {"configurable": {"thread_id": "1"}}
    return PsyAgent(config, knowledge_base_folder=f"{RESOURCE_FOLDER}/pdf", web_search_enabled=False, debug=True)


@pytest.fixture(scope="module")
def query_engines(agent):
    return {
        # "automerging_engine": agent.automerging_query_engine,
        # "rerank_engine": agent.rerank_query_engine,
        "basic_engine": agent.query_engine,
    }


@pytest.fixture(scope="module")
def langfuse_client():
    return Langfuse()


def test_query_engine(agent, query_engines, langfuse_client):
    dataset = langfuse_client.get_dataset("eval_dataset")
    instrumentor = LlamaIndexInstrumentor()

    for item in [dataset.items[0]]:

        trace_id = str(uuid.uuid4())

        # TODO substitute everything with instrumentor
        with instrumentor.observe(trace_id=trace_id), item.observe(run_name='rag_eval', trace_id=trace_id):
            metrics = [
                faithfulness,
                answer_relevancy,
                ContextUtilization(),
            ]

            result = evaluate(
                query_engine=query_engines["basic_engine"],
                metrics=metrics,
                dataset=EvaluationDataset.from_list([{"user_input": item.input}]),
                llm=agent.rag_model,
                embeddings=agent.embedding_model,
                show_progress=True,
            )

            langfuse_client.score(trace_id=trace_id, name="faithfulness", value=result.scores[0]["faithfulness"])

            langfuse_client.score(trace_id=trace_id, name="answer_relevancy", value=result.scores[0]["answer_relevancy"])

            langfuse_client.score(trace_id=trace_id, name="context_precision_no_ref", value=result.scores[0]["context_utilization"])
    langfuse_client.flush()
