import os
import sys
import uuid
from datetime import datetime
from time import sleep

import pytest
import yaml
from deepeval import assert_test, evaluate
from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from langfuse import Langfuse
from langfuse.client import DatasetStatus
from langfuse.llama_index import LlamaIndexInstrumentor
from ragas import EvaluationDataset
from ragas.metrics import ContextUtilization, answer_relevancy, context_recall, faithfulness

from src.agent import PsyAgent
from src.prompts import ACTION_DETECTION_PROMPT

from .mocked_tools import MockedTool

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/rag"))

NODE_NAMES = [
    "action_selector",
    "clarify",
    "rag_search",
    "web_search",
    "question_answering",
]


@pytest.fixture(scope="module")
def agent():
    mocked_tool = MockedTool()
    agent = PsyAgent(
        knowledge_base_folder=f"{RESOURCE_FOLDER}/pdf",
        web_search_enabled=False,
        tools=[mocked_tool.rag_search],
        debug=True,
    )
    return agent


@pytest.fixture(scope="module")
def langfuse_client():
    return Langfuse()


def test_agent_trajectory_stability(agent, langfuse_client):
    dataset = langfuse_client.get_dataset("agent_trajectory_test_dataset")

    total_runs = 5
    test_items = list(filter(lambda item: item.status == DatasetStatus.ACTIVE, dataset.items))

    for test_item in test_items:
        step_counts = []
        last_trace = None

        for run_index in range(total_runs):
            trace = langfuse_client.trace()
            langfuse_handler_trace = trace.get_langchain_handler(update_parent=True)

            agent.run_agent_with_messages(
                [test_item.input],
                {
                    "configurable": {"thread_id": str(uuid.uuid4())},
                    "callbacks": [langfuse_handler_trace],
                },
            )

            sleep(7)

            logged_trace = langfuse_client.fetch_trace(trace.trace_id).data

            steps = get_amount_of_steps(logged_trace)
            step_counts.append(steps)

            test_item.link(trace, run_name="agent_trajectory_test")

            last_trace = trace

        S_optimal = min(step_counts)
        convergence_score = sum(min(1, S_optimal / s) for s in step_counts) / total_runs

        langfuse_client.score(trace_id=last_trace.trace_id, name="convergence_score", value=convergence_score)

    langfuse_client.flush()

def test_agent_trajectory_generalization(agent, langfuse_client):
    #TODO create generalization test for similar queries to see if agent takes optimal paths
    ...

def get_amount_of_steps(logged_trace):
    steps_amount = 0

    for observation in logged_trace.observations:
        if observation.type == "SPAN" and observation.name in NODE_NAMES:
            steps_amount += 1

    return steps_amount
