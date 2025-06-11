import os
from datetime import datetime
from time import sleep

import pytest
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langfuse import Langfuse
from langfuse.client import DatasetStatus

from src.agent import PsyAgent, run_agent_with_messages
from src.prompts.prompts import ACTION_DETECTION_PROMPT

from .mocked_tools import MockedTool

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/rag"))


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


def test_agent_router(agent, langfuse_client):
    dataset = langfuse_client.get_dataset("agent_router_test_dataset")

    # TODO perform for more than one item - unarchive in Langfuse
    for test_item in filter(lambda item: item.status == DatasetStatus.ACTIVE, dataset.items):
        trace = langfuse_client.trace()
        langfuse_handler_trace = trace.get_langchain_handler(update_parent=True)
        run_agent_with_messages(
            [test_item.input],
            {"configurable": {"thread_id": "1"}, "callbacks": [langfuse_handler_trace]},
        )

        sleep(7)

        logged_trace = langfuse_client.fetch_trace(trace.trace_id).data

        model = OllamaChatDeepEvalTestModel(model=agent.text_generation_model)

        lowest_tool_score, highest_tool_score = score_tool_calling(test_item, model, logged_trace)
        action_selection_score = score_final_router_decision(test_item, model, logged_trace)

        langfuse_client.score(trace_id=trace.trace_id, name="lowest_tool", value=lowest_tool_score)

        langfuse_client.score(trace_id=trace.trace_id, name="highest_tool", value=highest_tool_score)

        langfuse_client.score(trace_id=trace.trace_id, name="action_selection", value=action_selection_score)

        test_item.link(
            trace,
            run_name="agent_router_test",
        )
    langfuse_client.flush()


def score_tool_calling(item, model, logged_trace):
    tool_calls = None
    action_selector_observation = None

    for observation in sorted(
        logged_trace.observations,
        key=lambda x: datetime.strptime(x.createdAt, "%Y-%m-%dT%H:%M:%S.%fZ"),
    ):
        if observation.name == "action_selector":
            action_selector_observation = observation
            tool_calls = action_selector_observation.output["messages"][0]["tool_calls"]
            break

    tool_tests = []
    tool_relevancy = GEval(
        name="Tool choice and argument relevancy",
        evaluation_steps=[
            "The input is user request.",
            "The actual output is combination of tool name with corresponding arguments selected to address request.",
            "Differentiate what is meant to be used in tool and what are general user instructions not correlated with potential tool usage in initial request.",
            "Determine whether actual output tool its arguments are well matching the input user request.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )

    for tool_call in tool_calls:
        test_case = LLMTestCase(
            input=item.input,
            actual_output=f"Tool Call: {tool_call}",
        )
        tool_tests.append(test_case)

    tool_eval_results = evaluate(test_cases=tool_tests, metrics=[tool_relevancy])

    lowest_tool_score = min(test_result.metrics_data[0].score for test_result in tool_eval_results.test_results)

    highest_tool_score = max(test_result.metrics_data[0].score for test_result in tool_eval_results.test_results)

    return lowest_tool_score, highest_tool_score


def score_final_router_decision(item, model, logged_trace):
    action_detector_input = None
    action_detector_output = None

    for observation in sorted(
        logged_trace.observations,
        key=lambda x: datetime.strptime(x.createdAt, "%Y-%m-%dT%H:%M:%S.%fZ"),
        reverse=True,
    ):
        if observation.name == "action_selector":
            action_selector_observation = observation
            action_detector_input = str(action_selector_observation.input)
            action_detector_output = str(action_selector_observation.output)
            break

    router_final_decision = GEval(
        name="Router node final decision",
        evaluation_steps=[
            "The input is user request with messages history which shows decisions taken by router and other agent nodes. Also, it has a prompt for router decision making.",
            "The actual output is the final output made by router node.",
            "Given messages history, tool usage, and action detection prompt evaluate the final decision of router.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )

    test_case = LLMTestCase(
        input=f"User request: {item.input}, Messages history: {action_detector_input}, Action Detection prompt: {ACTION_DETECTION_PROMPT}",
        actual_output=action_detector_output,
    )

    return router_final_decision.measure(test_case)


class OllamaChatDeepEvalTestModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "OllamaChat model"
