from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time
from functools import wraps
import time
from prometheus_client import Counter, Histogram, Gauge
from contextlib import ContextDecorator

# Request metrics
REQUEST_COUNT = Counter(
    "psy_agent_requests_total",
    "Total number of requests to the agent",
    ["action"],  # action can be clarify, knowledge_retrieval, question_answering etc.
)

REQUEST_LATENCY = Histogram("psy_agent_request_latency_seconds", "Request latency in seconds", ["action"])

# Error metrics
ERROR_COUNT = Counter(
    "psy_agent_errors_total",
    "Total number of errors",
    ["type"],  # type can be rag_error, web_error, llm_error etc.
)

# User feedback metrics
USER_FEEDBACK = Counter(
    "psy_agent_user_feedback_total",
    "User feedback counts",
    ["type"],  # type can be thumbs_up or thumbs_down
)

AGENT_RESPONSE_TIME = Histogram(
    "psy_agent_response_time_seconds",
    "Agent response generation time in seconds",
    ["node"],  # node can be any node in the graph
)


class TimerError(Exception):
    """Custom exception for timer errors"""


class timer(ContextDecorator):
    """Timer class that can be used as a decorator or context manager"""

    def __init__(self, name=None, metric=None):
        self.name = name
        self.metric = metric
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        duration = time.perf_counter() - self._start_time
        if self.metric:
            self.metric.labels(self.name).observe(duration)
        return False
