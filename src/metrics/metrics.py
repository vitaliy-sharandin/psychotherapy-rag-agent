import time
from contextlib import ContextDecorator

from prometheus_client import Counter, Histogram, REGISTRY


def is_metric_registered(name: str) -> bool:
    return name in REGISTRY._names_to_collectors  # underscore = private, but widely used

class MetricsSingleton:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_metrics()
            MetricsSingleton._initialized = True

    def _setup_metrics(self):
        def reg_counter(name, desc, labels):
            if not is_metric_registered(name):
                return Counter(name, desc, labels)
            return REGISTRY._names_to_collectors[name]

        def reg_histogram(name, desc, labels):
            if not is_metric_registered(name):
                return Histogram(name, desc, labels)
            return REGISTRY._names_to_collectors[name]

        self.REQUEST_COUNT = reg_counter(
            "psy_agent_requests_total",
            "Total number of requests to the agent",
            ["action"],
        )

        self.REQUEST_LATENCY = reg_histogram(
            "psy_agent_request_latency_seconds",
            "Request latency in seconds",
            ["action"],
        )

        self.AGENT_RESPONSE_TIME = reg_histogram(
            "psy_agent_response_time_seconds",
            "Agent response generation time in seconds",
            ["node"],
        )

        self.ERROR_COUNT = reg_counter(
            "psy_agent_errors_total",
            "Total number of errors",
            ["type"],
        )

        self.USER_FEEDBACK = reg_counter(
            "psy_agent_user_feedback_total",
            "User feedback counts",
            ["type"],
        )

        self.TOOL_USAGE_COUNT = reg_counter(
            "psy_agent_tool_usage_total",
            "Total number of tool calls",
            ["tool_name"],
        )

        self.TOOL_EXECUTION_TIME = reg_histogram(
            "psy_agent_tool_execution_seconds",
            "Tool execution time in seconds",
            ["tool_name"],
        )

# Initialize singleton instance
_metrics = MetricsSingleton()

# Export metrics for easy import
REQUEST_COUNT = _metrics.REQUEST_COUNT
REQUEST_LATENCY = _metrics.REQUEST_LATENCY
AGENT_RESPONSE_TIME = _metrics.AGENT_RESPONSE_TIME
ERROR_COUNT = _metrics.ERROR_COUNT
USER_FEEDBACK = _metrics.USER_FEEDBACK
TOOL_USAGE_COUNT = _metrics.TOOL_USAGE_COUNT
TOOL_EXECUTION_TIME = _metrics.TOOL_EXECUTION_TIME


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
