import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler
from langfuse.client import Langfuse
from prometheus_client import start_http_server

from agent import PsyAgent
from src.metrics.metrics import AGENT_RESPONSE_TIME, REQUEST_LATENCY, USER_FEEDBACK

load_dotenv()

LANGFUSE_MONITORING = os.getenv("LANGFUSE_MONITORING", "false")
PROMETHEUS_GRAFANA_MONITORING = os.getenv("PROMETHEUS_GRAFANA_MONITORING", "false")

st.title("Psy AI")


@st.cache_resource
def get_langfuse():
    return Langfuse()


def start_prometheus_server():
    if "server_started" not in st.session_state and PROMETHEUS_GRAFANA_MONITORING == "true":
        start_http_server(8001)
        st.session_state.server_started = True


def handle_feedback(msg_idx, msg_langfuse_trace_id):
    st.session_state.feedback[msg_idx] = "thumbs_up" if st.session_state[f"feedback_{msg_idx}"] else "thumbs_down"
    USER_FEEDBACK.labels(st.session_state.feedback[msg_idx]).inc()

    if LANGFUSE_MONITORING == "true":
        get_langfuse().score(
            trace_id=msg_langfuse_trace_id,
            name="helpfulness",
            value=st.session_state[f"feedback_{msg_idx}"],
            data_type="BOOLEAN",
        )


def render_feedback_buttons(msg_idx, response_time, msg_langfuse_trace_id):
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    feedback_state = st.session_state.feedback.get(msg_idx, None)

    st.feedback(
        "thumbs",
        key=f"feedback_{msg_idx}",
        disabled=feedback_state is not None,
        on_change=handle_feedback,
        args=[msg_idx, msg_langfuse_trace_id],
    )  # type: ignore


def output_agent_response(user_input, final_response, session_messages, input_start_time, msg_langfuse_trace_id):
    with st.chat_message("assistant"):
        response_time = time.time() - input_start_time
        AGENT_RESPONSE_TIME.labels("total").observe(response_time)

        agent_response = final_response["messages"][-1].content
        st.markdown(agent_response)

        msg_idx = len(session_messages)
        render_feedback_buttons(msg_idx, response_time, msg_langfuse_trace_id)

    session_messages.append(
        {
            "role": "assistant",
            "content": agent_response,
            "response_time": response_time,
            "msg_langfuse_trace_id": msg_langfuse_trace_id,
        },
    )


def get_config():
    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "1"}}
    config = st.session_state.config

    predefined_run_id = str(uuid.uuid4())
    config["run_id"] = predefined_run_id

    if LANGFUSE_MONITORING == "true" and "callbacks" not in config:
        session_id = str(uuid.uuid4())
        langfuse_handler = CallbackHandler(session_id=f"session_{session_id}")
        config["callbacks"] = [langfuse_handler]

    return config


def render_message_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.agent = PsyAgent(debug=True)

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_feedback_buttons(
                    idx,
                    message.get("response_time", 0),
                    message.get("msg_langfuse_trace_id", None),
                )


def run_agent():
    session_messages = st.session_state.messages
    session_agent = st.session_state.agent
    config = get_config()

    if user_input := st.chat_input("How can I help you?"):
        input_start_time = time.time()

        with st.chat_message("user"):
            st.markdown(user_input)
        session_messages.append({"role": "user", "content": user_input})

        user_prompt = HumanMessage(content=user_input)

        if "last_node" not in session_agent.graph.get_state(config).values:
            with st.spinner("Thinking..."):
                REQUEST_LATENCY.labels("total").observe(time.time() - input_start_time)

                session_agent.graph.update_state(
                    config,
                    {
                        "request": "",
                        "action": "",
                        "last_node": "",
                        "tool_usage_counter": 0,
                    },
                )

                final_response = None
                for event in session_agent.graph.stream({"messages": [user_prompt]}, config, stream_mode="values"):
                    final_response = event

                output_agent_response(
                    user_input,
                    final_response,
                    session_messages,
                    input_start_time,
                    config.get("run_id", None),
                )

        else:
            last_node = session_agent.graph.get_state(config).values["last_node"]
            session_agent.graph.update_state(config, {"messages": [user_prompt]}, as_node=last_node)

            with st.spinner("Thinking..."):
                REQUEST_LATENCY.labels("total").observe(time.time() - input_start_time)

                final_response = None
                for event in session_agent.graph.stream(None, config, stream_mode="values"):
                    final_response = event

                output_agent_response(
                    user_input,
                    final_response,
                    session_messages,
                    input_start_time,
                    config.get("run_id", None),
                )


start_prometheus_server()
render_message_history()
run_agent()
