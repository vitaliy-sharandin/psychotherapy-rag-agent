import streamlit as st
from agent import PsyAgent
from langchain_core.messages import HumanMessage
from metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT, USER_FEEDBACK, AGENT_RESPONSE_TIME
import time
from prometheus_client import start_http_server

start_http_server(8001)

st.title("Psy AI")

config = {"configurable": {"thread_id": "1"}}

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent = PsyAgent(config, knowledge_retrieval=False, debug=True)

if "feedback" not in st.session_state:
    st.session_state.feedback = {}


def log_response_feedback(session_messages, user_input, agent_response, response_time):
    col1, col2, col3 = st.columns([1, 1, 3])
    msg_idx = len(session_messages)

    with col1:
        if st.button("👍", key=f"thumbs_up_{msg_idx}"):
            st.session_state.feedback[msg_idx] = "positive"
            USER_FEEDBACK.labels("thumbs_up").inc()
            # TODO FEEDBACK IS NOT LOGGED HERE TO PROMETHEUS!!!

            # TODO log feeback with message to Arize or whatever

    with col2:
        if st.button("👎", key=f"thumbs_down_{msg_idx}"):
            st.session_state.feedback[msg_idx] = "negative"
            USER_FEEDBACK.labels("thumbs_down").inc()
            # TODO FEEDBACK IS NOT LOGGED HERE TO PROMETHEUS!!!

            # TODO log feeback with message to Arize or whatever

    with st.expander("View metrics"):
        st.metric("Response time", f"{response_time:0.2f}s")
        feedback = st.session_state.feedback.get(msg_idx, "No feedback yet")
        st.text(f"Feedback: {feedback}")


def output_agent_response(user_input, final_response, session_messages):
    with st.chat_message("assistant"):
        response_time = time.time() - input_start_time
        AGENT_RESPONSE_TIME.labels("total").observe(response_time)

        agent_response = final_response["messages"][-1].content
        st.markdown(agent_response)

        log_response_feedback(session_messages, user_input, agent_response, response_time)

    session_messages.append({"role": "assistant", "content": agent_response, "response_time": response_time})


session_agent = st.session_state.agent
session_messages = st.session_state.messages
session_feedback = st.session_state.feedback

for idx, message in enumerate(session_messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add feedback buttons and metrics for assistant messages
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 3])

            # Feedback buttons
            with col1:
                if st.button("👍", key=f"thumbs_up_{idx}"):
                    st.session_state.feedback[idx] = "positive"

            with col2:
                if st.button("👎", key=f"thumbs_down_{idx}"):
                    st.session_state.feedback[idx] = "negative"

            # Metrics expander
            with st.expander("View metrics"):
                st.metric("Response time", f"{message.get('response_time', 0):.2f}s")
                feedback = st.session_state.feedback.get(idx, "No feedback yet")
                st.text(f"Feedback: {feedback}")

if user_input := st.chat_input("How can I help you?"):
    input_start_time = time.time()

    with st.chat_message("user"):
        st.markdown(user_input)
    session_messages.append({"role": "user", "content": user_input})

    user_prompt = HumanMessage(content=user_input)

    if "last_node" not in session_agent.graph.get_state(config).values:
        with st.spinner("Thinking..."):
            REQUEST_LATENCY.labels("total").observe(time.time() - input_start_time)

            for event in session_agent.graph.stream({"messages": [user_prompt]}, config, stream_mode="values"):
                final_response = event

            output_agent_response(user_input, final_response, session_messages)

    else:
        last_node = session_agent.graph.get_state(config).values["last_node"]
        session_agent.graph.update_state(config, {"messages": [user_prompt]}, as_node=last_node)

        with st.spinner("Thinking..."):
            for event in session_agent.graph.stream(None, config, stream_mode="values"):
                final_response = event

            output_agent_response(user_input, final_response, session_messages)
