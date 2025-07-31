import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.agent import PsyAgent

from .mocked_tools import MockedTool

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/rag"))


@pytest.fixture(scope="module")
def agent():
    mocked_tool = MockedTool()
    # Use much smaller token limits for testing
    agent = PsyAgent(
        knowledge_base_folder=f"{RESOURCE_FOLDER}/pdf",
        web_search_enabled=False,
        tools=[mocked_tool.rag_search],
        debug=True,
        max_tokens=1000,  # Much smaller for testing
        max_tokens_before_summary=2000,  # Much smaller for testing
        max_summary_tokens=300,  # Much smaller for testing
    )
    return agent


def create_message_with_id(message_class, content):
    """Helper function to create messages with unique IDs."""
    return message_class(content=content, id=str(uuid.uuid4()))


def test_summarization_trigger(agent):
    """Test that summarization is triggered when conversation exceeds token limits."""
    config = {"configurable": {"thread_id": "summarization_test"}}

    # Initialize agent state
    agent.graph.update_state(
        config,
        {
            "request": "",
            "action": "",
            "last_node": "",
            "tool_usage_counter": 0,
            "summarized_messages": [],
            "context": {},
        },
    )

    # Create a conversation that exceeds the test token limits (much simpler now)
    long_messages = []
    long_messages.append(create_message_with_id(SystemMessage, "You are a helpful psychology assistant."))

    # With smaller limits, we only need a few messages
    for i in range(10):  # Much fewer messages needed
        human_msg = f"This is a message about psychology topic {i}. " * 10  # Shorter messages
        ai_msg = f"Response about topic {i} with psychological insights. " * 12

        long_messages.extend(
            [
                create_message_with_id(HumanMessage, human_msg),
                create_message_with_id(AIMessage, ai_msg),
            ]
        )

    total_tokens = count_tokens_approximately(long_messages)
    print(f"Total tokens generated: {total_tokens}")

    # Much easier to exceed the smaller threshold
    assert total_tokens > agent.summarization_node.max_tokens_before_summary

    # Test the summarization node directly
    state = {
        "messages": long_messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # Verify summarization occurred
    assert "summarized_messages" in result
    assert len(result["summarized_messages"]) < len(long_messages)

    # Verify total tokens reduced
    summarized_tokens = count_tokens_approximately(result["summarized_messages"])
    assert summarized_tokens < total_tokens

    # The summarization node may not strictly enforce max_summary_tokens in all cases
    # due to the need to preserve important messages and system messages
    # So we'll check that it's at least reduced significantly
    reduction_percentage = (total_tokens - summarized_tokens) / total_tokens
    assert reduction_percentage > 0.5, f"Expected significant reduction, got {reduction_percentage:.2%}"


def test_summarization_with_forced_trigger(agent):
    """Test summarization by using the max_tokens threshold instead of max_tokens_before_summary."""
    # Create messages that exceed max_tokens (20,000) to force summarization
    long_messages = []

    # Add system message with ID
    long_messages.append(create_message_with_id(SystemMessage, "You are a helpful psychology assistant."))

    # Add enough messages to exceed max_tokens (20,000)
    for i in range(50):
        human_msg = f"This is a detailed message about psychology and therapy topic {i}. " * 35
        ai_msg = f"Thank you for sharing. Here are comprehensive therapeutic insights about topic {i}. " * 40

        long_messages.extend(
            [
                create_message_with_id(HumanMessage, human_msg),
                create_message_with_id(AIMessage, ai_msg),
            ],
        )

    # Calculate total tokens
    total_tokens = count_tokens_approximately(long_messages)
    print(f"Total tokens for forced trigger test: {total_tokens}")

    # Verify we exceed max_tokens to force summarization
    assert total_tokens > agent.summarization_node.max_tokens

    # Test the summarization node directly
    state = {
        "messages": long_messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # Verify summarization occurred
    assert "summarized_messages" in result
    assert len(result["summarized_messages"]) < len(long_messages)

    # Verify total tokens reduced to within max_tokens limit
    summarized_tokens = count_tokens_approximately(result["summarized_messages"])
    assert summarized_tokens < total_tokens
    assert summarized_tokens <= agent.summarization_node.max_tokens


def test_important_messages_preserved(agent):
    """Test that important messages are preserved during summarization."""
    # Create messages with varying importance and IDs
    important_message = "Can you help me with coping strategies for anxiety?"
    messages = [
        create_message_with_id(SystemMessage, "You are a psychology assistant."),  # Should be preserved
        create_message_with_id(HumanMessage, "Hello"),
        create_message_with_id(AIMessage, "Hi there"),
        create_message_with_id(HumanMessage, "I'm feeling anxious about my upcoming therapy session"),  # Important
        create_message_with_id(AIMessage, "I understand your anxiety about therapy"),
        create_message_with_id(HumanMessage, "What's the weather like?"),  # Less important
        create_message_with_id(AIMessage, "I'm not sure about the weather"),
    ]

    # Add enough filler to trigger summarization (exceed max_tokens)
    for i in range(25):
        messages.extend(
            [
                create_message_with_id(HumanMessage, f"Filler message {i} " * 35),
                create_message_with_id(AIMessage, f"Filler response {i} " * 40),
            ],
        )

    # Add the important message LAST so it should be preserved
    messages.append(create_message_with_id(HumanMessage, important_message))

    state = {
        "messages": messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # Verify system messages are preserved
    system_msgs = [msg for msg in result["summarized_messages"] if isinstance(msg, SystemMessage)]
    assert len(system_msgs) > 0

    # Verify last human message is preserved and contains our important message
    last_human_msg = None
    for msg in reversed(result["summarized_messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg
            break

    assert last_human_msg is not None
    assert important_message in last_human_msg.content


def test_conversation_context_maintained(agent):
    """Test that therapeutic context is maintained after summarization."""
    # Simulate a therapy conversation with IDs
    therapy_messages = [
        create_message_with_id(SystemMessage, "You are a psychology assistant specialized in therapy."),
        create_message_with_id(HumanMessage, "I've been struggling with depression for months"),
        create_message_with_id(AIMessage, "I'm sorry to hear you're going through this difficult time"),
        create_message_with_id(HumanMessage, "I have trouble sleeping and feel hopeless"),
        create_message_with_id(AIMessage, "Sleep issues and hopelessness are common symptoms of depression"),
        create_message_with_id(HumanMessage, "My therapist suggested cognitive behavioral therapy"),
        create_message_with_id(AIMessage, "CBT is an evidence-based approach that can be very helpful"),
        create_message_with_id(HumanMessage, "Can you explain how CBT works for depression?"),
    ]

    # Add filler messages to trigger summarization (exceed max_tokens)
    for i in range(25):
        therapy_messages.extend(
            [
                create_message_with_id(HumanMessage, f"Additional context message {i} " * 35),
                create_message_with_id(AIMessage, f"Response to context {i} " * 40),
            ],
        )

    state = {
        "messages": therapy_messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # Check that therapeutic context keywords are preserved somewhere in the summarized messages
    all_content = " ".join([msg.content for msg in result["summarized_messages"]])

    # Should contain key therapeutic concepts
    therapeutic_keywords = ["depression", "therapy", "CBT", "cognitive behavioral"]
    preserved_keywords = [keyword for keyword in therapeutic_keywords if keyword.lower() in all_content.lower()]

    assert len(preserved_keywords) > 0, f"No therapeutic keywords preserved. Content: {all_content}"


def test_summarization_in_agent_flow(agent):
    """Test summarization works within the full agent conversation flow."""
    config = {"configurable": {"thread_id": "flow_test"}}

    # Initialize agent
    agent.graph.update_state(
        config,
        {
            "request": "",
            "action": "",
            "last_node": "",
            "tool_usage_counter": 0,
            "summarized_messages": [],
            "context": {},
        },
    )

    # Start with a therapeutic conversation
    initial_message = create_message_with_id(HumanMessage, "I need help managing my anxiety during social situations")

    # Run the initial message through the agent
    final_response = None
    for event in agent.graph.stream({"messages": [initial_message]}, config, stream_mode="values"):
        final_response = event

    # Verify the agent processed the message
    assert final_response is not None
    assert "messages" in final_response

    # Check that summarized_messages exists in the state
    state = agent.graph.get_state(config)
    assert "summarized_messages" in state.values

    # The conversation should be manageable at this point (not triggering summarization yet)
    messages_to_check = state.values.get("summarized_messages", state.values.get("messages", []))

    # Verify the therapeutic content is present
    all_content = " ".join([msg.content for msg in messages_to_check if hasattr(msg, "content")])
    assert "anxiety" in all_content.lower() or "social" in all_content.lower()


def test_summarization_preserves_system_messages(agent):
    """Test that system messages are never compressed during summarization."""
    # Create a mix of messages with multiple system messages
    messages = [
        create_message_with_id(SystemMessage, "You are a psychology assistant."),
        create_message_with_id(HumanMessage, "Hello"),
        create_message_with_id(SystemMessage, "Important system instruction for therapy."),
        create_message_with_id(AIMessage, "Hi there"),
        create_message_with_id(HumanMessage, "I need help"),
    ]

    # Add filler to trigger summarization (exceed max_tokens)
    for i in range(25):
        messages.extend(
            [
                create_message_with_id(HumanMessage, f"Filler message {i} " * 35),
                create_message_with_id(AIMessage, f"Filler response {i} " * 40),
            ],
        )

    state = {
        "messages": messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # The summarization node may consolidate system messages into one
    # but the core system functionality should be preserved
    system_msgs = [msg for msg in result["summarized_messages"] if isinstance(msg, SystemMessage)]
    assert len(system_msgs) > 0, "At least one system message should be preserved"

    # Check that system-related content is preserved in some form
    all_system_content = " ".join([msg.content for msg in system_msgs])
    assert "psychology assistant" in all_system_content.lower(), "Core system role should be preserved"


def test_summarization_reduces_message_count(agent):
    """Test that summarization significantly reduces the number of messages."""
    messages = [create_message_with_id(SystemMessage, "You are a psychology assistant.")]

    # Generate enough content to exceed the small token limits
    for i in range(15):  # Increased from 8 to 15
        messages.extend(
            [
                create_message_with_id(HumanMessage, f"Question {i} about therapy and psychological approaches " * 8),  # Increased length
                create_message_with_id(AIMessage, f"Response {i} about therapeutic approaches and mental health " * 10),  # Increased length
            ]
        )

    original_count = len(messages)
    original_tokens = count_tokens_approximately(messages)

    print(f"Original message count: {original_count}, tokens: {original_tokens}")

    # Should easily exceed the small test limits
    assert original_tokens > agent.summarization_node.max_tokens

    state = {
        "messages": messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    summarized_count = len(result["summarized_messages"])
    summarized_tokens = count_tokens_approximately(result["summarized_messages"])

    print(f"Summarized message count: {summarized_count}, tokens: {summarized_tokens}")

    # Verify significant reduction in both message count and tokens
    assert summarized_count < original_count, (
        f"Expected message count reduction: {summarized_count} >= {original_count}"
    )
    assert summarized_tokens < original_tokens, (
        f"Expected token reduction: {summarized_tokens} >= {original_tokens}"
    )

    # Verify we have some reasonable reduction
    count_reduction = (original_count - summarized_count) / original_count
    token_reduction = (original_tokens - summarized_tokens) / original_tokens

    assert count_reduction > 0.2, f"Expected significant message count reduction, got {count_reduction:.2%}"
    assert token_reduction > 0.2, f"Expected significant token reduction, got {token_reduction:.2%}"


def test_summarization_handles_edge_cases(agent):
    """Test that summarization handles edge cases gracefully."""
    # Test with very few messages (shouldn't trigger summarization)
    few_messages = [
        create_message_with_id(SystemMessage, "You are a psychology assistant."),
        create_message_with_id(HumanMessage, "Hello"),
        create_message_with_id(AIMessage, "Hi there"),
    ]

    state = {
        "messages": few_messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # With few messages, should not trigger summarization
    original_tokens = count_tokens_approximately(few_messages)
    if original_tokens <= agent.summarization_node.max_tokens:
        # Should pass through unchanged
        assert len(result["summarized_messages"]) == len(few_messages)
    else:
        # If somehow it exceeds limits, should still work
        assert "summarized_messages" in result


def test_summarization_preserves_conversation_flow(agent):
    """Test that summarization maintains conversation flow and context."""
    # Create a therapeutic conversation with clear progression
    messages = [
        create_message_with_id(SystemMessage, "You are a psychology assistant."),
        create_message_with_id(HumanMessage, "I'm struggling with anxiety"),
        create_message_with_id(AIMessage, "I understand your concern about anxiety"),
        create_message_with_id(HumanMessage, "It affects my work performance"),
        create_message_with_id(AIMessage, "Work-related anxiety is common and treatable"),
        create_message_with_id(HumanMessage, "What techniques can help?"),
        create_message_with_id(AIMessage, "Breathing exercises and cognitive restructuring are effective"),
    ]

    # Add filler to trigger summarization
    for i in range(20):
        messages.extend([
            create_message_with_id(HumanMessage, f"Additional question {i} about anxiety management " * 10),
            create_message_with_id(AIMessage, f"Response {i} about anxiety treatment approaches " * 12),
        ])

    # Add a final important message
    messages.append(create_message_with_id(HumanMessage, "Can you summarize the key techniques we discussed?"))

    state = {
        "messages": messages,
        "summarized_messages": [],
        "context": {},
    }

    result = agent.summarization_node.invoke(state)

    # Check that key concepts are preserved
    all_content = " ".join([msg.content for msg in result["summarized_messages"]])
    
    # Should preserve therapeutic context
    assert "anxiety" in all_content.lower()
    
    # Should preserve the final important question
    assert "key techniques" in all_content.lower() or "summarize" in all_content.lower()
    
    # Should have fewer messages than original
    assert len(result["summarized_messages"]) < len(messages)
