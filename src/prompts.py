import os
import pathlib

from langgraph.graph import END

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PSYCHOLOGY_AGENT_PROMPT = """You are a highly qualified and experienced psychologist, psychotherapist, and psychiatrist.
    Your role is to combine deep theoretical knowledge with practical therapeutic skills, adhering to professional, ethical, and clinical guidelines in every interaction."""

THERAPIST_POLICY_PROMPT = pathlib.Path(f"{SCRIPT_DIR}/resources/therapist-policy.txt").read_text()

ACTION_DETECTION_OPTIONS = {
    "clarify": "clarify",
    "tools": "tools",
    "question_answering": "question_answering",
}

KNOWLEDGE_RETRIEVAL_PROMPT = "If everything is clear, but you think some additional knowledge from local RAG db or web is needed to give best answer, return 'knowledge_retrieval'.\""

ACTION_DETECTION_PROMPT = """
    <INSTRUCTIONS_START>
    This is a router prompt. Here you can either return one word action from list below or use tools, you can also invoke both.
    Follow user instructions as closely as possible.

    Action options as content response:
    'clarify' - ask user to clarify his request, if you are not sure what he wants, unless user tells you he doesn't have additional info or  it's all you need to know.
    'question_answering' - if you are sure that you already have all necessary knowledge to answer question.
    Note: This is an action detection prompt, so respond only with lower-case one word options provided above or use tools!

    <INSTRUCTIONS_END>

    <USER_REQUEST_START>
    {prompt}
    <USER_REQUEST_END>
    """

CLARIFICATION_PROMPT = """<INSTRUCTIONS_START>
    The user\'s intent in his request could be interpreted in many ways.
    Ask user to specify exactly what he wants potentially giving him options.
    <INSTRUCTIONS_END>

    <USER_REQUEST_START>
    {prompt}
    <USER_REQUEST_END>"""

QUERIES_GENERATION_PROMPT = """<INSTRUCTIONS_START>Generate maximum 3 queries for RAG search and also maximum 3 queries for web search based on user request. Formulate queries in a way that they are most likely to return relevant information.<INSTRUCTIONS_END>"""

RAG_RENENERATION_PROMPT = """<INSTRUCTIONS_START>
    Following queries were generated for RAG based on user request, yet provided results which were not that relevant.
    Regenerate queries for RAG based on user request. Formulate queries in a way that they are most likely to return relevant information.
    <INSTRUCTIONS_END>

    <KNOWLEDGE_START>
    RAG queries: {rag}
    User request: {request}
    <KNOWLEDGE_END>"""

WEB_REGENERATION_PROMPT = """<INSTRUCTIONS_START>
    Following queries were generated for web search based on user request, yet provided results which were not that relevant.
    Regenerate queries for web search based on user request. Formulate queries in a way that they are most likely to return relevant information.
    <INSTRUCTIONS_END>

    <KNOWLEDGE_START>
    Web queries: {web}
    User request: {request}
    <KNOWLEDGE_END>"""

QUERIES_REGENERATION_PROMPT = """<INSTRUCTIONS_START>
    Following queries were generated for RAG and web search based on user request, yet provided results which were not that relevant.
    Regenerate queries for RAG and web search based on user request. Formulate queries in a way that they are most likely to return relevant information.
    <INSTRUCTIONS_END>

    <KNOWLEDGE_START>
    RAG queries: {rag}
    Web queries: {web}
    User request: {request}
    <KNOWLEDGE_END>"""

KNOWLEDGE_RELEVANCY_EVALUATION_PROMPT = """<INSTRUCTIONS_START>
    Evaluate if information retrieved from RAG and web search is relevant enough to user request.
    If both RAG and web are relevant, return \'none\'.
    If both RAG and web are irrelevant, return \'both\'.
    If RAG is irrelevant and web is relevant, return \'rag\'.
    If RAG is relevant and web is irrelevant, return \'web\'.
    Remember, respond with only one word!
    <INSTRUCTIONS_END>

    <KNOWLEDGE_START>
    User request: {request}
    RAG search results: {rag}
    Web search results: {web}
    <KNOWLEDGE_END>"""

KNOWLEDGE_SUMMARY_PROMPT = """<INSTRUCTIONS_START>
    Filter, order and summarize information retrieved from RAG and web search, so only information relevant to user request in conversation history context is left.
    <INSTRUCTIONS_END>

    <KNOWLEDGE_START>
    User request: {request}
    RAG search results: {rag}
    Web search results: {web}
    <KNOWLEDGE_END>"""

QUESTION_ANSWERING_PROMPT = """<INSTRUCTIONS_START>
    Answer user request according to policy.
    Take into account the conversation history as well.
    Talk to user in a natural manner, he doesn't need to know about your internal workings.
    <INSTRUCTIONS_END>

    User request: {request}
    """
