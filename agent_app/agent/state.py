"""LangGraph agent state definition."""

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class AgentState(TypedDict, total=False):
    """State for the LakeRCM document intelligence agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    conversation_id: str
    # Rolling summary of older turns that were trimmed out of the window.
    # The pre-model hook prepends this to the prompt so the model still sees
    # relevant context even after trim_messages drops the raw transcript.
    summary: str
    # Required by create_react_agent for recursion-limit enforcement.
    remaining_steps: RemainingSteps
    # MLflow trace id for the *just-completed* model call. Set by the
    # post-model hook so the AG-UI path can surface it to the frontend via
    # STATE_DELTA — the React side reads it via useCoAgent and binds the
    # thumbs feedback UI to it. The legacy /responses path emits trace_id as
    # an explicit SSE event and does not depend on this field.
    last_trace_id: str
