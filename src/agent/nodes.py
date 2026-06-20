# src/agents/nodes.py
"""
LangGraph node implementations for the financial analysis agent.
Each node represents a processing step in the analysis workflow.
"""
from __future__ import annotations

from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings

logger = structlog.get_logger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a senior financial analyst with 15 years of experience
in personal and corporate banking. You have access to:

1. A RAG system containing indexed bank statement data
2. Analytical tools for spending analysis, anomaly detection, and cash flow analysis

Your workflow:
1. Understand what the user is asking about their financial data
2. Decide which tools to use and in what order
3. Synthesize tool outputs into a comprehensive, actionable analysis
4. Provide recommendations where appropriate

Key principles:
- Be precise with numbers — always use exact figures from the data
- Provide context — raw numbers mean nothing without comparison
- Be actionable — tell the user what the numbers mean for their finances
- Flag concerns — proactively identify potential issues

Available tools handle: spending by category, spending over time, large transactions,
monthly summaries, recurring transactions, cash flow analysis, and anomaly detection."""


class FinancialAnalystNode:
    """
    Main analyst node that reasons about the query and orchestrates tool calls.
    Uses a ReAct-style reasoning loop.
    """

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=0.1,
        )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the current state and decide next action."""
        messages = state.get("messages", [])
        tools = state.get("tools", [])
        rag_context = state.get("rag_context", "")

        # Build the system message with RAG context
        system_content = ANALYST_SYSTEM_PROMPT
        if rag_context:
            system_content += f"\n\nRelevant document context:\n{rag_context}"

        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

        all_messages = [SystemMessage(content=system_content)] + messages
        response = llm_with_tools.invoke(all_messages)

        logger.info(
            "analyst_node_response",
            has_tool_calls=bool(response.tool_calls),
            content_length=len(response.content) if response.content else 0,
        )

        return {"messages": [response]}


class RetrievalNode:
    """Node that retrieves relevant context from the RAG system before analysis."""

    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant documents for the current query."""
        messages = state.get("messages", [])
        if not messages:
            return {"rag_context": ""}

        # Get the latest human message
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return {"rag_context": ""}

        try:
            result = self.rag_chain.retriever._retrieve(query)
            context = "\n\n".join([doc.page_content for doc in result])
            logger.info("retrieval_node", query=query[:50], docs=len(result))
            return {"rag_context": context, "retrieved_docs": result}
        except Exception as e:
            logger.error("retrieval_node_error", error=str(e))
            return {"rag_context": ""}


class ToolExecutionNode:
    """Node that executes tool calls and returns results."""

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute tool calls from the last AI message."""
        messages = state.get("messages", [])
        tools = state.get("tools", [])
        tool_map = {t.name: t for t in tools}

        if not messages:
            return {"messages": []}

        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}

        results = []
        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            logger.info("executing_tool", tool=tool_name, args=str(tool_args)[:100])

            if tool_name in tool_map:
                try:
                    output = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    output = f"Error executing {tool_name}: {str(e)}"
                    logger.error("tool_error", tool=tool_name, error=str(e))
            else:
                output = f"Unknown tool: {tool_name}"

            results.append({
                "role": "tool",
                "content": str(output),
                "tool_call_id": tool_id,
            })

        return {"messages": results}


class SynthesisNode:
    """
    Final synthesis node that produces the polished analysis output.
    Runs after all tools have been executed.
    """

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=0.2,
            max_tokens=settings.openai.max_tokens,
        )

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Synthesize all gathered information into a final response."""
        messages = state.get("messages", [])

        synthesis_prompt = SystemMessage(
            content="""You are producing the final analysis report. Take all the tool
outputs and context gathered and create a clear, structured financial analysis.

Format your response as:
1. **Summary** — One-paragraph overview
2. **Key Findings** — Bullet points of the most important insights
3. **Detailed Analysis** — In-depth breakdown with numbers
4. **Recommendations** — Actionable suggestions
5. **Flags & Concerns** — Anything that needs attention

Be precise with all numbers. Use markdown formatting for readability."""
        )

        response = self.llm.invoke([synthesis_prompt] + messages)

        logger.info(
            "synthesis_complete",
            response_length=len(response.content),
        )

        return {
            "messages": [response],
            "final_answer": response.content,
        }


def should_continue(state: dict[str, Any]) -> Literal["tools", "synthesis"]:
    """Router: decide whether to execute tools or move to synthesis."""
    messages = state.get("messages", [])
    if not messages:
        return "synthesis"

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesis"