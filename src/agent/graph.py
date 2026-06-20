# src/agents/graph.py
"""
LangGraph agent graph definition for financial analysis workflow.

The graph orchestrates:
1. Document retrieval (RAG)
2. LLM reasoning and tool selection
3. Tool execution (spending analysis, anomaly detection, etc.)
4. Final synthesis and reporting

Graph topology:
    START → retrieval → analyst → [tools ↔ analyst] → synthesis → END
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

import structlog
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.nodes import (
    FinancialAnalystNode,
    RetrievalNode,
    SynthesisNode,
    ToolExecutionNode,
    should_continue,
)
from src.agents.tools import FinancialToolKit
from src.rag.chain import FinancialRAGChain

logger = structlog.get_logger(__name__)


class FinancialAnalysisState(TypedDict):
    """State schema for the financial analysis graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    rag_context: str
    retrieved_docs: list
    tools: list
    final_answer: str


def build_financial_analysis_graph(
    rag_chain: FinancialRAGChain,
    toolkit: FinancialToolKit,
) -> Any:
    """
    Build and compile the LangGraph financial analysis agent.

    Args:
        rag_chain: Initialized RAG chain for document retrieval
        toolkit: Financial toolkit with analysis tools

    Returns:
        Compiled LangGraph application
    """
    # Initialize nodes
    retrieval_node = RetrievalNode(rag_chain)
    analyst_node = FinancialAnalystNode()
    tool_node = ToolExecutionNode()
    synthesis_node = SynthesisNode()

    # Get tools from toolkit
    tools = toolkit.get_tools()

    # Build the graph
    graph = StateGraph(FinancialAnalysisState)

    # Add nodes
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesis", synthesis_node)

    # Define edges
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "analyst")

    # Conditional routing from analyst
    graph.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "tools": "tools",
            "synthesis": "synthesis",
        },
    )

    # After tool execution, go back to analyst for reasoning
    graph.add_edge("tools", "analyst")

    # End after synthesis
    graph.add_edge("synthesis", END)

    # Compile with tools injected into initial state
    compiled = graph.compile()

    logger.info("financial_analysis_graph_built", nodes=list(graph.nodes.keys()))
    return compiled


def create_analysis_input(
    question: str,
    toolkit: FinancialToolKit,
    chat_history: list[BaseMessage] | None = None,
) -> dict:
    """Create the input state for running the financial analysis graph."""
    messages = chat_history or []
    messages.append({"role": "user", "content": question})

    return {
        "messages": messages,
        "tools": toolkit.get_tools(),
        "rag_context": "",
        "retrieved_docs": [],
        "final_answer": "",
    }