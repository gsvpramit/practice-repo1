"""First piece of code: a minimal LangGraph workflow.

The graph has two nodes:
  1. ``prepare`` - cleans up the user's question.
  2. ``answer``  - answers it with Claude (via LangChain) when ANTHROPIC_API_KEY
                   is set, otherwise returns an offline echo so the graph can
                   still be run and explored without credentials.

Run:
    python main.py "What is gradient descent?"
"""

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

MODEL = "claude-sonnet-5-5"


class State(TypedDict):
    question: str
    answer: str


def prepare(state: State) -> dict:
    question = state["question"].strip()
    if not question.endswith("?"):
        question += "?"
    return {"question": question}


def answer(state: State) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"answer": f"[offline mode] You asked: {state['question']}"}

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=MODEL, max_tokens=1024)
    response = llm.invoke(state["question"])
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("prepare", prepare)
    graph.add_node("answer", answer)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is machine learning"
    result = build_graph().invoke({"question": question, "answer": ""})
    print(result["answer"])


if __name__ == "__main__":
    main()
