"""
Multi-Agent System with an Orchestrator Agent (Ollama + llama3.2)
==================================================================

Architecture
------------
                     ┌─────────────────────┐
   user task  ─────► │   ORCHESTRATOR AGENT │
                     │  (routes the task)   │
                     └──────────┬───────────┘
                                │  decides which sub-agent fits best
                  ┌──────────────────┴───────────────────┐
                  ▼                                       ▼
        ┌───────────────┐                      ┌──────────────────┐
        │ Research Agent │                      │  Coder Agent     │
        │ (Wikipedia)    │                      │  (Python REPL)   │
        └───────────────┘                      └──────────────────┘

Each sub-agent is built with `langgraph.prebuilt.create_react_agent`, which
uses the LLM's native tool-calling (no manual prompt parsing needed — this
is the current, LangChain 1.0+ compatible way to build ReAct agents; the old
`from langchain.agents import AgentExecutor, create_react_agent` was moved
out of the core `langchain` package).

Requirements
------------
    pip install -U langchain-ollama langchain-community langgraph wikipedia

    (langgraph ships as a dependency of modern langchain, but pin it
    explicitly above just in case)

    # Ollama must be running locally with the model pulled:
    ollama pull llama3.2
    ollama serve   # (usually already running as a service)

Run
---
    python multi_agent_ollama.py
"""

import io
import re
import contextlib
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper


# ---------------------------------------------------------------------------
# 1. LLM (shared by every agent, but you could give each a different model)
# ---------------------------------------------------------------------------
MODEL_NAME = "llama3.2"


def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, temperature=temperature)


# ---------------------------------------------------------------------------
# 2. Tools
#    Using the @tool decorator (not the old Tool(...) class) so the model's
#    native tool-calling gets a proper name/description/args schema.
# ---------------------------------------------------------------------------
_wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)


@tool
def wikipedia_search(query: str) -> str:
    """Look up factual/background/encyclopedic information about a person,
    place, concept, or historical event on Wikipedia.
    Input should be a concise search phrase, e.g. 'Alan Turing'."""
    return _wiki.run(query)


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return whatever it prints to stdout.
    Use this for calculations, data processing, or algorithms.
    Input must be valid Python code, and it must print() the final result."""
    buffer = io.StringIO()
    local_namespace: dict = {}
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, local_namespace)
    except Exception as exc:  # surface errors back to the agent as text
        return f"Error while executing code: {exc}"
    output = buffer.getvalue().strip()
    return output if output else "(code ran with no printed output)"


# ---------------------------------------------------------------------------
# 3. Sub-agents — each is a langgraph ReAct agent bound to exactly one tool
# ---------------------------------------------------------------------------
def make_subagent(tool_fn, system_prompt: str):
    llm = get_llm()
    return create_react_agent(llm, tools=[tool_fn], prompt=system_prompt)


def run_subagent(subagent, task: str) -> str:
    result = subagent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# 4. Orchestrator — classifies the task, then dispatches to the right sub-agent
# ---------------------------------------------------------------------------
Route = Literal["wikipedia", "python", "general"]

ROUTER_PROMPT = """You are a routing classifier. Given a user task, decide which
single category best fits it. Reply with ONLY one word, no punctuation:

- wikipedia  -> the task needs general/background/encyclopedic knowledge (history, biography, definitions, static facts, "who was", "what is")
- python     -> the task needs a calculation, data transformation, or algorithm executed in code
- general    -> anything else a plain language model can answer directly, no tool needed (creative writing, opinions, simple chat)

Task: {task}
Category:"""


class Orchestrator:
    def __init__(self):
        self.router_llm = get_llm(temperature=0.0)
        self.subagents = {
            "wikipedia": make_subagent(
                wikipedia_search,
                "You are a research assistant. Use the wikipedia_search tool "
                "to answer the user's question, then give a clear, concise answer.",
            ),
            "python": make_subagent(
                python_repl,
                "You are a coding assistant. Use the python_repl tool to "
                "compute the answer, then report the final result clearly.",
            ),
        }

    def classify(self, task: str) -> Route:
        raw = self.router_llm.invoke(ROUTER_PROMPT.format(task=task)).content
        match = re.search(r"wikipedia|python|general", raw.lower())
        route = match.group(0) if match else "general"
        return route  # type: ignore[return-value]

    def run(self, task: str) -> str:
        route = self.classify(task)
        print(f"\n[orchestrator] routing task -> '{route}'")

        if route == "general":
            # No tool needed — answer directly with the base LLM.
            return self.router_llm.invoke(task).content

        return run_subagent(self.subagents[route], task)


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    orchestrator = Orchestrator()
"""
    demo_tasks = [
        "Who was Ada Lovelace and why is she important to computing?",
        "Compute the 15th Fibonacci number and print it.",
        "What is the capital of Japan?",
        "Write a haiku about the ocean.",
    ]

    for t in demo_tasks:
        print("=" * 80)
        print(f"TASK: {t}")
        answer = orchestrator.run(t)
        #print(f"\nANSWER: {answer}\n")
        """
