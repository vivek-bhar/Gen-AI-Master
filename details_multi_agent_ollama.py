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

# Import standard Python libraries needed for:
# - capturing output from executed Python code
# - pattern matching for route detection
# - redirecting output to an in-memory buffer
import io
import re
import contextlib
from typing import Literal

# Import LangChain / LangGraph / Wikipedia libraries needed to build the agents:
# - ChatOllama connects Python to a local Ollama language model
# - tool decorator creates callable tools for the agents
# - create_react_agent builds a ReAct agent in LangGraph
# - WikipediaAPIWrapper fetches summary information from Wikipedia
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper


# ---------------------------------------------------------------------------
# 1. LLM (shared by every agent, but you could give each a different model)
#    This sets the model name used across all agents.
# ---------------------------------------------------------------------------
MODEL_NAME = "llama3.2"


# Function to create a language model instance.
# temperature=0.0 makes responses more deterministic and less random.
def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, temperature=temperature)


# ---------------------------------------------------------------------------
# 2. Tools
#    Using the @tool decorator (not the old Tool(...) class) so the model's
#    native tool-calling gets a proper name/description/args schema.
#    These tools allow the agents to access external capabilities:
#    - Wikipedia for factual information
#    - Python execution for calculations / logic
# ---------------------------------------------------------------------------

# Create a Wikipedia wrapper configured with a small result limit.
# This avoids returning too much content and keeps responses concise.
_wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)


# Tool 1: Wikipedia search
# The LLM can call this whenever it needs background factual knowledge.
@tool
def wikipedia_search(query: str) -> str:
    """Look up factual/background/encyclopedic information about a person,
    place, concept, or historical event on Wikipedia.
    Input should be a concise search phrase, e.g. 'Alan Turing'."""
    return _wiki.run(query)


# Tool 2: Python REPL
# This lets the agent run Python code and return any output that was printed.
@tool
def python_repl(code: str) -> str:
    """Read-Eval-Print Loop = Execute Python code and return whatever it prints to stdout.
    Use this for calculations, data processing, or algorithms.
    Input must be valid Python code, and it must print() the final result."""
    # Capture printed output in an in-memory buffer so it can be passed back.
    buffer = io.StringIO()
    local_namespace: dict = {}
    try:
        # Redirect stdout to the buffer while executing the code.
        with contextlib.redirect_stdout(buffer):
            # exec() runs the Python code in a controlled environment.
            exec(code, {"__builtins__": __builtins__}, local_namespace)
    except Exception as exc:  # surface errors back to the agent as text
        # If the code fails, return the exception message instead of crashing.
        return f"Error while executing code: {exc}"

    # Get the captured output from stdout.
    output = buffer.getvalue().strip()

    # If there was no printed output, return a fallback message.
    return output if output else "(code ran with no printed output)"


# ---------------------------------------------------------------------------
# 3. Sub-agents — each is a langgraph ReAct agent bound to exactly one tool
#    Each sub-agent is specialized for one job:
#    - research agent uses Wikipedia
#    - coding agent uses Python execution
# ---------------------------------------------------------------------------

# Create a sub-agent with a specific tool and a custom instruction prompt.
def make_subagent(tool_fn, system_prompt: str):
    llm = get_llm()
    return create_react_agent(llm, tools=[tool_fn], prompt=system_prompt)


# Run a sub-agent with a user task and return the agent's final message.
def run_subagent(subagent, task: str) -> str:
    # Format the task as a user message in the standard chat format.
    result = subagent.invoke({"messages": [{"role": "user", "content": task}]})

    # Return the last message in the result as the final answer.
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# 4. Orchestrator — classifies the task, then dispatches to the right sub-agent
#    This is the main controller that decides what the system should do.
# ---------------------------------------------------------------------------

# Allowed route names for the task router.
Route = Literal["wikipedia", "python", "general"]

# The router prompt instructs the LLM to classify the task into one category.
# It must output only one word: wikipedia, python, or general.
ROUTER_PROMPT = """You are a routing classifier. Given a user task, decide which
single category best fits it. Reply with ONLY one word, no punctuation:

- wikipedia  -> the task needs general/background/encyclopedic knowledge (history, biography, definitions, static facts, "who was", "what is")
- python     -> the task needs a calculation, data transformation, or algorithm executed in code
- general    -> anything else a plain language model can answer directly, no tool needed (creative writing, opinions, simple chat)

Task: {task}
Category:"""


# Orchestrator class manages the routing logic and sub-agents.
class Orchestrator:
    def __init__(self):
        # Router LLM: used only to decide which category a task belongs to.
        self.router_llm = get_llm(temperature=0.0)

        # Sub-agents dictionary. Each agent is specialized for one tool.
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

    # Decide which category the task fits into.
    def classify(self, task: str) -> Route:
        # Ask the router model to classify the task.
        raw = self.router_llm.invoke(ROUTER_PROMPT.format(task=task)).content

        # Use regex to find one of the valid route names in the model output.
        match = re.search(r"wikipedia|python|general", raw.lower())

        # If a valid route is found, use it. Otherwise default to general.
        route = match.group(0) if match else "general"
        return route  # type: ignore[return-value]

    # Run the full orchestration process for one task.
    def run(self, task: str) -> str:
        # First, classify the task.
        route = self.classify(task)
        print(f"\n[orchestrator] routing task -> '{route}'")

        # If the route is general, no tool is needed; answer directly.
        if route == "general":
            return self.router_llm.invoke(task).content

        # Otherwise, send the task to the appropriate sub-agent.
        return run_subagent(self.subagents[route], task)


# ---------------------------------------------------------------------------
# 5. Demo
#    This part runs only when the script is executed directly.
#    It creates the orchestrator and can be used to test the system.
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
