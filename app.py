"""
Chatbot UI for the Multi-Agent Orchestrator (Gradio)
=====================================================

Wraps the Orchestrator from multi_agent_ollama.py in a simple chat window.
The user types a question, the orchestrator classifies it (wikipedia /
python / general), routes it to the right sub-agent, and the answer is
shown in the chat.

Requirements
------------
    pip install -U gradio
    (plus everything multi_agent_ollama.py needs — see that file's header)

Run
---
    python app.py
    # then open the local URL Gradio prints, e.g. http://127.0.0.1:7860
"""

import gradio as gr
 
from multi_agent_ollama import Orchestrator
 
 
# Build the orchestrator ONCE at startup (not on every message) — this is
# what actually loads the LLM connections and sub-agents.
print("Initializing orchestrator and sub-agents... (this connects to Ollama)")
orchestrator = Orchestrator()
print("Ready.")
 
 
def chat_fn(message: str, history: list) -> str:
    """
    Called by Gradio every time the user sends a message.
      message: the new user question (str)
      history: prior turns (unused here — the orchestrator is stateless
                per question, but you could thread it in if you want
                multi-turn memory)
    """
    if not message or not message.strip():
        return "Please type a question."
 
    try:
        answer = orchestrator.run(message)
    except Exception as exc:
        answer = (
            f"Something went wrong while handling that request: {exc}\n\n"
            "Make sure Ollama is running (`ollama serve`) and the model is "
            "pulled (`ollama pull llama3.2`)."
        )
    return answer
 
 
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Multi-Agent Assistant (Wikipedia + Python REPL)",
    description=(
        "Type your own question below. An orchestrator agent decides whether "
        "it needs Wikipedia research, a Python calculation, or a plain "
        "answer — and routes it to the matching sub-agent."
    ),
)
 
if __name__ == "__main__":
    demo.launch()
 