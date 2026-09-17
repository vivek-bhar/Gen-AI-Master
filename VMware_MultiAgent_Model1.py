"""
Multi-Agent VMware Cluster Log & Alert Remediation System (Ollama + llama3.2)
=============================================================================

Architecture
------------
                     ┌─────────────────────┐
   alert / task ────►│  ORCHESTRATOR AGENT │
                     │  (routes the task)  │
                     └──────────┬──────────┘
                                │  decides which specialized sub-agent fits best
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                      ▼
   ┌───────────────┐     ┌──────────────┐     ┌──────────────────┐
   │  Vector DB    │     │  VMware Log  │     │ Python REPL      │
   │  RAG Agent    │     │  DB Agent    │     │ Diagnostic Agent │
   │ (KBs & Fixes) │     │ (Cluster DB) │     │ (Calculations)   │
   └───────────────┘     └──────────────┘     └──────────────────┘

Requirements
------------
    pip install -U langchain-ollama langchain-community langgraph chromadb

Run
---
    python vmware_rag_orchestrator.py
"""

import contextlib
import io
import re
import sqlite3
from typing import Literal

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Import Notes:
# - sqlite3 stores VMware alerts in a local database.
# - Chroma + OllamaEmbeddings create a searchable vector knowledge base.
# - ChatOllama connects Python to the local LLM (llama3.2).
# - create_react_agent builds a reasoning agent that can call tools.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global Constants & Paths
# ---------------------------------------------------------------------------
# These values define the file names and model used by the whole application.
# The SQLite database stores VMware cluster log events.
# The Chroma directory stores the knowledge base as vector embeddings.
LOG_DB_FILE = "vmware_cluster.db"
CHROMA_PERSIST_DIR = "./chroma_vmware_kb"
MODEL_NAME = "llama3.2"


# ---------------------------------------------------------------------------
# 0. Database & Vector Store Setup
# ---------------------------------------------------------------------------
# This function creates the demo environment used by the app.
# It prepares a local SQLite database containing sample VMware cluster alerts
# and a Chroma vector database containing VMware troubleshooting knowledge.
# The idea is to simulate the data that an operations agent would inspect in real life.
def init_demo_databases():
    """Initializes a mock SQLite database for VMware logs and populates a Chroma Vector DB."""
    # 1. Setup SQLite for Cluster Logs
    # sqlite3.connect() opens a local database file named vmware_cluster.db.
    # This file will hold VMware cluster log entries for later SQL inspection.
    conn = sqlite3.connect(LOG_DB_FILE)
    cursor = conn.cursor()

    # CREATE TABLE IF NOT EXISTS ensures the table is created once.
    # The table stores the event timestamp, host involved, severity level,
    # error code, and descriptive message text for each cluster alert.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            host_name TEXT,
            severity TEXT,
            error_code TEXT,
            message TEXT
        )
    """
    )

    # DELETE FROM cluster_logs clears previous sample rows before reloading data.
    # This keeps the demo environment fresh and consistent every time the script runs.
    cursor.execute("DELETE FROM cluster_logs")

    # executemany() inserts multiple rows in one efficient batch.
    # Each row simulates a VMware event such as a host crash, datastore warning, or network problem.
    cursor.executemany(
        """
        INSERT INTO cluster_logs (timestamp, host_name, severity, error_code, message) 
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                "2026-03-15 08:30:00",
                "esxi-node-01.corp.local",
                "CRITICAL",
                "PSOD_0x0000000A",
                "Host crashed with Purple Screen of Death: Kernel panic in module vmw_pvscsi",
            ),
            (
                "2026-03-15 09:12:00",
                "esxi-node-02.corp.local",
                "WARNING",
                "DATASTORE_FULL",
                "Datastore ds_vSAN_01 capacity reached 94%. Latency threshold exceeded.",
            ),
            (
                "2026-03-15 10:05:00",
                "esxi-node-03.corp.local",
                "ERROR",
                "NET_DISCONNECT",
                "vSwitch0 Uplink vmnic1 link state DOWN. Redundancy lost.",
            ),
        ],
    )

    # commit() saves the inserted rows in the database.
    conn.commit()
    conn.close()

    # 2. Setup Vector Database (VMware Knowledge Base & Fixes)
    # OllamaEmbeddings converts text into vectors so the system can search by meaning,
    # not only exact words. The chosen model is llama3.2.
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # kb_docs creates knowledge-base documents that contain troubleshooting guidance.
    # Each document has both text content and metadata such as source ID and topic.
    kb_docs = [
        Document(
            page_content="Fix for PSOD_0x0000000A (vmw_pvscsi): Update the PVSCSI driver to version 2.1.5 or disable queue depth scaling. Perform an ESXi host reboot after patch application.",
            metadata={
                "source": "KB88201",
                "topic": "PSOD Kernel Panic Driver Issue",
            },
        ),
        Document(
            page_content="Fix for DATASTORE_FULL / vSAN Latency: Consolidate VM snapshots on datastore, initiate vSAN rebalance, or expand storage tier capacity using PowerCLI `Get-Datastore | Extension`.",
            metadata={"source": "KB71044", "topic": "vSAN Storage Capacity"},
        ),
        Document(
            page_content="Fix for NET_DISCONNECT / vmnic Down: Check physical switch port config (LACP/Trunking). Restart management agents on ESXi using `services.sh restart`.",
            metadata={"source": "KB10034", "topic": "Network Redundancy Loss"},
        ),
    ]

    # Chroma.from_documents() stores all the knowledge articles in the vector database.
    # Later, a user question can be converted into embeddings and matched to the closest fix.
    Chroma.from_documents(
        kb_docs,
        embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="vmware_fixes",
    )


# ---------------------------------------------------------------------------
# 1. LLM Helper
# ---------------------------------------------------------------------------
# This function creates the main language model instance used by the router and agents.
# temperature=0.0 keeps responses more focused and deterministic instead of creative.
def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Instantiates a ChatOllama instance with standard model configuration."""
    return ChatOllama(model=MODEL_NAME, temperature=temperature)


# ---------------------------------------------------------------------------
# 2. Agent Tools
# ---------------------------------------------------------------------------
# These functions are wrapped as LangChain tools so an LLM can call them autonomously.
# Each tool gives the agent a specific capability: search the KB, run Python code, or query logs.

@tool
def vector_kb_search(query: str) -> str:
    """Search the Vector Database (VMware Knowledge Base) for solutions, fix procedures,
    and remediation steps for known VMware error codes or cluster alerts."""
    # This tool attempts to find the most relevant VMware fixes using semantic similarity.
    try:
        # Recreate the embedding model and open the persisted Chroma collection.
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="vmware_fixes",
        )

        # similarity_search() returns the top matching documents for the query.
        docs = vectorstore.similarity_search(query, k=2)

        # If nothing matches, the tool returns a useful fallback message.
        if not docs:
            return "No matching KB articles found in vector database."

        # Each result is formatted with its source ID and the article text.
        return "\n\n".join(
            [
                f"[Source: {d.metadata.get('source')}]\n{d.page_content}"
                for d in docs
            ]
        )
    except Exception as exc:
        # If the vector DB search fails, capture the exception and report it.
        return f"Vector DB Search Error: {exc}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code for data processing, log parsing calculations, or diagnostic metrics generation.
    Input must be valid Python code and must use print() to output results."""
    # Redirect standard output to an in-memory buffer so the function can return the result.
    buffer = io.StringIO()
    local_namespace: dict = {}
    try:
        # exec() runs the submitted Python code in a local namespace.
        # redirect_stdout ensures that any print() output is captured and returned.
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, local_namespace)
    except Exception as exc:
        # Any execution error is returned as a readable string.
        return f"Error executing Python code: {exc}"

    # Gather the printed output and return it to the agent.
    output = buffer.getvalue().strip()
    return output if output else "(code ran with no printed output)"


@tool
def execute_sql(query: str) -> str:
    """Execute a READ-ONLY SQL query against the VMware cluster logs database (`vmware_cluster.db`).
    Table: 'cluster_logs'
    Columns: id (INTEGER), timestamp (DATETIME), host_name (TEXT), severity (TEXT), error_code (TEXT), message (TEXT)."""
    try:
        # Connect to the SQLite database and run the SQL command.
        conn = sqlite3.connect(LOG_DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)

        # Fetch rows and column names so the results are human-readable.
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        # If query returns nothing, the tool reports that clearly.
        if not rows:
            return "Query executed successfully, but returned no matching logs."

        # Format the output as column names followed by matching log records.
        return f"Columns: {columns}\nLogs:\n" + "\n".join(
            [str(row) for row in rows]
        )
    except Exception as exc:
        # Any database failure is surfaced back to the calling agent.
        return f"Database Query Error: {exc}"


# ---------------------------------------------------------------------------
# 3. Sub-Agents Builder
# ---------------------------------------------------------------------------
# This section creates specialized agents by attaching one tool to one LLM behavior.
# Each agent is designed for a different function: solving VMware issues,
# executing Python code, or querying logs.

def make_subagent(tool_fn, system_prompt: str):
    """Creates a ReAct sub-agent using LangGraph."""
    # The agent uses the same local LLM but gets a specific tool and role prompt.
    llm = get_llm()
    return create_react_agent(llm, tools=[tool_fn], prompt=system_prompt)


def run_subagent(subagent, task: str) -> str:
    """Executes the given sub-agent task and extracts the final text response."""
    # Invoke the agent with a user message and return the last message content.
    result = subagent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# 4. Orchestrator Class
# ---------------------------------------------------------------------------
# This is the controller that decides which specialist should handle each task.
# The router model looks at the user request and routes it to rag_fix, python,
# log_database, or a generic response.
Route = Literal["rag_fix", "python", "log_database", "general"]

# ROUTER_PROMPT defines the classification instructions for the router LLM.
# It tells the model exactly what each category means and asks for ONLY one word.
ROUTER_PROMPT = """You are a VMware Cluster Operations Routing Classifier. Analyze the incoming user task or cluster alert, and route it to the best matching sub-agent. Reply with ONLY one word:

- rag_fix      -> Queries seeking solutions, KB fixes, troubleshooting steps, or resolutions for VMware errors/alerts.
- python       -> Data processing, resource calculations, automated scripting, or algorithm execution.
- log_database -> Requests to query, retrieve, or inspect cluster logs, hosts, error events, or severities.
- general      -> General VMware concepts, introductory questions, or conversational responses.

Task: {task}
Category:"""


class VMwareOrchestrator:

    def __init__(self):
        # router_llm decides which category fits the user request.
        self.router_llm = get_llm(temperature=0.0)

        # subagents is a dictionary of specialist agents keyed by route name.
        # This is a simple dispatcher pattern: input task -> select best agent -> perform task.
        self.subagents = {
            "rag_fix": make_subagent(
                vector_kb_search,
                "You are a VMware Remediation Specialist. Query the vector database to find exact KB resolutions and fixes for VMware issues.",
            ),
            "python": make_subagent(
                python_repl,
                "You are an Automation & Diagnostic Coder. Use python_repl to process log metrics or perform cluster calculation tasks.",
            ),
            "log_database": make_subagent(
                execute_sql,
                "You are a VMware Log Database Analyst. Formulate read-only SQL queries to extract error details, host logs, or severity levels from 'cluster_logs'.",
            ),
        }

    def classify(self, task: str) -> Route:
        # Ask the router LLM to assign a category to the request.
        raw = self.router_llm.invoke(ROUTER_PROMPT.format(task=task)).content

        # Use regex to detect valid route names in the model response.
        # If the output doesn't match, default to 'general'.
        match = re.search(
            r"rag_fix|python|log_database|general", raw.lower()
        )
        route = match.group(0) if match else "general"
        return route  # type: ignore[return-value]

    def run(self, task: str) -> str:
        # Determine which specialist should respond.
        route = self.classify(task)
        print(f"\n[orchestrator] routing task -> '{route}'")

        # If the request is general VMware knowledge, the router can answer directly.
        if route == "general":
            return self.router_llm.invoke(task).content

        # Otherwise, delegate to the proper specialized agent.
        return run_subagent(self.subagents[route], task)


# ---------------------------------------------------------------------------
# 5. Main Execution Entry Point
# ---------------------------------------------------------------------------
# This block runs only when the script is executed directly.
# It sets up the data and launches the demo tasks to show the system working.
if __name__ == "__main__":
    print("Setting up local SQLite and Chroma Vector DB...")
    init_demo_databases()

    # Create the orchestrator once, then reuse it for multiple user requests.
    orchestrator = VMwareOrchestrator()

    # demo_tasks contains sample prompts that exercise all the supported routes.
    demo_tasks = [
        "What is the recommended fix for the PSOD_0x0000000A kernel panic error on ESXi?",
        "Find all CRITICAL and ERROR level logs recorded in the cluster database.",
        "Calculate the mean recovery time if 3 ESXi host failures took 14 mins, 22 mins, and 18 mins.",
        "How do I resolve a DATASTORE_FULL alert on a vSAN cluster?",
        "What is VMware vMotion?",
    ]

    # Loop through each example task and print the answer returned by the orchestrator.
    for task in demo_tasks:
        print("=" * 80)
        print(f"TASK: {task}")
        answer = orchestrator.run(task)
        print(f"\nANSWER:\n{answer}\n")