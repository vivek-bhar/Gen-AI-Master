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
# Global Constants & Paths
# ---------------------------------------------------------------------------
LOG_DB_FILE = "vmware_cluster.db"
CHROMA_PERSIST_DIR = "./chroma_vmware_kb"
MODEL_NAME = "llama3.2"


# ---------------------------------------------------------------------------
# 0. Database & Vector Store Setup
# ---------------------------------------------------------------------------
def init_demo_databases():
    """Initializes a mock SQLite database for VMware logs and populates a Chroma Vector DB."""
    # 1. Setup SQLite for Cluster Logs
    conn = sqlite3.connect(LOG_DB_FILE)
    cursor = conn.cursor()
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
    cursor.execute("DELETE FROM cluster_logs")
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
    conn.commit()
    conn.close()

    # 2. Setup Vector Database (VMware Knowledge Base & Fixes)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
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
    Chroma.from_documents(
        kb_docs,
        embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="vmware_fixes",
    )


# ---------------------------------------------------------------------------
# 1. LLM Helper
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Instantiates a ChatOllama instance with standard model configuration."""
    return ChatOllama(model=MODEL_NAME, temperature=temperature)


# ---------------------------------------------------------------------------
# 2. Agent Tools
# ---------------------------------------------------------------------------
@tool
def vector_kb_search(query: str) -> str:
    """Search the Vector Database (VMware Knowledge Base) for solutions, fix procedures,
    and remediation steps for known VMware error codes or cluster alerts."""
    try:
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="vmware_fixes",
        )
        docs = vectorstore.similarity_search(query, k=2)
        if not docs:
            return "No matching KB articles found in vector database."
        return "\n\n".join(
            [
                f"[Source: {d.metadata.get('source')}]\n{d.page_content}"
                for d in docs
            ]
        )
    except Exception as exc:
        return f"Vector DB Search Error: {exc}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code for data processing, log parsing calculations, or diagnostic metrics generation.
    Input must be valid Python code and must use print() to output results."""
    buffer = io.StringIO()
    local_namespace: dict = {}
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, local_namespace)
    except Exception as exc:
        return f"Error executing Python code: {exc}"
    output = buffer.getvalue().strip()
    return output if output else "(code ran with no printed output)"


@tool
def execute_sql(query: str) -> str:
    """Execute a READ-ONLY SQL query against the VMware cluster logs database (`vmware_cluster.db`).
    Table: 'cluster_logs'
    Columns: id (INTEGER), timestamp (DATETIME), host_name (TEXT), severity (TEXT), error_code (TEXT), message (TEXT)."""
    try:
        conn = sqlite3.connect(LOG_DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        if not rows:
            return "Query executed successfully, but returned no matching logs."

        return f"Columns: {columns}\nLogs:\n" + "\n".join(
            [str(row) for row in rows]
        )
    except Exception as exc:
        return f"Database Query Error: {exc}"


# ---------------------------------------------------------------------------
# 3. Sub-Agents Builder
# ---------------------------------------------------------------------------
def make_subagent(tool_fn, system_prompt: str):
    """Creates a ReAct sub-agent using LangGraph."""
    llm = get_llm()
    return create_react_agent(llm, tools=[tool_fn], prompt=system_prompt)


def run_subagent(subagent, task: str) -> str:
    """Executes the given sub-agent task and extracts the final text response."""
    result = subagent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# 4. Orchestrator Class
# ---------------------------------------------------------------------------
Route = Literal["rag_fix", "python", "log_database", "general"]

ROUTER_PROMPT = """You are a VMware Cluster Operations Routing Classifier. Analyze the incoming user task or cluster alert, and route it to the best matching sub-agent. Reply with ONLY one word:

- rag_fix      -> Queries seeking solutions, KB fixes, troubleshooting steps, or resolutions for VMware errors/alerts.
- python       -> Data processing, resource calculations, automated scripting, or algorithm execution.
- log_database -> Requests to query, retrieve, or inspect cluster logs, hosts, error events, or severities.
- general      -> General VMware concepts, introductory questions, or conversational responses.

Task: {task}
Category:"""


class VMwareOrchestrator:

    def __init__(self):
        self.router_llm = get_llm(temperature=0.0)
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
        raw = self.router_llm.invoke(ROUTER_PROMPT.format(task=task)).content
        match = re.search(
            r"rag_fix|python|log_database|general", raw.lower()
        )
        route = match.group(0) if match else "general"
        return route  # type: ignore[return-value]

    def run(self, task: str) -> str:
        route = self.classify(task)
        print(f"\n[orchestrator] routing task -> '{route}'")

        if route == "general":
            return self.router_llm.invoke(task).content

        return run_subagent(self.subagents[route], task)


# ---------------------------------------------------------------------------
# 5. Main Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Setting up local SQLite and Chroma Vector DB...")
    init_demo_databases()

    orchestrator = VMwareOrchestrator()

    demo_tasks = [
        "What is the recommended fix for the PSOD_0x0000000A kernel panic error on ESXi?",
        "Find all CRITICAL and ERROR level logs recorded in the cluster database.",
        "Calculate the mean recovery time if 3 ESXi host failures took 14 mins, 22 mins, and 18 mins.",
        "How do I resolve a DATASTORE_FULL alert on a vSAN cluster?",
        "What is VMware vMotion?",
    ]

    for task in demo_tasks:
        print("=" * 80)
        print(f"TASK: {task}")
        answer = orchestrator.run(task)
        print(f"\nANSWER:\n{answer}\n")