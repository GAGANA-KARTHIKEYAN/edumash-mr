# main.py — eduMASH-MR Pipeline Runner (CLI version)

import sys
import os

# ── Module imports ───────────────────────────────────────────────────
from utils.pdf_loader import load_all_documents
from retrieval_module.rag_retriever import GraphRAGRetriever
from misconception_module.detector import detect_misconceptions
from explanation_module.engine import explanation_engine, configure_gemini
from input_module.input_handler import get_input


def build_system(data_dir: str = "data/") -> GraphRAGRetriever:
    """Load curriculum data and build the Graph RAG index."""
    from utils.pdf_loader import load_all_documents
    documents = load_all_documents(data_dir)
    if not documents:
        raise ValueError(
            "Could not extract any readable text from your file(s). "
            "This usually means your PDF is image-scanned (no selectable text) or uses an encrypted font. "
            "Fix: Open your file → File → Print → Save as PDF (this creates a text-based PDF). "
            "Or paste your content into a .txt file and upload that instead."
        )


    retriever = GraphRAGRetriever()
    retriever.build(documents)
    return retriever


def run_pipeline(retriever: GraphRAGRetriever, student_answer: str) -> dict:
    """Full Graph RAG pipeline for one student answer."""
    # 1. Graph RAG Retrieval
    flat_chunks, graph_context, seed_nodes = retriever.retrieve(student_answer, k=3)
    retrieved_text = " ".join(flat_chunks)

    # 2. Misconception Detection
    report = detect_misconceptions(
        student_answer,
        retrieved_text,
        graph_nodes=seed_nodes
    )

    # 3. Explanation Engine
    result = explanation_engine(
        student_answer,
        report,
        flat_chunks,
        graph_context=graph_context,
    )

    return {
        "student_answer"  : student_answer,
        "misconception"   : report,
        "explanation"     : result,
        "graph_context"   : graph_context,
    }


# ── Built-in demo curriculum (works without any PDF) ────────────────
DEMO_TEXT = """
Photosynthesis is a process used by plants to convert light energy into chemical energy.
Plants produce glucose from carbon dioxide and water using sunlight.
Chlorophyll is the green pigment that absorbs light.
Oxygen is produced as a by-product when water molecules are split.
The overall reaction is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.

Kinematics is the study of motion without considering forces.
Velocity is a vector quantity that refers to the rate of change of displacement.
Speed is a scalar quantity and does not include direction.
Acceleration is the rate of change of velocity with respect to time.
Displacement is the change in position and includes direction.

Arrays are data structures that store elements at contiguous memory locations.
Linked lists store elements at non-contiguous memory using pointers.
A node in a linked list contains data and a reference to the next node.
Binary search trees store data in sorted order for efficient lookup.
"""


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  eduMASH-MR — Graph RAG Tutoring System")
    print("=" * 60)

    # Optional: configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        configure_gemini(api_key)

    retriever = build_system()

    print("\n[Pipeline ready] Type 'quit' to exit.\n")
    while True:
        answer = get_input()
        if answer.lower() in ("quit", "exit", "q"):
            break

        output = run_pipeline(retriever, answer)

        print("\n── RESULT ──────────────────────────────────")
        print(json.dumps(output["explanation"], indent=2))
        print("\n── MISCONCEPTION REPORT ─────────────────────")
        print(json.dumps(output["misconception"], indent=2))
        print("\n── GRAPH CONTEXT ────────────────────────────")
        print(output["graph_context"] or "(none)")
        print("─" * 45 + "\n")
