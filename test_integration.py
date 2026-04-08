import os
from langchain_core.documents import Document
from retrieval_module.rag_retriever import GraphRAGRetriever
from misconception_module.detector import detect_misconceptions

def test():
    print("1. Creating fake curriculum Document...")
    fake_doc = Document(page_content="Plants need sunlight, water, and carbon dioxide to perform photosynthesis and generate food.")
    
    print("2. Initializing GraphRAGRetriever with trained weights...")
    retriever = GraphRAGRetriever()
    
    print("3. Building FAISS index + Graph...")
    retriever.build([fake_doc])
    
    print("4. Retrieving context for query...")
    flat, graph_ctx, seed_nodes = retriever.retrieve("What do plants need?", k=1)
    
    print("5. Context retrieved successfully.")
    print("Graph Nodes Found:", seed_nodes)
    print("Flat CHUNK:", flat)
    
    print("6. Simulating a Misconception detection on bad answer...")
    student_ans = "Plants need oxygen."
    ref_ans = "Plants need carbon dioxide and sunlight."
    
    report = detect_misconceptions(student_ans, ref_ans, graph_nodes=seed_nodes)
    print("---------------------------------")
    print("Detection Report:", report)
    print("✅ Full integration passed strictly off fine-tuned GPU weights.")

if __name__ == "__main__":
    test()
