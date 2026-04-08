# retrieval_module/rag_retriever.py
# ── Graph RAG retriever ──
# Step 1 → FAISS vector search finds top-k semantically similar chunks.
# Step 2 → Graph traversal expands those chunks into a concept subgraph.
# Result → Richer, structured context vs plain flat RAG.

import faiss
import numpy as np
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from utils.graph_builder import build_graph, get_subgraph, subgraph_to_text, _clean

class EdumashEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # We explicitly trained on this Multilingual model
        self.text_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        from training.models.fusion_module import load_fusion
        
        weights_path = "weights/fusion_module_final.pt"
        if os.path.exists(weights_path):
            print("[retriever] 🌟 Detected fine-tuned Edumash weights! Utilizing Multimodal Fusion Matrix.")
            self.fusion = load_fusion(weights_path)
            self.fusion.to(self.device)
            self.fusion.eval()
        else:
            self.fusion = None
            
    def encode(self, texts: list[str], show_progress_bar=False, convert_to_numpy=True) -> np.ndarray:
        with torch.no_grad():
            embs = self.text_model.encode(texts, convert_to_tensor=True, show_progress_bar=show_progress_bar)
            embs = embs.to(self.device)
            if self.fusion:
                embs = self.fusion(text_emb=embs) # Projects via tuned offline weights
            if convert_to_numpy:
                return embs.cpu().numpy()
            return embs
            

class GraphRAGRetriever:
    def __init__(self):
        print("[retriever] Loading eduMASH custom embedder…")
        self.model = EdumashEmbedder()
        self.index = None
        self.chunks = []
        self.knowledge_graph = None
        self.graph_nodes = []     # lowercase node labels for matching

    # ── Build stage ──────────────────────────────────────────────────
    def build(self, documents):
        """Chunk docs → embed → FAISS + build knowledge graph."""

        # STEP 1: Chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = splitter.split_documents(documents)
        print(f"[retriever] Created {len(self.chunks)} text chunks.")

        # STEP 2: Embed chunks
        texts = [doc.page_content for doc in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # STEP 3: FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))
        print(f"[retriever] FAISS index built ({dim}d, {len(texts)} vectors).")

        # STEP 4: Build knowledge graph over the same documents
        self.knowledge_graph = build_graph(documents)
        self.graph_nodes = list(self.knowledge_graph.nodes())

    # ── Retrieve stage ───────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 3):
        """
        Graph RAG retrieval:
          1. FAISS top-k chunks  (vector similarity)
          2. Match query words to graph nodes  (concept anchoring)
          3. BFS subgraph expansion  (relational context)
          4. Return flat text + graph text combined
        """
        if self.index is None:
            raise RuntimeError("Call build() before retrieve().")

        # --- 1. Vector retrieval ---
        q_vec = self.model.encode([query], show_progress_bar=False).astype("float32")
        distances, indices = self.index.search(q_vec, k)
        flat_chunks = [self.chunks[i].page_content for i in indices[0] if i != -1]

        # --- 2. Concept anchoring: find matching graph nodes ---
        query_words = set(_clean(query).split())
        seed_nodes = [
            node for node in self.graph_nodes
            if any(w in node for w in query_words)
        ]

        # --- 3. Graph traversal (Graph RAG) ---
        graph_context = ""
        if seed_nodes and self.knowledge_graph:
            subgraph = get_subgraph(self.knowledge_graph, seed_nodes, hops=2)
            graph_context = subgraph_to_text(subgraph)

        return flat_chunks, graph_context, seed_nodes
