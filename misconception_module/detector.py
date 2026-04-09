# misconception_module/detector.py
# Detects conceptual gaps between student answer and reference content.
# Uses keyword overlap + optional T5 structural graph analysis.

import re
from typing import List, Dict

# ── Lazy-loaded T5 Graph Generator (fully optional) ─────────────────
_graph_gen = None
_graph_gen_loaded = False  # tracks whether we already attempted loading

def _get_graph_gen():
    global _graph_gen, _graph_gen_loaded
    if _graph_gen_loaded:
        return _graph_gen
    _graph_gen_loaded = True
    try:
        from training.models.misconception_graph_gen import load_graph_gen
        print("[Detector] Loading T5 Graph Generator...")
        _graph_gen = load_graph_gen("weights/t5_graph_gen_final")
    except Exception as e:
        print(f"[Detector] T5 Graph Gen unavailable: {e}")
        _graph_gen = None
    return _graph_gen


# ── Lazy-loaded Semantic Model ──────────────────────────────────────
_semantic_model = None

def _get_semantic_model():
    global _semantic_model
    if _semantic_model is not None:
        return _semantic_model
    try:
        from sentence_transformers import SentenceTransformer
        print("[Detector] Loading Semantic Similarity Model...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[Detector] ✓ Semantic Model loaded.")
    except Exception as e:
        print(f"[Detector] Semantic model load failed: {e}")
        _semantic_model = None
    return _semantic_model


STOP_WORDS = {
    "the", "a", "an", "is", "of", "in", "to", "and", "or", "it", "this",
    "that", "which", "with", "from", "by", "on", "at", "are", "was", "were",
    "be", "been", "has", "have", "had", "do", "does", "can", "will", "as",
    "for", "but", "not", "its", "also", "more", "into", "than", "their",
    "they", "these", "those", "such", "each", "would", "should", "could",
    "about", "between", "through", "where", "when", "how", "what", "there",
}


def extract_keywords(text: str) -> List[str]:
    """Tokenise, lowercase, remove stop-words and short tokens."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def detect_misconceptions(
    student_answer: str,
    reference_text: str,
    graph_nodes: List[str] = None,
) -> Dict:
    """
    Compare student answer against reference context.

    Returns dict with:
      missing_concepts, incorrect_relations, score, graph_missing,
      student_triplets, ref_triplets, missing_links
    """
    student_words = set(extract_keywords(student_answer))
    ref_words     = set(extract_keywords(reference_text))

    if not ref_words:
        return {
            "missing_concepts": [], "incorrect_relations": [],
            "score": 1.0, "graph_missing": [], "missing_links": [],
            "student_triplets": [], "ref_triplets": [],
        }

    # ── 1. Keyword Overlap (Structural) ──
    match_count = len(student_words & ref_words)
    keyword_score = match_count / len(ref_words)

    # ── 2. Semantic Similarity (Meaning-based) ──
    semantic_score = keyword_score # Default fallback
    model = _get_semantic_model()
    if model:
        try:
            from sentence_transformers import util
            emb1 = model.encode(student_answer, convert_to_tensor=True)
            emb2 = model.encode(reference_text, convert_to_tensor=True)
            semantic_score = float(util.cos_sim(emb1, emb2)[0][0])
        except Exception as e:
            print(f"[Detector] Semantic scoring failed: {e}")

    # ── 3. Blended Score ──
    # We give more weight to semantic meaning (70%) than exact keyword matches (30%)
    # This prevents 'injustice' when students use synonyms.
    score = (keyword_score * 0.3) + (max(0, semantic_score) * 0.7)

    missing = sorted(ref_words - student_words)

    # ── Graph node keyword analysis (always runs) ──
    wrong_connections = []
    missing_links = []
    graph_missing = []
    student_triplets = []
    ref_triplets = []

    if graph_nodes:
        for node in graph_nodes:
            node_words = set(node.lower().split())
            if not node_words & student_words:
                graph_missing.append(node)

    # ── Optional: T5 Structural Graph Analysis ──
    graph_gen = _get_graph_gen()
    if graph_gen:
        try:
            from inference.graph_comparator import compare_graphs

            student_triplets = graph_gen.generate_graph(
                question="Explain the concept.",
                student_answer=student_answer,
                reference=reference_text
            )

            ref_triplets = graph_gen.generate_graph(
                question="Explain the concept.",
                student_answer=reference_text,
                reference=reference_text
            )

            graph_report = compare_graphs(student_triplets, ref_triplets)

            # Blend: 50% keyword overlap + 50% structural graph similarity
            graph_score = graph_report.get("score", 0)
            score = (score * 0.5) + (graph_score * 0.5)

            graph_missing.extend(graph_report.get("missing_concepts", []))
            wrong_connections = graph_report.get("wrong_connections", [])
            missing_links = graph_report.get("missing_links", [])
        except Exception as e:
            print(f"[Detector] Graph comparison failed: {e}")

    return {
        "missing_concepts": list(set(missing))[:5],
        "incorrect_relations": wrong_connections,
        "missing_links": missing_links,
        "score": round(score, 3),
        "graph_missing": list(set(graph_missing))[:5],
        "student_triplets": student_triplets,
        "ref_triplets": ref_triplets,
    }
