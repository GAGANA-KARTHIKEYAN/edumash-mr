# utils/graph_builder.py
# ── The GRAPH RAG novelty module ──
# Builds a NetworkX knowledge graph from curriculum text.
# Graph RAG = Vector retrieval (FAISS) + Graph traversal for richer context.

import re
import networkx as nx
from typing import List, Tuple

# -------------------------------------------------------------------
# Comprehensive relation extractor — works with diverse PDFs
# Pattern: "<concept> verb <concept>" with bounded match lengths
# -------------------------------------------------------------------
RELATION_PATTERNS = [
    # IS_A patterns
    (r"([A-Za-z][\w\s]{2,40}?)\s+is\s+(?:a|an|the)?\s*(?:type\s+of\s+)?([A-Za-z][\w\s]{2,40}?)(?:\.|,|;|$)", "IS_A"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+are\s+(?:a|an|the)?\s*([\w\s]{2,40}?)(?:\.|,|;|$)", "IS_A"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+refers?\s+to\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "REFERS_TO"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+means?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "MEANS"),
    # HAS / CONTAINS
    (r"([A-Za-z][\w\s]{2,40}?)\s+has\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "HAS"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+contains?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "CONTAINS"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+includes?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "INCLUDES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+consists?\s+of\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "CONSISTS_OF"),
    # ACTION patterns
    (r"([A-Za-z][\w\s]{2,40}?)\s+uses?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "USES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+requires?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "REQUIRES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+provides?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "PROVIDES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+produces?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "PRODUCES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+converts?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "CONVERTS"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+stores?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "STORES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+enables?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "ENABLES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+supports?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "SUPPORTS"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+affects?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "AFFECTS"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+causes?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "CAUSES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+leads?\s+to\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "LEADS_TO"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+results?\s+in\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "RESULTS_IN"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+depends?\s+on\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "DEPENDS_ON"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+reduces?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "REDUCES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+increases?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "INCREASES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+improves?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "IMPROVES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+measures?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "MEASURES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+defines?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "DEFINES"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+implements?\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "IMPLEMENTS"),
    # COMPARISON
    (r"([A-Za-z][\w\s]{2,40}?)\s+differs?\s+from\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "DIFFERS_FROM"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+(?:is\s+)?similar\s+to\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "SIMILAR_TO"),
    # PART-OF
    (r"([A-Za-z][\w\s]{2,40}?)\s+(?:is\s+)?part\s+of\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "PART_OF"),
    (r"([A-Za-z][\w\s]{2,40}?)\s+belongs?\s+to\s+([\w\s]{2,40}?)(?:\.|,|;|$)", "BELONGS_TO"),
]

# Extended stop words: functional verbs, generic academic filler, and noisy fragments
STOP_WORDS = {
    "the", "a", "an", "is", "of", "in", "to", "and", "or", "it", "this",
    "that", "which", "with", "from", "by", "on", "at", "are", "was", "were",
    "be", "been", "has", "have", "had", "do", "does", "can", "will", "as",
    "for", "but", "not", "its", "also", "more", "into", "than", "their",
    "they", "these", "those", "such", "each", "would", "should", "could",
    "about", "between", "through", "where", "when", "how", "what", "there",
    "too", "faster", "slower", "better", "worse", "higher", "lower", "less",
    "more", "much", "many", "very", "most", "only", "well", "your", "mine",
    "our", "them", "both", "all", "any", "some", "each", "every", "other",
    "uses", "provides", "shows", "described", "discussed", "within", "without",
    "using", "given", "during", "across", "along", "after", "before", "while", "though",
    "decides", "performs", "refers", "means", "consists",
    # Academic Layout Noise
    "figure", "table", "section", "chapter", "page", "contents", "introduction",
    "conclusion", "appendix", "references", "summary", "example", "note", "exercise",
    "abstract", "keywords", "author", "authors", "date", "published", "journal"
}


def _clean(text: str) -> str:
    """Clean node labels and filter out low-value terms aggressively."""
    # 1. Basic normalization
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text) # Remove symbols
    
    # 2. Linguistic Pruning: Remove leading/trailing fragments often caught by regex
    # e.g., "than faster" -> "faster", "the system" -> "system"
    words = text.split()
    while words and (words[0] in STOP_WORDS or len(words[0]) < 2):
        words.pop(0)
    while words and (words[-1] in STOP_WORDS or len(words[-1]) < 2):
        words.pop(-1)
        
    cleaned = " ".join(words).strip()
    
    # 3. Quality Sanity Check
    # Reject if: empty, too short, only common words, or looks like a mangled PDF fragment (e.g. 'lpus')
    if not cleaned or len(cleaned) < 4:
        return ""
    if cleaned in STOP_WORDS:
        return ""
    
    # If the word has no vowels (in English) AND no Unicode non-ASCII chars, it's likely noise
    # This allows Kannada/Telugu/Hindi script to pass through even without "aeiouy"
    has_vowels = any(v in cleaned for v in "aeiouy")
    has_unicode = any(ord(char) > 127 for char in cleaned)
    
    if not has_vowels and not has_unicode and not cleaned.isupper():
        return ""

    return cleaned.title()[:50] # Camel Case looks more professional


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from raw text."""
    triples = []
    # Process sentence by sentence for better extraction
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        for pattern, relation in RELATION_PATTERNS:
            for m in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                subj = _clean(m.group(1))
                obj  = _clean(m.group(2))
                if len(subj) > 2 and len(obj) > 2 and subj != obj:
                    triples.append((subj, relation, obj))
    return triples


def _extract_noun_phrases(text: str) -> List[str]:
    """Fallback: extract capitalized noun phrases as concept nodes."""
    # Match sequences of capitalized words (likely concept names)
    phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    # Also match common technical terms (all-caps abbreviations)
    abbrevs = re.findall(r'\b([A-Z]{2,6})\b', text)
    all_concepts = list(set([p.lower() for p in phrases if len(p) > 3] +
                            [a.lower() for a in abbrevs if len(a) > 1]))
    return all_concepts[:50]


def build_graph(documents) -> nx.DiGraph:
    """
    Build a directed knowledge graph from LangChain Document objects.
    Each document's page_content is parsed for concept triples.
    Falls back to noun-phrase extraction if regex finds too few triples.
    """
    G = nx.DiGraph()
    for doc in documents:
        triples = extract_triples(doc.page_content)
        for subj, rel, obj in triples:
            G.add_node(subj, label=subj)
            G.add_node(obj,  label=obj)
            G.add_edge(subj, obj, relation=rel)

    print(f"[graph_builder] Graph has {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.")
    return G


def prune_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Retroactively clean an existing graph. 
    Removes noisy fragments and remaps nodes to their cleaned versions.
    """
    nodes_to_remove = []
    mapping = {}
    
    # We must use a list of nodes because we might modify the graph shape
    for node in list(G.nodes()):
        cleaned = _clean(node)
        if not cleaned:
            nodes_to_remove.append(node)
        elif cleaned != node:
            mapping[node] = cleaned
            
    # Apply mapping (merges nodes if they clean to the same thing)
    if mapping:
        import networkx as nx
        nx.relabel_nodes(G, mapping, copy=False)
        
    # Remove junk
    G.remove_nodes_from(nodes_to_remove)
    
    print(f"[graph_builder] Pruned graph. Removed {len(nodes_to_remove)} junk nodes. Remapped {len(mapping)} nodes.")
    return G


def get_subgraph(G: nx.DiGraph, seed_nodes: List[str], hops: int = 2) -> nx.DiGraph:
    """
    Extract a neighbourhood subgraph around seed_nodes (BFS up to `hops`).
    This is the core of Graph RAG — retrieve structured context.
    """
    visited = set()
    frontier = set(seed_nodes)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            if node in G:
                visited.add(node)
                next_frontier.update(G.predecessors(node))
                next_frontier.update(G.successors(node))
        frontier = next_frontier - visited
    visited.update(frontier)
    return G.subgraph(visited).copy()


def subgraph_to_text(subgraph: nx.DiGraph) -> str:
    """Serialize a subgraph into readable text for the LLM prompt."""
    lines = []
    for u, v, data in subgraph.edges(data=True):
        rel = data.get("relation", "RELATED_TO")
        lines.append(f"  {u}  --[{rel}]-->  {v}")
    return "\n".join(lines) if lines else "(no graph context found)"
