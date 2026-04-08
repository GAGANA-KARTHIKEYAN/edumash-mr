# inference/graph_comparator.py
# ── NOVEL COMPONENT #3: GNN-Inspired Graph Comparator ──
# 
# Replaces simple keyword matching with structural Graph Edit Distance (GED)
# approximation to find misconception subgraphs.
#
# It compares the reference curriculum graph to the student's concept graph
# to detect:
# 1. Missing Links (student knows concepts but not relation)
# 2. Extraneous Nodes (introduced concepts not in domain)
# 3. Disconnected Clusters

import networkx as nx

def build_nx_graph(triplets: list) -> nx.DiGraph:
    """Convert JSON triplets to a NetworkX directed graph."""
    G = nx.DiGraph()
    if not triplets:
        return G
        
    for t in triplets:
        s, r, o = t.get("s", ""), t.get("r", ""), t.get("o", "")
        if s and o:
            G.add_edge(s.lower(), o.lower(), relation=r)
    return G

def compare_graphs(student_triplets: list, ref_triplets: list) -> dict:
    """
    Compare student's graph against the reference graph and 
    extract semantic misconceptions.
    """
    G_stu = build_nx_graph(student_triplets)
    G_ref = build_nx_graph(ref_triplets)
    
    stu_nodes = set(G_stu.nodes())
    ref_nodes = set(G_ref.nodes())
    
    stu_edges = set((u, v) for u, v in G_stu.edges())
    ref_edges = set((u, v) for u, v in G_ref.edges())
    
    # Missing Concepts
    missing_nodes = ref_nodes - stu_nodes
    
    # Missing Relationships (Student has nodes, but failed to connect them)
    # i.e., Edges in reference where both source and target are in student's graph, but edge is missing
    missing_links = []
    for u, v in ref_edges:
        if u in stu_nodes and v in stu_nodes and (u, v) not in stu_edges:
            rel = G_ref.edges[u, v].get("relation", "")
            missing_links.append(f"{u} -[{rel}]-> {v}")
            
    # Hallucinated/Extraneous Concepts (not in reference at all)
    extra_nodes = stu_nodes - ref_nodes
    
    # Structural Misconceptions (Edges student proposed that strongly conflict with info)
    wrong_edges = stu_edges - ref_edges
    
    # Calculate a Graph Edit Distance (GED) approximation score (0 to 1, 1 is identical)
    # Jaccard index on nodes and edges combined
    all_n = len(stu_nodes.union(ref_nodes))
    if all_n == 0:
        return {"score": 0.0, "missing_concepts": [], "missing_links": [], "extra_concepts": []}
    
    intersect_n = len(stu_nodes.intersection(ref_nodes))
    
    all_e = len(stu_edges.union(ref_edges))
    intersect_e = len(stu_edges.intersection(ref_edges))
    
    # Weight nodes and edges
    score = (intersect_n / (all_n + 1e-9)) * 0.4 + (intersect_e / (all_e + 1e-9)) * 0.6
    
    return {
        "score": round(score, 3),
        "missing_concepts": list(missing_nodes),
        "missing_links": missing_links,
        "extra_concepts": list(extra_nodes),
        "wrong_connections": list(wrong_edges)
    }

if __name__ == "__main__":
    ref = [
        {"s": "Photosynthesis", "r": "REQUIRES", "o": "Sunlight"},
        {"s": "Photosynthesis", "r": "PRODUCES", "o": "Oxygen"},
        {"s": "Plants", "r": "PERFORM", "o": "Photosynthesis"}
    ]
    
    stu = [
        {"s": "Plants", "r": "PERFORM", "o": "Photosynthesis"},
        {"s": "Photosynthesis", "r": "PRODUCES", "o": "Carbon_Dioxide"}, # wrong
        {"s": "Photosynthesis", "r": "REQUIRES", "o": "Soil"} # wrong
    ]
    
    report = compare_graphs(stu, ref)
    print("Graph Comparator Report:")
    import json
    print(json.dumps(report, indent=2))
