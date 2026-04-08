# utils/graph_renderer.py
# Renders NetworkX graphs as interactive HTML using pyvis for Streamlit embedding.

import networkx as nx
import tempfile, os

def render_knowledge_graph_html(G: nx.DiGraph, title: str = "Knowledge Graph", height: int = 500) -> str:
    """
    Convert a NetworkX DiGraph into an interactive pyvis HTML string
    for embedding in Streamlit via st.components.v1.html().
    """
    from pyvis.network import Network

    net = Network(
        height=f"{height}px", width="100%",
        directed=True, notebook=False,
        bgcolor="#0f0c29", font_color="#e0e0ff",
    )
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=120)

    # Color scheme for node types
    colors = [
        "#6366f1", "#8b5cf6", "#a78bfa", "#c084fc",
        "#818cf8", "#7c3aed", "#5b21b6", "#4f46e5",
        "#60a5fa", "#38bdf8", "#22d3ee", "#2dd4bf",
    ]

    for i, node in enumerate(G.nodes()):
        label = str(node).title()[:30]
        degree = G.degree(node)
        size = max(15, min(40, 10 + degree * 5))
        color = colors[i % len(colors)]
        net.add_node(
            node, label=label, size=size, color=color,
            font={"size": 12, "color": "#e0e0ff"},
            borderWidth=2, borderWidthSelected=4,
            shadow=True,
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "RELATED_TO")
        net.add_edge(u, v, title=rel, label=rel, color="#4a4a7a",
                     font={"size": 9, "color": "#94a3b8", "align": "middle"},
                     arrows="to", width=1.5, smooth={"type": "curvedCW", "roundness": 0.2})

    # Generate HTML string
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)

    # Inject custom title bar
    title_bar = f"""
    <div style="text-align:center; padding:8px 0 4px 0; font-family:'Inter',sans-serif;">
      <span style="font-size:1.1rem; font-weight:600; color:#a5b4fc;">🔗 {title}</span>
      <span style="font-size:0.8rem; color:#64748b; margin-left:12px;">
        {G.number_of_nodes()} nodes · {G.number_of_edges()} edges
      </span>
    </div>
    """
    html = html.replace("<body>", f"<body>{title_bar}", 1)
    return html


def render_comparison_graph_html(
    student_triplets: list,
    ref_triplets: list,
    height: int = 450,
) -> str:
    """
    Render a side-by-side comparison-style graph showing:
    - Reference concepts (blue)
    - Student concepts (green if correct, red if wrong/missing)
    - Missing links (dashed red)
    """
    from pyvis.network import Network

    net = Network(
        height=f"{height}px", width="100%",
        directed=True, notebook=False,
        bgcolor="#0f0c29", font_color="#e0e0ff",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.35, spring_length=100)

    # Build sets for comparison
    ref_nodes = set()
    ref_edges = set()
    stu_nodes = set()
    stu_edges = set()

    for t in (ref_triplets or []):
        s, o = t.get("s", "").lower(), t.get("o", "").lower()
        if s and o:
            ref_nodes.update([s, o])
            ref_edges.add((s, o, t.get("r", "")))

    for t in (student_triplets or []):
        s, o = t.get("s", "").lower(), t.get("o", "").lower()
        if s and o:
            stu_nodes.update([s, o])
            stu_edges.add((s, o, t.get("r", "")))

    all_nodes = ref_nodes | stu_nodes

    for node in all_nodes:
        label = node.title()[:25]
        if node in ref_nodes and node in stu_nodes:
            # Student got this concept — green
            color = "#22c55e"
            border = "#166534"
        elif node in ref_nodes:
            # Missing from student — red
            color = "#ef4444"
            border = "#991b1b"
        else:
            # Extra concept from student (hallucination) — amber
            color = "#f59e0b"
            border = "#92400e"

        net.add_node(node, label=label, size=22, color=color,
                     font={"size": 11, "color": "#e0e0ff"},
                     borderWidth=2, borderWidthSelected=4,
                     shadow=True)

    # Reference edges
    for s, o, r in ref_edges:
        if (s, o, r) in stu_edges:
            # Correct relationship
            net.add_edge(s, o, title=f"✅ {r}", label=r,
                         color="#22c55e", width=2, arrows="to",
                         font={"size": 8, "color": "#86efac"},
                         smooth={"type": "curvedCW", "roundness": 0.15})
        else:
            # Missing relationship
            net.add_edge(s, o, title=f"❌ MISSING: {r}", label=f"❌ {r}",
                         color="#ef4444", width=2, arrows="to",
                         dashes=True,
                         font={"size": 8, "color": "#fca5a5"},
                         smooth={"type": "curvedCW", "roundness": 0.15})

    # Student-only edges (hallucinated)
    for s, o, r in stu_edges:
        if (s, o, r) not in ref_edges:
            net.add_edge(s, o, title=f"⚠️ EXTRA: {r}", label=f"⚠️ {r}",
                         color="#f59e0b", width=1.5, arrows="to",
                         dashes=[5, 5],
                         font={"size": 8, "color": "#fcd34d"},
                         smooth={"type": "curvedCCW", "roundness": 0.2})

    # Generate HTML
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)

    # Legend + title
    legend = """
    <div style="text-align:center; padding:8px 0 2px 0; font-family:'Inter',sans-serif;">
      <span style="font-size:1.05rem; font-weight:600; color:#a5b4fc;">🧠 Misconception Graph Comparison</span><br>
      <span style="font-size:0.75rem;">
        <span style="color:#22c55e;">● Correct Concept</span> &nbsp;
        <span style="color:#ef4444;">● Missing Concept</span> &nbsp;
        <span style="color:#f59e0b;">● Extra/Hallucinated</span> &nbsp;
        <span style="color:#22c55e;">─ Correct Link</span> &nbsp;
        <span style="color:#ef4444;">┅ Missing Link</span> &nbsp;
        <span style="color:#f59e0b;">┅ Hallucinated Link</span>
      </span>
    </div>
    """
    html = html.replace("<body>", f"<body>{legend}", 1)
    return html
