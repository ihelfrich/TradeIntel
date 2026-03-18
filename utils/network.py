"""
Network analysis utilities for trade networks.
Builds NetworkX graphs and computes centrality, communities, dependencies.
"""
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def build_trade_graph(trade_df: pd.DataFrame, weight_col: str = "trade_value_usd_millions") -> dict:
    G = nx.DiGraph()
    for _, row in trade_df.iterrows():
        G.add_edge(
            row["iso_o"], row["iso_d"],
            weight=row[weight_col],
        )
    return nx.node_link_data(G)


def graph_from_link_data(data: dict) -> nx.DiGraph:
    return nx.node_link_graph(data)


@st.cache_data(ttl=3600)
def compute_centrality_measures(graph_data: dict) -> pd.DataFrame:
    G = graph_from_link_data(graph_data)
    if len(G) == 0:
        return pd.DataFrame()

    degree_cent = nx.degree_centrality(G)
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)

    # Weighted measures
    strength_in = {}
    strength_out = {}
    for node in G.nodes():
        strength_in[node] = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        strength_out[node] = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))

    try:
        betweenness = nx.betweenness_centrality(G, weight="weight", k=min(100, len(G)))
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    try:
        pagerank = nx.pagerank(G, weight="weight")
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    df = pd.DataFrame({
        "degree_centrality": degree_cent,
        "in_degree_centrality": in_degree,
        "out_degree_centrality": out_degree,
        "betweenness_centrality": betweenness,
        "eigenvector_centrality": eigenvector,
        "pagerank": pagerank,
        "import_strength": strength_in,
        "export_strength": strength_out,
    })
    df.index.name = "iso3"
    df["total_strength"] = df["import_strength"] + df["export_strength"]
    return df.sort_values("total_strength", ascending=False)


@st.cache_data(ttl=3600)
def detect_communities(graph_data: dict) -> dict:
    G = graph_from_link_data(graph_data)
    G_undirected = G.to_undirected()

    # Use Louvain if available, otherwise greedy modularity
    try:
        communities = nx.community.louvain_communities(G_undirected, weight="weight", seed=42)
    except AttributeError:
        communities = list(nx.community.greedy_modularity_communities(G_undirected, weight="weight"))

    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    return community_map


@st.cache_data(ttl=3600)
def compute_dependency_metrics(trade_df: pd.DataFrame, country_iso3: str) -> dict:
    """Compute how dependent a country is on each trading partner."""
    exports = trade_df[trade_df["iso_o"] == country_iso3].copy()
    imports = trade_df[trade_df["iso_d"] == country_iso3].copy()

    total_exports = exports["trade_value_usd_millions"].sum()
    total_imports = imports["trade_value_usd_millions"].sum()

    # Export dependency: share of exports going to each partner
    if total_exports > 0:
        exports["export_share"] = exports["trade_value_usd_millions"] / total_exports
        export_dep = exports[["iso_d", "trade_value_usd_millions", "export_share"]].copy()
        export_dep.columns = ["partner", "export_value", "export_share"]
        # HHI for export concentration
        export_hhi = (exports["export_share"] ** 2).sum()
    else:
        export_dep = pd.DataFrame(columns=["partner", "export_value", "export_share"])
        export_hhi = 0

    # Import dependency
    if total_imports > 0:
        imports["import_share"] = imports["trade_value_usd_millions"] / total_imports
        import_dep = imports[["iso_o", "trade_value_usd_millions", "import_share"]].copy()
        import_dep.columns = ["partner", "import_value", "import_share"]
        import_hhi = (imports["import_share"] ** 2).sum()
    else:
        import_dep = pd.DataFrame(columns=["partner", "import_value", "import_share"])
        import_hhi = 0

    return {
        "total_exports": total_exports,
        "total_imports": total_imports,
        "trade_balance": total_exports - total_imports,
        "export_partners": len(exports),
        "import_partners": len(imports),
        "export_hhi": export_hhi,
        "import_hhi": import_hhi,
        "export_dependency": export_dep.sort_values("export_value", ascending=False),
        "import_dependency": import_dep.sort_values("import_value", ascending=False),
    }


@st.cache_data(ttl=3600)
def compute_network_stats_over_time(years: list, _trade_full: pd.DataFrame) -> pd.DataFrame:
    """Compute network-level statistics for each year."""
    records = []
    for year in years:
        df_year = _trade_full[_trade_full["year"] == year]
        if len(df_year) == 0:
            continue

        G = nx.DiGraph()
        for _, row in df_year.iterrows():
            G.add_edge(row["iso_o"], row["iso_d"], weight=row["trade_value_usd_millions"])

        total_trade = df_year["trade_value_usd_millions"].sum()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        max_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1

        # Concentration: share of top 10 countries in total trade
        country_trade = {}
        for n in G.nodes():
            ct = sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
            ct += sum(d["weight"] for _, _, d in G.in_edges(n, data=True))
            country_trade[n] = ct
        sorted_countries = sorted(country_trade.values(), reverse=True)
        top10_share = sum(sorted_countries[:10]) / (2 * total_trade) if total_trade > 0 else 0

        try:
            avg_clustering = nx.average_clustering(G.to_undirected(), weight="weight")
        except Exception:
            avg_clustering = 0

        records.append({
            "year": year,
            "countries": n_nodes,
            "trade_links": n_edges,
            "density": density,
            "total_trade_billions": total_trade / 1000,
            "top10_concentration": top10_share,
            "avg_clustering": avg_clustering,
        })

    return pd.DataFrame(records)


@st.cache_data(ttl=3600)
def compute_3d_force_layout(graph_data: dict, top_n: int = 60) -> dict:
    """
    Compute a 3D spring layout (Fruchterman-Reingold) for trade network visualization.
    Returns node positions, edge data, community assignments, and centrality for sizing.
    """
    G = graph_from_link_data(graph_data)

    # Keep only top N nodes by total trade
    strength = {}
    for n in G.nodes():
        s = sum(d["weight"] for _, _, d in G.in_edges(n, data=True))
        s += sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
        strength[n] = s
    top_nodes = sorted(strength, key=strength.get, reverse=True)[:top_n]
    G_sub = G.subgraph(top_nodes).copy()

    if len(G_sub) == 0:
        return {"nodes": [], "edges": [], "communities": {}}

    # Convert to undirected for layout
    G_und = G_sub.to_undirected()

    # Spring layout in 3D
    pos_3d = nx.spring_layout(
        G_und, dim=3, weight="weight", k=2.0 / (len(G_und) ** 0.5),
        iterations=100, seed=42,
    )

    # Community detection
    try:
        communities = nx.community.louvain_communities(G_und, weight="weight", seed=42)
    except AttributeError:
        communities = list(nx.community.greedy_modularity_communities(G_und, weight="weight"))

    comm_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = i

    # PageRank for node sizing
    try:
        pr = nx.pagerank(G_sub, weight="weight")
    except Exception:
        pr = {n: 1.0 / len(G_sub) for n in G_sub.nodes()}

    # Build serializable result
    nodes = []
    for node in G_sub.nodes():
        x, y, z = pos_3d[node]
        nodes.append({
            "id": node,
            "x": float(x), "y": float(y), "z": float(z),
            "community": comm_map.get(node, 0),
            "pagerank": float(pr.get(node, 0)),
            "strength": float(strength.get(node, 0)),
        })

    edges = []
    for u, v, d in G_sub.edges(data=True):
        if u in pos_3d and v in pos_3d:
            edges.append({
                "source": u, "target": v,
                "weight": float(d.get("weight", 1)),
                "x0": float(pos_3d[u][0]), "y0": float(pos_3d[u][1]), "z0": float(pos_3d[u][2]),
                "x1": float(pos_3d[v][0]), "y1": float(pos_3d[v][1]), "z1": float(pos_3d[v][2]),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "n_communities": len(communities),
        "community_sizes": [len(c) for c in communities],
    }


@st.cache_data(ttl=3600)
def compute_comprehensive_network_invariants(graph_data: dict) -> dict:
    """
    Compute a comprehensive set of network-theoretic invariants for the trade graph.
    Returns graph-level and node-level statistics.
    """
    G = graph_from_link_data(graph_data)
    G_und = G.to_undirected()
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {}

    # Basic graph invariants
    density = nx.density(G)
    reciprocity = nx.reciprocity(G) if m > 0 else 0

    # Connectivity
    n_strongly_connected = nx.number_strongly_connected_components(G)
    n_weakly_connected = nx.number_weakly_connected_components(G)
    largest_scc = max(nx.strongly_connected_components(G), key=len)

    # Clustering
    avg_clustering = nx.average_clustering(G_und, weight="weight")
    transitivity = nx.transitivity(G_und)

    # Degree distribution
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    # Assortativity (degree correlation)
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        degree_assortativity = 0.0

    # Small world metrics
    try:
        avg_path_length = nx.average_shortest_path_length(G.subgraph(largest_scc))
    except Exception:
        avg_path_length = float("inf")

    try:
        diameter = nx.diameter(G_und) if nx.is_connected(G_und) else -1
    except Exception:
        diameter = -1

    # Weighted strength distribution
    strengths = []
    for node in G.nodes():
        s_in = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        s_out = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        strengths.append(s_in + s_out)

    # Gini coefficient of trade (inequality measure)
    sorted_strengths = sorted(strengths)
    n_s = len(sorted_strengths)
    if n_s > 0 and sum(sorted_strengths) > 0:
        cumulative = np.cumsum(sorted_strengths)
        gini = (2 * np.sum((np.arange(1, n_s + 1) * sorted_strengths))) / (n_s * cumulative[-1]) - (n_s + 1) / n_s
    else:
        gini = 0.0

    # Core-periphery structure (k-core decomposition)
    core_numbers = nx.core_number(G_und)
    max_core = max(core_numbers.values()) if core_numbers else 0

    return {
        "n_nodes": n,
        "n_edges": m,
        "density": density,
        "reciprocity": reciprocity,
        "n_strongly_connected": n_strongly_connected,
        "n_weakly_connected": n_weakly_connected,
        "largest_scc_size": len(largest_scc),
        "avg_clustering": avg_clustering,
        "transitivity": transitivity,
        "degree_assortativity": degree_assortativity,
        "avg_path_length": avg_path_length,
        "diameter": diameter,
        "avg_in_degree": np.mean(in_degrees),
        "avg_out_degree": np.mean(out_degrees),
        "max_in_degree": max(in_degrees),
        "max_out_degree": max(out_degrees),
        "trade_gini": gini,
        "max_core_number": max_core,
        "mean_strength": np.mean(strengths),
        "median_strength": np.median(strengths),
        "strength_std": np.std(strengths),
    }
