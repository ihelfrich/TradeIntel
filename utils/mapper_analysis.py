"""
Mapper Algorithm for Trade Network Topology.

The Mapper algorithm (Singh, Mémoli, Carlsson 2007) constructs a simplicial
complex that is a compressed topological summary of the data. Applied to
trade networks, it reveals:
- Clusters of similar trading nations
- Transitions between trade regimes
- Outlier economies
- Structural loops in the trade landscape

The choice of filter function (lens) determines what aspect of trade
structure is revealed:
- total_trade: groups by economic size
- trade_balance: separates surplus vs deficit nations
- gravity_residual: groups by unexplained trade intensity
- eigenvector_centrality: groups by network influence
"""
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st

try:
    import kmapper as km
    from sklearn.cluster import DBSCAN
    HAS_KMAPPER = True
except ImportError:
    HAS_KMAPPER = False

from utils.topology import trade_to_distance_matrix
from utils.data_loader import COUNTRY_COORDS


def compute_filter_function(
    trade_df: pd.DataFrame,
    countries: list,
    W: np.ndarray,
    filter_type: str,
) -> np.ndarray:
    """Compute a 1D filter function over countries for the Mapper lens."""
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}

    if filter_type == "total_trade":
        # Total trade volume per country
        return W.sum(axis=0) + W.sum(axis=1)

    elif filter_type == "trade_balance":
        # Exports - imports (using asymmetric original)
        exports = np.zeros(n)
        imports = np.zeros(n)
        for _, row in trade_df.iterrows():
            i = idx.get(row["iso_o"])
            j = idx.get(row["iso_d"])
            if i is not None and j is not None:
                exports[i] += row["trade_value_usd_millions"]
                imports[j] += row["trade_value_usd_millions"]
        return exports - imports

    elif filter_type == "eigenvector_centrality":
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > 0:
                    G.add_edge(i, j, weight=W[i, j])
        try:
            eig = nx.eigenvector_centrality_numpy(G, weight="weight")
            return np.array([eig.get(i, 0) for i in range(n)])
        except Exception:
            return W.sum(axis=0)

    elif filter_type == "geographic_remoteness":
        # Average geographic distance to trade partners
        remoteness = np.zeros(n)
        for i, c_i in enumerate(countries):
            coords_i = COUNTRY_COORDS.get(c_i, (0, 0))
            if coords_i == (0, 0):
                remoteness[i] = 50  # default
                continue
            dists = []
            for j, c_j in enumerate(countries):
                if i != j and W[i, j] > 0:
                    coords_j = COUNTRY_COORDS.get(c_j, (0, 0))
                    if coords_j != (0, 0):
                        # Haversine-lite (sufficient for lens)
                        dlat = abs(coords_i[0] - coords_j[0])
                        dlon = abs(coords_i[1] - coords_j[1])
                        dists.append((dlat**2 + dlon**2) ** 0.5)
            remoteness[i] = np.mean(dists) if dists else 50
        return remoteness

    elif filter_type == "gravity_residual":
        # Mean gravity residual per country (how much does trade deviate from gravity?)
        try:
            from utils.gravity_model import compute_gravity_predicted_matrix
            grav = compute_gravity_predicted_matrix(year=2019, top_n=len(countries))
            R = grav["R_residual"]
            grav_countries = grav["countries"]
            grav_idx = {c: i for i, c in enumerate(grav_countries)}
            residuals = np.ones(n)
            for i, c in enumerate(countries):
                gi = grav_idx.get(c)
                if gi is not None:
                    row_residuals = R[gi, :]
                    row_residuals = row_residuals[row_residuals > 0]
                    if len(row_residuals) > 0:
                        residuals[i] = np.mean(row_residuals)
            return residuals
        except Exception:
            return W.sum(axis=0)  # fallback

    return W.sum(axis=0)  # default


@st.cache_data(ttl=3600)
def compute_trade_mapper(
    _trade_hash: str,
    trade_df_bytes: bytes,
    filter_function: str = "total_trade",
    top_n: int = 60,
    n_cubes: int = 10,
    perc_overlap: float = 0.3,
    distance_method: str = "negative_log",
) -> dict:
    """
    Apply the Mapper algorithm to the trade network.

    Returns a graph where nodes = clusters of countries with similar
    filter values, and edges = shared countries between overlapping clusters.
    """
    import io
    trade_df = pd.read_json(io.BytesIO(trade_df_bytes))
    D, countries = trade_to_distance_matrix(trade_df, method=distance_method, top_n_countries=top_n)
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}

    # Build weight matrix
    W = np.zeros((n, n))
    for _, row in trade_df.iterrows():
        i, j = idx.get(row["iso_o"]), idx.get(row["iso_d"])
        if i is not None and j is not None:
            W[i, j] += row["trade_value_usd_millions"]
    W_sym = np.maximum(W, W.T)

    # Compute filter values
    filter_vals = compute_filter_function(trade_df, countries, W_sym, filter_function)

    # Normalize distance matrix
    max_d = D[D > 0].max() if np.any(D > 0) else 1
    D_norm = D / max_d

    # Run KMapper
    mapper = km.KeplerMapper(verbose=0)
    lens = filter_vals.reshape(-1, 1)

    # Use DBSCAN on the precomputed distance matrix
    # precomputed=True tells KMapper to subset both rows AND columns
    graph = mapper.map(
        lens, D_norm,
        precomputed=True,
        clusterer=DBSCAN(eps=0.5, min_samples=2, metric="precomputed"),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
    )

    if not graph["nodes"]:
        return {"nodes": {}, "edges": [], "n_nodes": 0, "n_edges": 0,
                "countries": countries, "filter_function": filter_function}

    # Build NetworkX graph for layout
    G_mapper = nx.Graph()
    node_data = {}

    for node_id, member_indices in graph["nodes"].items():
        member_countries = [countries[i] for i in member_indices]
        member_trade = sum(filter_vals[i] for i in member_indices)
        G_mapper.add_node(node_id)
        node_data[node_id] = {
            "members": member_countries,
            "size": len(member_indices),
            "avg_filter": float(np.mean([filter_vals[i] for i in member_indices])),
            "total_trade_M": float(sum(W_sym[i].sum() for i in member_indices)),
            "indices": member_indices,
        }

    edges = []
    for node_id, connected in graph["links"].items():
        for target_id in connected:
            shared = set(graph["nodes"][node_id]) & set(graph["nodes"][target_id])
            G_mapper.add_edge(node_id, target_id, weight=len(shared))
            edges.append({
                "source": node_id,
                "target": target_id,
                "shared_countries": [countries[i] for i in shared],
                "n_shared": len(shared),
            })

    # Compute layout
    try:
        pos = nx.spring_layout(G_mapper, weight="weight", seed=42, k=2.0 / (len(G_mapper) ** 0.5 + 0.1))
    except Exception:
        pos = nx.random_layout(G_mapper, seed=42)

    # Add positions to node data
    for node_id in node_data:
        if node_id in pos:
            node_data[node_id]["x"] = float(pos[node_id][0])
            node_data[node_id]["y"] = float(pos[node_id][1])
        else:
            node_data[node_id]["x"] = 0.0
            node_data[node_id]["y"] = 0.0

    return {
        "nodes": node_data,
        "edges": edges,
        "n_nodes": len(node_data),
        "n_edges": len(edges),
        "countries": countries,
        "filter_function": filter_function,
        "filter_values": {c: float(filter_vals[i]) for i, c in enumerate(countries)},
        "graph_stats": {
            "n_components": nx.number_connected_components(G_mapper),
            "has_cycles": len(edges) >= len(node_data),
        },
    }
