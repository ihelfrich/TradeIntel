"""
Persistent Homology & Topological Data Analysis of Trade Networks.

Computes topological invariants of the global trade network across
filtration thresholds and time, revealing structural features invisible
to standard network statistics.

Key constructions:
1. Weighted Rips filtration: trade intensity → distance matrix → Rips complex
2. Persistent homology: H_k for k=0,1,2
   - H₀: Connected components → trade blocs
   - H₁: 1-cycles → triangular/circular trade patterns
   - H₂: 2-voids → higher-order structural holes
3. Persistence diagrams, barcodes, Betti curves, landscapes
4. Bottleneck distance between diagrams across time
5. Euler characteristic χ = β₀ - β₁ + β₂ as complexity measure

References:
- Edelsbrunner & Harer (2010) "Computational Topology"
- Carlsson (2009) "Topology and Data"
- Zomorodian & Carlsson (2005) "Computing Persistent Homology"
"""
import numpy as np
import pandas as pd
import streamlit as st
import ripser
import gudhi
from gudhi.representations import Landscape
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def trade_to_distance_matrix(
    trade_df: pd.DataFrame,
    method: str = "negative_log",
    top_n_countries: int = 80,
) -> tuple:
    """
    Convert bilateral trade flows to a distance matrix for TDA.
    High trade ↔ small distance. Vectorized for speed.
    """
    # Get top N countries by total trade (vectorized)
    export_sums = trade_df.groupby("iso_o")["trade_value_usd_millions"].sum()
    import_sums = trade_df.groupby("iso_d")["trade_value_usd_millions"].sum()
    country_trade = export_sums.add(import_sums, fill_value=0).sort_values(ascending=False)
    top_countries = country_trade.head(top_n_countries).index.tolist()

    df = trade_df[
        trade_df["iso_o"].isin(top_countries) & trade_df["iso_d"].isin(top_countries)
    ]

    countries = sorted(top_countries)
    n = len(countries)
    idx = {c: i for i, c in enumerate(countries)}

    # Build adjacency via pivot (vectorized)
    df_indexed = df.copy()
    df_indexed["i"] = df_indexed["iso_o"].map(idx)
    df_indexed["j"] = df_indexed["iso_d"].map(idx)
    df_indexed = df_indexed.dropna(subset=["i", "j"])

    W = np.zeros((n, n))
    np.add.at(W, (df_indexed["i"].astype(int).values, df_indexed["j"].astype(int).values),
              df_indexed["trade_value_usd_millions"].values)

    # Symmetrize (max)
    W = np.maximum(W, W.T)

    # Convert to distance
    if method == "inverse":
        eps = np.percentile(W[W > 0], 1) if np.any(W > 0) else 1.0
        D = 1.0 / (W + eps)
    elif method == "log_inverse":
        D = 1.0 / np.log1p(W + 1)
    else:  # negative_log
        max_w = W.max() if W.max() > 0 else 1
        with np.errstate(divide="ignore"):
            D = -np.log(W / max_w + 1e-10)
        D = np.clip(D, 0, None)

    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0)
    return D, countries


REGION_MAP = {
    "NAM": ["USA", "CAN", "MEX"],
    "SAM": ["BRA", "ARG", "CHL", "COL", "PER", "VEN", "ECU", "URY", "BOL", "PRY"],
    "EUR": ["DEU", "FRA", "GBR", "ITA", "ESP", "NLD", "BEL", "CHE", "AUT", "POL",
            "SWE", "NOR", "DNK", "FIN", "IRL", "PRT", "CZE", "ROU", "HUN", "GRC",
            "UKR", "SVK", "BGR", "HRV", "LTU", "SVN", "LVA", "EST", "LUX", "TUR"],
    "EAS": ["CHN", "JPN", "KOR", "HKG", "TWN", "MNG"],
    "SEA": ["VNM", "THA", "MYS", "IDN", "PHL", "SGP", "MMR", "KHM", "LAO", "BRN"],
    "SAS": ["IND", "PAK", "BGD", "LKA", "NPL"],
    "MEN": ["SAU", "ARE", "ISR", "EGY", "IRQ", "IRN", "QAT", "KWT", "OMN", "JOR", "LBN"],
    "AFR": ["ZAF", "NGA", "KEN", "ETH", "GHA", "TZA", "AGO", "COD", "CMR", "CIV",
            "SEN", "MOZ", "MDG", "DZA", "MAR", "TUN", "LBY"],
    "OCE": ["AUS", "NZL"],
    "CAS": ["RUS", "KAZ", "UZB", "AZE", "GEO", "ARM", "TKM"],
}

def _get_region(iso: str) -> str:
    for region, countries in REGION_MAP.items():
        if iso in countries:
            return region
    return "OTH"


def _classify_cycle(countries: list) -> str:
    regions = set(_get_region(c) for c in countries)
    if len(countries) == 3:
        if len(regions) == 1:
            return "regional triangle"
        return "cross-regional triangle"
    if len(regions) == 1:
        return "regional circuit"
    return "cross-regional circuit"


@st.cache_data(ttl=3600)
def compute_attributed_persistent_homology(
    _trade_hash: str,
    trade_df_bytes: bytes,
    max_dim: int = 2,
    top_n: int = 60,
    distance_method: str = "negative_log",
) -> dict:
    """
    Persistent homology WITH economic attribution.

    Uses GUDHI's persistence_pairs() to map each topological feature
    back to specific countries and dollar-valued trade flows:
    - H₀ merges: which trade blocs merged, triggered by which bilateral link
    - H₁ cycles: which countries form each trade cycle, with edge values
    - H₂ voids: which country groups have incomplete multilateral integration
    """
    import io
    trade_df = pd.read_json(io.BytesIO(trade_df_bytes))
    D, countries = trade_to_distance_matrix(trade_df, method=distance_method, top_n_countries=top_n)
    n = len(countries)

    # Build weight matrix for dollar attribution
    W = np.zeros((n, n))
    idx = {c: i for i, c in enumerate(countries)}
    for _, row in trade_df.iterrows():
        i, j = idx.get(row["iso_o"]), idx.get(row["iso_d"])
        if i is not None and j is not None:
            W[i, j] += row["trade_value_usd_millions"]
    W = np.maximum(W, W.T)  # symmetrize

    # Normalize distances
    max_d = D[D > 0].max() if np.any(D > 0) else 1
    D_norm = D / max_d

    # Build GUDHI simplex tree for persistence_pairs
    rips = gudhi.RipsComplex(distance_matrix=D_norm.tolist(), max_edge_length=1.0)
    stree = rips.create_simplex_tree(max_dimension=max_dim + 1)
    stree.compute_persistence()

    # Also run ripser for standard diagrams (faster, cleaner output)
    import ripser as _ripser
    rips_result = _ripser.ripser(D_norm, maxdim=max_dim, distance_matrix=True)
    diagrams = rips_result["dgms"]

    # ── H₀ merges via union-find ──
    parent = list(range(n))
    rank = [0] * n
    members = {i: [countries[i]] for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        members[rx] = members[rx] + members[ry]
        del members[ry]
        return True

    h0_merges = []
    # Walk filtration, track edge additions
    filtration_edges = []
    for simplex, filt in stree.get_filtration():
        if len(simplex) == 2:
            filtration_edges.append((simplex[0], simplex[1], filt))

    filtration_edges.sort(key=lambda x: x[2])
    for i_v, j_v, filt in filtration_edges:
        comp_a = list(members.get(find(i_v), []))
        comp_b = list(members.get(find(j_v), []))
        if union(i_v, j_v):
            h0_merges.append({
                "filtration": float(filt),
                "edge": (countries[i_v], countries[j_v]),
                "trade_value_M": float(W[i_v, j_v]),
                "component_a": comp_a,
                "component_b": comp_b,
                "component_a_size": len(comp_a),
                "component_b_size": len(comp_b),
            })

    # ── H₁ cycles via persistence_pairs ──
    h1_cycles = []
    pairs = stree.persistence_pairs()
    for birth_simplex, death_simplex in pairs:
        if len(birth_simplex) == 2 and len(death_simplex) == 3:
            # H₁: birth is an edge, death is a triangle
            tri_vertices = death_simplex
            tri_countries = [countries[v] for v in tri_vertices]
            birth_filt = stree.filtration(birth_simplex)
            death_filt = stree.filtration(death_simplex)
            persistence = death_filt - birth_filt

            # Edge trade values
            edges = [(tri_vertices[0], tri_vertices[1]),
                     (tri_vertices[0], tri_vertices[2]),
                     (tri_vertices[1], tri_vertices[2])]
            edge_values = [float(W[e[0], e[1]]) for e in edges]
            edge_labels = [(countries[e[0]], countries[e[1]]) for e in edges]

            h1_cycles.append({
                "birth": float(birth_filt),
                "death": float(death_filt),
                "persistence": float(persistence),
                "countries": tri_countries,
                "edges": edge_labels,
                "trade_values_M": edge_values,
                "total_trade_M": sum(edge_values),
                "classification": _classify_cycle(tri_countries),
                "regions": list(set(_get_region(c) for c in tri_countries)),
            })

    h1_cycles.sort(key=lambda x: x["persistence"], reverse=True)

    # ── H₂ voids via persistence_pairs ──
    h2_voids = []
    for birth_simplex, death_simplex in pairs:
        if len(birth_simplex) == 3 and len(death_simplex) == 4:
            # H₂: birth is triangle, death is tetrahedron
            tet_countries = [countries[v] for v in death_simplex]
            birth_filt = stree.filtration(birth_simplex)
            death_filt = stree.filtration(death_simplex)
            h2_voids.append({
                "birth": float(birth_filt),
                "death": float(death_filt),
                "persistence": float(death_filt - birth_filt),
                "countries": tet_countries,
                "classification": "structural void — incomplete multilateral integration",
            })

    h2_voids.sort(key=lambda x: x["persistence"], reverse=True)

    # Standard persistence data from ripser (for diagrams/Betti curves)
    persistence_data = {}
    betti_curves = {}
    n_points = 100
    filtration_values = np.linspace(0, 1, n_points)

    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        finite = dgm[dgm[:, 1] < np.inf]
        infinite = dgm[dgm[:, 1] == np.inf]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            persistence_data[str(dim)] = {
                "births": finite[:, 0].tolist(),
                "deaths": finite[:, 1].tolist(),
                "n_features": len(finite),
                "total_persistence": float(lifetimes.sum()),
                "n_infinite": len(infinite),
            }
        else:
            persistence_data[str(dim)] = {
                "births": [], "deaths": [],
                "n_features": 0, "total_persistence": 0,
                "n_infinite": len(infinite),
            }
        # Betti curve
        births = dgm[:, 0]
        deaths = np.where(dgm[:, 1] == np.inf, 1.1, dgm[:, 1])
        curve = np.zeros(n_points)
        for b, d in zip(births, deaths):
            curve[(filtration_values >= b) & (filtration_values < d)] += 1
        betti_curves[str(dim)] = curve.tolist()

    euler_curve = np.zeros(n_points)
    for dim in range(max_dim + 1):
        sign = 1 if dim % 2 == 0 else -1
        euler_curve += sign * np.array(betti_curves[str(dim)])

    return {
        "diagrams_json": {str(dim): {"births": persistence_data[str(dim)]["births"],
                                      "deaths": persistence_data[str(dim)]["deaths"]}
                          for dim in range(max_dim + 1)},
        "persistence_data": persistence_data,
        "betti_curves": betti_curves,
        "euler_curve": euler_curve.tolist(),
        "filtration_values": filtration_values.tolist(),
        "countries": countries,
        "n_countries": n,
        # The new attribution data:
        "h0_merges": h0_merges,
        "h1_cycles": h1_cycles,
        "h2_voids": h2_voids,
        "n_h1_cycles": len(h1_cycles),
        "n_h2_voids": len(h2_voids),
    }


@st.cache_data(ttl=7200)
def compute_topological_null_models(
    _trade_hash: str,
    trade_df_bytes: bytes,
    n_simulations: int = 100,
    top_n: int = 50,
    distance_method: str = "negative_log",
) -> dict:
    """
    Test whether the observed topology is statistically significant.

    Three null models:
    1. Erdős-Rényi: random graph with same density + empirical weight distribution
    2. Configuration: random graph with same degree sequence
    3. Gravity: topology of the gravity-predicted network (deterministic)

    Returns p-values and z-scores for each Betti number.
    """
    import io
    trade_df = pd.read_json(io.BytesIO(trade_df_bytes))
    D_obs, countries = trade_to_distance_matrix(trade_df, method=distance_method, top_n_countries=top_n)
    n = len(countries)
    max_d = D_obs[D_obs > 0].max() if np.any(D_obs > 0) else 1
    D_obs_norm = D_obs / max_d

    # Observed topology
    obs_result = ripser.ripser(D_obs_norm, maxdim=2, distance_matrix=True)
    observed = {}
    for dim in range(3):
        dgm = obs_result["dgms"][dim]
        finite = dgm[dgm[:, 1] < np.inf]
        observed[f"beta_{dim}"] = len(dgm)
        observed[f"total_persistence_{dim}"] = float((finite[:, 1] - finite[:, 0]).sum()) if len(finite) > 0 else 0.0

    # Build weight matrix for null model generation
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((n, n))
    for _, row in trade_df.iterrows():
        i, j = idx.get(row["iso_o"]), idx.get(row["iso_d"])
        if i is not None and j is not None:
            W[i, j] += row["trade_value_usd_millions"]
    W_sym = np.maximum(W, W.T)

    # Empirical weight distribution (for ER null)
    nonzero_weights = W_sym[W_sym > 0].flatten()
    edge_density = (W_sym > 0).sum() / (n * (n - 1)) if n > 1 else 0

    null_distributions = {"erdos_renyi": {}, "configuration": {}}
    for key in ["beta_0", "beta_1", "beta_2", "total_persistence_0", "total_persistence_1", "total_persistence_2"]:
        null_distributions["erdos_renyi"][key] = []
        null_distributions["configuration"][key] = []

    rng = np.random.RandomState(42)

    for sim in range(n_simulations):
        # ── Erdős-Rényi null ──
        W_er = np.zeros((n, n))
        mask = rng.random((n, n)) < edge_density
        np.fill_diagonal(mask, False)
        mask = np.triu(mask) | np.triu(mask).T  # symmetrize
        weights = rng.choice(nonzero_weights, size=mask.sum(), replace=True)
        W_er[mask] = weights[:mask.sum()]

        # Convert to distance
        max_w_er = W_er.max() if W_er.max() > 0 else 1
        with np.errstate(divide="ignore"):
            D_er = -np.log(W_er / max_w_er + 1e-10)
        D_er = np.clip(D_er, 0, None)
        D_er = np.minimum(D_er, D_er.T)
        np.fill_diagonal(D_er, 0)
        max_d_er = D_er[D_er > 0].max() if np.any(D_er > 0) else 1
        D_er_norm = D_er / max_d_er

        er_result = ripser.ripser(D_er_norm, maxdim=2, distance_matrix=True)
        for dim in range(3):
            dgm = er_result["dgms"][dim]
            finite = dgm[dgm[:, 1] < np.inf]
            null_distributions["erdos_renyi"][f"beta_{dim}"].append(len(dgm))
            null_distributions["erdos_renyi"][f"total_persistence_{dim}"].append(
                float((finite[:, 1] - finite[:, 0]).sum()) if len(finite) > 0 else 0.0
            )

        # ── Configuration model null ──
        degrees = (W_sym > 0).sum(axis=1)
        import networkx as nx
        try:
            G_config = nx.configuration_model(degrees.tolist(), seed=sim)
            G_config = nx.Graph(G_config)  # remove multi-edges
            G_config.remove_edges_from(nx.selfloop_edges(G_config))
            W_config = np.zeros((n, n))
            for u, v in G_config.edges():
                if u < n and v < n:
                    w = rng.choice(nonzero_weights)
                    W_config[u, v] = w
                    W_config[v, u] = w
        except Exception:
            W_config = W_er  # fallback

        max_w_c = W_config.max() if W_config.max() > 0 else 1
        with np.errstate(divide="ignore"):
            D_config = -np.log(W_config / max_w_c + 1e-10)
        D_config = np.clip(D_config, 0, None)
        D_config = np.minimum(D_config, D_config.T)
        np.fill_diagonal(D_config, 0)
        max_d_c = D_config[D_config > 0].max() if np.any(D_config > 0) else 1
        D_config_norm = D_config / max_d_c

        config_result = ripser.ripser(D_config_norm, maxdim=2, distance_matrix=True)
        for dim in range(3):
            dgm = config_result["dgms"][dim]
            finite = dgm[dgm[:, 1] < np.inf]
            null_distributions["configuration"][f"beta_{dim}"].append(len(dgm))
            null_distributions["configuration"][f"total_persistence_{dim}"].append(
                float((finite[:, 1] - finite[:, 0]).sum()) if len(finite) > 0 else 0.0
            )

    # ── Gravity null (deterministic) ──
    from utils.gravity_model import compute_gravity_predicted_matrix
    try:
        grav = compute_gravity_predicted_matrix(year=2019, top_n=top_n)
        W_grav = grav["W_predicted"]
        # Align countries: the gravity matrix may have different country ordering
        grav_idx = {c: i for i, c in enumerate(grav["countries"])}
        W_grav_aligned = np.zeros((n, n))
        for i_c, c_i in enumerate(countries):
            for j_c, c_j in enumerate(countries):
                gi, gj = grav_idx.get(c_i), grav_idx.get(c_j)
                if gi is not None and gj is not None:
                    W_grav_aligned[i_c, j_c] = W_grav[gi, gj]

        W_grav_sym = np.maximum(W_grav_aligned, W_grav_aligned.T)
        max_w_g = W_grav_sym.max() if W_grav_sym.max() > 0 else 1
        with np.errstate(divide="ignore"):
            D_grav = -np.log(W_grav_sym / max_w_g + 1e-10)
        D_grav = np.clip(D_grav, 0, None)
        D_grav = np.minimum(D_grav, D_grav.T)
        np.fill_diagonal(D_grav, 0)
        max_d_g = D_grav[D_grav > 0].max() if np.any(D_grav > 0) else 1
        D_grav_norm = D_grav / max_d_g

        grav_result = ripser.ripser(D_grav_norm, maxdim=2, distance_matrix=True)
        gravity_observed = {}
        gravity_diagrams = {}
        for dim in range(3):
            dgm = grav_result["dgms"][dim]
            finite = dgm[dgm[:, 1] < np.inf]
            gravity_observed[f"beta_{dim}"] = len(dgm)
            gravity_observed[f"total_persistence_{dim}"] = float((finite[:, 1] - finite[:, 0]).sum()) if len(finite) > 0 else 0.0
            gravity_diagrams[str(dim)] = {
                "births": finite[:, 0].tolist() if len(finite) > 0 else [],
                "deaths": finite[:, 1].tolist() if len(finite) > 0 else [],
            }
    except Exception:
        gravity_observed = {f"beta_{d}": 0 for d in range(3)}
        gravity_observed.update({f"total_persistence_{d}": 0 for d in range(3)})
        gravity_diagrams = {str(d): {"births": [], "deaths": []} for d in range(3)}

    # Compute p-values and z-scores
    p_values = {}
    z_scores = {}
    for null_type in ["erdos_renyi", "configuration"]:
        p_values[null_type] = {}
        z_scores[null_type] = {}
        for key in ["beta_0", "beta_1", "beta_2"]:
            null_vals = np.array(null_distributions[null_type][key])
            obs_val = observed[key]
            p_values[null_type][key] = float((np.sum(null_vals >= obs_val) + 1) / (n_simulations + 1))
            std = null_vals.std()
            z_scores[null_type][key] = float((obs_val - null_vals.mean()) / std) if std > 0 else 0.0

    return {
        "observed": observed,
        "null_distributions": null_distributions,
        "gravity_topology": gravity_observed,
        "gravity_diagrams": gravity_diagrams,
        "p_values": p_values,
        "z_scores": z_scores,
        "n_simulations": n_simulations,
    }


@st.cache_data(ttl=3600)
def compute_persistent_homology(
    _trade_df_hash: str,
    trade_df_bytes: bytes,
    max_dim: int = 2,
    top_n: int = 60,
    distance_method: str = "negative_log",
) -> dict:
    """
    Compute persistent homology of the trade network via Rips filtration.
    """
    import io
    trade_df = pd.read_json(io.BytesIO(trade_df_bytes))
    D, countries = trade_to_distance_matrix(trade_df, method=distance_method, top_n_countries=top_n)

    max_d = D[D > 0].max() if np.any(D > 0) else 1
    D_norm = D / max_d

    result = ripser.ripser(D_norm, maxdim=max_dim, distance_matrix=True)
    diagrams = result["dgms"]

    persistence_data = {}
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        finite = dgm[dgm[:, 1] < np.inf]
        infinite = dgm[dgm[:, 1] == np.inf]

        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            persistence_data[str(dim)] = {
                "births": finite[:, 0].tolist(),
                "deaths": finite[:, 1].tolist(),
                "lifetimes": lifetimes.tolist(),
                "n_features": len(finite),
                "n_persistent": int(np.sum(lifetimes > np.median(lifetimes))),
                "max_lifetime": float(lifetimes.max()),
                "mean_lifetime": float(lifetimes.mean()),
                "total_persistence": float(lifetimes.sum()),
                "n_infinite": len(infinite),
            }
        else:
            persistence_data[str(dim)] = {
                "births": [], "deaths": [], "lifetimes": [],
                "n_features": 0, "n_persistent": 0,
                "max_lifetime": 0, "mean_lifetime": 0,
                "total_persistence": 0, "n_infinite": len(infinite),
            }

    # Betti curves
    n_points = 100
    filtration_values = np.linspace(0, 1, n_points)
    betti_curves = {}
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        births = dgm[:, 0]
        deaths = np.where(dgm[:, 1] == np.inf, 1.1, dgm[:, 1])
        curve = np.zeros(n_points)
        for b, d in zip(births, deaths):
            curve[(filtration_values >= b) & (filtration_values < d)] += 1
        betti_curves[str(dim)] = curve.tolist()

    euler_curve = np.zeros(n_points)
    for dim in range(max_dim + 1):
        sign = 1 if dim % 2 == 0 else -1
        euler_curve += sign * np.array(betti_curves[dim])

    # Persistence landscape (H₁)
    landscapes_h1 = []
    if len(diagrams[1]) > 0:
        finite_h1 = diagrams[1][diagrams[1][:, 1] < np.inf]
        if len(finite_h1) > 0:
            try:
                landscape = Landscape(num_landscapes=5, resolution=100).fit_transform([finite_h1])
                landscapes_h1 = landscape[0].tolist()
            except Exception:
                pass

    return {
        "diagrams_json": {str(dim): {"births": persistence_data[str(dim)]["births"],
                                      "deaths": persistence_data[str(dim)]["deaths"]}
                          for dim in range(max_dim + 1)},
        "persistence_data": persistence_data,
        "betti_curves": betti_curves,
        "euler_curve": euler_curve.tolist(),
        "filtration_values": filtration_values.tolist(),
        "landscapes_h1": landscapes_h1,
        "countries": countries,
        "n_countries": len(countries),
        "distance_method": distance_method,
    }


@st.cache_data(ttl=3600)
def compute_topological_evolution(
    _trade_full_hash: str,
    trade_full_bytes: bytes,
    years: list,
    top_n: int = 50,
    distance_method: str = "negative_log",
) -> dict:
    """Track topological invariants of the trade network over time."""
    import io
    trade_full = pd.read_json(io.BytesIO(trade_full_bytes))
    records = []
    diagrams_by_year = {}

    for year in years:
        df_year = trade_full[trade_full["year"] == year]
        if len(df_year) == 0:
            continue

        D, countries = trade_to_distance_matrix(df_year, method=distance_method, top_n_countries=top_n)
        max_d = D[D > 0].max() if np.any(D > 0) else 1
        D_norm = D / max_d

        dgms = ripser.ripser(D_norm, maxdim=2, distance_matrix=True)["dgms"]
        diagrams_by_year[year] = dgms

        record = {"year": year, "n_countries": len(countries)}
        for dim in range(3):
            dgm = dgms[dim]
            finite = dgm[dgm[:, 1] < np.inf]
            infinite = dgm[dgm[:, 1] == np.inf]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                record[f"beta_{dim}"] = len(finite) + len(infinite)
                record[f"total_persistence_{dim}"] = float(lifetimes.sum())
                record[f"max_persistence_{dim}"] = float(lifetimes.max())
            else:
                record[f"beta_{dim}"] = len(infinite)
                record[f"total_persistence_{dim}"] = 0
                record[f"max_persistence_{dim}"] = 0

        # Euler at midpoint
        euler = 0
        for dim in range(3):
            alive = sum(1 for b, d in dgms[dim] if b <= 0.5 and (d > 0.5 or d == np.inf))
            euler += ((-1) ** dim) * alive
        record["euler_char"] = euler
        records.append(record)

    # Bottleneck distances
    sorted_years = sorted(diagrams_by_year.keys())
    bottleneck_records = []
    for i in range(len(sorted_years) - 1):
        y1, y2 = sorted_years[i], sorted_years[i + 1]
        for dim in range(3):
            f1 = diagrams_by_year[y1][dim]
            f2 = diagrams_by_year[y2][dim]
            f1 = f1[f1[:, 1] < np.inf]
            f2 = f2[f2[:, 1] < np.inf]
            try:
                dist = gudhi.bottleneck_distance(f1.tolist(), f2.tolist()) if len(f1) > 0 and len(f2) > 0 else 0
            except Exception:
                dist = 0
            bottleneck_records.append({"year_from": y1, "year_to": y2, "dimension": dim, "bottleneck_distance": dist})

    return {
        "evolution": pd.DataFrame(records).to_json(),
        "bottleneck": pd.DataFrame(bottleneck_records).to_json(),
    }


def compute_clique_complex_stats(trade_df: pd.DataFrame, top_n: int = 50, threshold_percentile: float = 75) -> dict:
    """
    Compute clique complex (flag complex) statistics.
    K(G) has a k-simplex for every (k+1)-clique in G.
    """
    D, countries = trade_to_distance_matrix(trade_df, method="negative_log", top_n_countries=top_n)
    max_d = D[D > 0].max() if np.any(D > 0) else 1
    D_norm = D / max_d

    threshold = np.percentile(D_norm[D_norm > 0], threshold_percentile)

    rips = gudhi.RipsComplex(distance_matrix=D_norm.tolist(), max_edge_length=threshold)
    st_tree = rips.create_simplex_tree(max_dimension=4)

    simplex_counts = {}
    for simplex, filtration in st_tree.get_filtration():
        dim = len(simplex) - 1
        simplex_counts[dim] = simplex_counts.get(dim, 0) + 1

    euler = sum((-1) ** k * count for k, count in simplex_counts.items())
    f_vector = [simplex_counts.get(k, 0) for k in range(max(simplex_counts.keys()) + 1)] if simplex_counts else [0]

    st_tree.compute_persistence()
    betti = st_tree.betti_numbers()

    triangles = []
    for simplex, filt in st_tree.get_filtration():
        if len(simplex) == 3:
            triangles.append(([countries[i] for i in simplex], filt))
    triangles.sort(key=lambda x: x[1])

    return {
        "f_vector": f_vector,
        "euler_characteristic": euler,
        "betti_numbers": betti,
        "simplex_counts": simplex_counts,
        "n_vertices": simplex_counts.get(0, 0),
        "n_edges": simplex_counts.get(1, 0),
        "n_triangles": simplex_counts.get(2, 0),
        "n_tetrahedra": simplex_counts.get(3, 0),
        "n_4simplices": simplex_counts.get(4, 0),
        "total_simplices": sum(simplex_counts.values()),
        "top_triangles": triangles[:20],
        "threshold": threshold,
        "countries": countries,
    }


@st.cache_data(ttl=3600)
def topological_tariff_sensitivity(
    _trade_hash: str,
    trade_df_bytes: bytes,
    imposing_iso: str,
    target_iso: str,
    tariff_range: list,
    elasticity: float = 5.0,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Measure how the TOPOLOGY changes under tariff shocks. Cached for speed.
    """
    import io
    trade_df = pd.read_json(io.BytesIO(trade_df_bytes))
    records = []

    for tariff_pct in tariff_range:
        df_shocked = trade_df.copy()

        if tariff_pct > 0:
            mask_imp = (df_shocked["iso_o"] == target_iso) & (df_shocked["iso_d"] == imposing_iso)
            mask_exp = (df_shocked["iso_o"] == imposing_iso) & (df_shocked["iso_d"] == target_iso)
            reduction_factor = (1 + tariff_pct / 100) ** (-elasticity)
            df_shocked.loc[mask_imp, "trade_value_usd_millions"] *= reduction_factor
            df_shocked.loc[mask_exp, "trade_value_usd_millions"] *= reduction_factor

        D, countries = trade_to_distance_matrix(df_shocked, method="negative_log", top_n_countries=top_n)
        max_d = D[D > 0].max() if np.any(D > 0) else 1
        D_norm = D / max_d

        dgms = ripser.ripser(D_norm, maxdim=2, distance_matrix=True)["dgms"]

        record = {"tariff_pct": tariff_pct}
        for dim in range(3):
            dgm = dgms[dim]
            finite = dgm[dgm[:, 1] < np.inf]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                record[f"beta_{dim}"] = len(dgm)
                record[f"total_persistence_{dim}"] = float(lifetimes.sum())
                record[f"max_persistence_{dim}"] = float(lifetimes.max())
            else:
                record[f"beta_{dim}"] = len(dgm)
                record[f"total_persistence_{dim}"] = 0
                record[f"max_persistence_{dim}"] = 0

        euler = 0
        for dim in range(3):
            alive = sum(1 for b, d in dgms[dim] if b <= 0.5 and (d > 0.5 or d == np.inf))
            euler += ((-1) ** dim) * alive
        record["euler_char"] = euler
        records.append(record)

    return pd.DataFrame(records)
