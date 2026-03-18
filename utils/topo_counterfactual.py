"""
Topology-Counterfactual Bridge.

Connects the GE counterfactual engine to topological data analysis.
Answers: "How does the topology of the global trade network change
under different tariff regimes?"

This is the novel research contribution — no existing paper combines
structural trade models with persistent homology on counterfactual networks.

Key analyses:
1. Compute PH on factual vs counterfactual trade flows
2. Identify which H₁ cycles survive/die under tariff wars
3. Compare Betti numbers across tariff scenarios
4. Track topological complexity as a function of tariff rate
"""

import numpy as np
import streamlit as st
import hashlib
from scipy.spatial.distance import squareform

from utils.ge_counterfactual import (
    load_trade_data, solve_counterfactual, compute_derived_cubes,
    balance_trade, build_tariff_cube, _counterfactual_equations,
    EPS,
)


def _trade_matrix_from_ge(data: dict, X_sol: np.ndarray, elasticity: str,
                           tjik_h3D: np.ndarray) -> tuple:
    """
    Reconstruct the counterfactual bilateral trade matrix from the GE solution.

    Returns:
        W_cf: (N, N) aggregate bilateral trade matrix under counterfactual
        countries: list of ISO3 codes
    """
    N, S = data["N"], data["S"]
    sigma_S = data["sigma"][elasticity]["sigma_S"]
    sigma_k3D = np.tile(sigma_S[np.newaxis, np.newaxis, :], (N, N, 1))

    # Balance trade first
    Xjik_balanced = balance_trade(data["Xjik_3D"], sigma_k3D, data["tjik_3D"], N, S)
    lambda_jik3D, Yi3D, Ri3D, e_ik3D = compute_derived_cubes(
        Xjik_balanced, data["tjik_3D"], N, S)

    wi_h = np.abs(X_sol[:N])
    Yi_h = np.abs(X_sol[N:2*N])

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yi_h3D = np.tile(Yi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yj_h3D = Yi_h3D.transpose(1, 0, 2)
    Yj3D = Yi3D.transpose(1, 0, 2)

    tjik_3D_cf = tjik_h3D * (1 + data["tjik_3D"]) - 1

    # Updated trade shares
    AUX0 = lambda_jik3D * ((tjik_h3D * wi_h3D) ** (1 - sigma_k3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    AUX2 = AUX0 / np.maximum(AUX1, EPS)

    # Counterfactual bilateral flows by sector
    Xjik_cf = AUX2 * e_ik3D * (Yj_h3D * Yj3D)

    # Aggregate across sectors
    W_cf = Xjik_cf.sum(axis=2)  # (N, N)

    return W_cf, data["countries"]


def _trade_to_distance(W: np.ndarray) -> np.ndarray:
    """Convert trade matrix to distance matrix via negative log transformation."""
    N = W.shape[0]
    W_sym = np.maximum(W, W.T)
    np.fill_diagonal(W_sym, 0)

    max_w = W_sym[W_sym > 0].max() if np.any(W_sym > 0) else 1
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            w = W_sym[i, j]
            if w > 0:
                D[i, j] = D[j, i] = -np.log(w / max_w + 1e-10)
            else:
                D[i, j] = D[j, i] = -np.log(1e-10 / max_w)

    return D


def _compute_ph(D: np.ndarray, max_dim: int = 1):
    """Compute persistent homology from a distance matrix."""
    import ripser
    result = ripser.ripser(D, maxdim=max_dim, distance_matrix=True)
    return result["dgms"]


@st.cache_data(ttl=3600)
def compare_topology_factual_vs_counterfactual(
    _data_hash: str,
    dataset: str,
    year: int,
    elasticity: str,
    scenario: dict,
    retaliation: str = "none",
    top_n: int = 50,
) -> dict:
    """
    Compare topological features of the factual vs counterfactual trade network.

    Returns:
        factual_betti:    dict of Betti numbers for factual network
        cf_betti:         dict of Betti numbers for counterfactual network
        factual_diagrams: persistence diagrams for factual
        cf_diagrams:      persistence diagrams for counterfactual
        factual_W:        factual aggregate trade matrix
        cf_W:             counterfactual aggregate trade matrix
        welfare:          welfare changes
        countries:        country list (top_n)
        persistence_change: dict summarizing topological changes
    """
    data = load_trade_data(dataset, year)
    N_full = data["N"]

    # Solve counterfactual
    result = solve_counterfactual(data, elasticity, scenario, retaliation)

    # Reconstruct counterfactual trade matrix
    W_cf, countries_full = _trade_matrix_from_ge(
        data, result["X_sol"], elasticity, result["tjik_h3D"])

    # Factual trade matrix (aggregate across sectors)
    sigma_S = data["sigma"][elasticity]["sigma_S"]
    sigma_k3D = np.tile(sigma_S[np.newaxis, np.newaxis, :], (N_full, N_full, 1))
    Xjik_balanced = balance_trade(data["Xjik_3D"], sigma_k3D, data["tjik_3D"], N_full, data["S"])
    W_factual = Xjik_balanced.sum(axis=2)  # (N, N)

    # Select top_n countries by total trade
    total_trade = W_factual.sum(axis=0) + W_factual.sum(axis=1)
    top_idx = np.argsort(total_trade)[-top_n:][::-1]

    countries = [countries_full[i] for i in top_idx]
    W_fact_sub = W_factual[np.ix_(top_idx, top_idx)]
    W_cf_sub = W_cf[np.ix_(top_idx, top_idx)]
    welfare_sub = result["welfare"][top_idx]

    # Distance matrices
    D_fact = _trade_to_distance(W_fact_sub)
    D_cf = _trade_to_distance(W_cf_sub)

    # Persistent homology
    dgms_fact = _compute_ph(D_fact, max_dim=1)
    dgms_cf = _compute_ph(D_cf, max_dim=1)

    # Betti numbers at median filtration value
    def count_betti(dgms, dim, threshold):
        if dim >= len(dgms):
            return 0
        dgm = dgms[dim]
        return int(np.sum((dgm[:, 0] <= threshold) & (dgm[:, 1] > threshold)))

    # Use median of factual H1 births as threshold
    if len(dgms_fact) > 1 and len(dgms_fact[1]) > 0:
        median_thresh = np.median(dgms_fact[1][:, 0])
    else:
        median_thresh = np.median(D_fact[D_fact > 0]) if np.any(D_fact > 0) else 1.0

    betti_fact = {
        "b0": count_betti(dgms_fact, 0, median_thresh),
        "b1": count_betti(dgms_fact, 1, median_thresh),
    }
    betti_cf = {
        "b0": count_betti(dgms_cf, 0, median_thresh),
        "b1": count_betti(dgms_cf, 1, median_thresh),
    }

    # Total persistence (sum of lifetimes)
    def total_persistence(dgms, dim):
        if dim >= len(dgms):
            return 0.0
        dgm = dgms[dim]
        finite_mask = np.isfinite(dgm[:, 1])
        if not np.any(finite_mask):
            return 0.0
        return float(np.sum(dgm[finite_mask, 1] - dgm[finite_mask, 0]))

    tp_fact_h1 = total_persistence(dgms_fact, 1)
    tp_cf_h1 = total_persistence(dgms_cf, 1)

    # Serialize diagrams for JSON/cache
    def serialize_dgm(dgms):
        out = {}
        for dim, dgm in enumerate(dgms):
            out[str(dim)] = [
                {"birth": float(b), "death": float(d) if np.isfinite(d) else None}
                for b, d in dgm
            ]
        return out

    return {
        "factual_betti": betti_fact,
        "cf_betti": betti_cf,
        "factual_diagrams": serialize_dgm(dgms_fact),
        "cf_diagrams": serialize_dgm(dgms_cf),
        "factual_total_persistence_h1": tp_fact_h1,
        "cf_total_persistence_h1": tp_cf_h1,
        "welfare": welfare_sub.tolist(),
        "countries": countries,
        "n_countries": len(countries),
        "scenario": scenario,
        "retaliation": retaliation,
        "threshold": float(median_thresh),
        "persistence_change": {
            "delta_b0": betti_cf["b0"] - betti_fact["b0"],
            "delta_b1": betti_cf["b1"] - betti_fact["b1"],
            "delta_total_persistence_h1": tp_cf_h1 - tp_fact_h1,
            "pct_change_persistence": (
                100 * (tp_cf_h1 - tp_fact_h1) / tp_fact_h1
                if tp_fact_h1 > 0 else 0.0
            ),
        },
    }


@st.cache_data(ttl=3600)
def topological_laffer_curve(
    _data_hash: str,
    dataset: str,
    year: int,
    elasticity: str,
    country: str = "USA",
    rates: list = None,
    top_n: int = 40,
) -> dict:
    """
    Compute Betti numbers and total persistence as a function of tariff rate.

    This is the topological Laffer curve — showing how network complexity
    responds to tariff policy. Novel result.

    Returns dict with:
        rates: list of tariff rates
        b0: list of β₀ at each rate
        b1: list of β₁ at each rate
        total_persistence_h1: list of total H₁ persistence
        welfare_imposer: list of imposer welfare at each rate
        welfare_world: list of world avg welfare at each rate
    """
    if rates is None:
        rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]

    data = load_trade_data(dataset, year)
    N_full = data["N"]
    sigma_S = data["sigma"][elasticity]["sigma_S"]
    sigma_k3D = np.tile(sigma_S[np.newaxis, np.newaxis, :], (N_full, N_full, 1))
    Xjik_balanced = balance_trade(data["Xjik_3D"], sigma_k3D, data["tjik_3D"], N_full, data["S"])
    W_factual = Xjik_balanced.sum(axis=2)
    total_trade = W_factual.sum(axis=0) + W_factual.sum(axis=1)
    top_idx = np.argsort(total_trade)[-top_n:][::-1]
    countries_full = data["countries"]
    imposer_full_idx = countries_full.index(country) if country in countries_full else 0

    results = {"rates": [], "b0": [], "b1": [], "total_persistence_h1": [],
               "welfare_imposer": [], "welfare_world": []}

    for rate in rates:
        if rate == 0.0:
            # Factual network
            W_sub = W_factual[np.ix_(top_idx, top_idx)]
            D = _trade_to_distance(W_sub)
            dgms = _compute_ph(D, max_dim=1)
            welfare_imp = 0.0
            welfare_avg = 0.0
        else:
            scenario = {"type": "uniform", "country": country, "rate": rate}
            try:
                r = solve_counterfactual(data, elasticity, scenario, "none")
                W_cf, _ = _trade_matrix_from_ge(data, r["X_sol"], elasticity, r["tjik_h3D"])
                W_sub = W_cf[np.ix_(top_idx, top_idx)]
                D = _trade_to_distance(W_sub)
                dgms = _compute_ph(D, max_dim=1)
                welfare_imp = float(r["welfare"][imposer_full_idx])
                welfare_avg = float(np.mean(r["welfare"]))
            except Exception:
                continue

        # Betti at median threshold
        if len(dgms) > 1 and len(dgms[1]) > 0:
            thresh = np.median(dgms[1][:, 0]) if len(dgms[1]) > 0 else 1.0
        else:
            thresh = np.median(D[D > 0]) if np.any(D > 0) else 1.0

        b0 = int(np.sum((dgms[0][:, 0] <= thresh) & (dgms[0][:, 1] > thresh))) if len(dgms) > 0 else 0
        b1 = int(np.sum((dgms[1][:, 0] <= thresh) & (dgms[1][:, 1] > thresh))) if len(dgms) > 1 else 0

        tp = 0.0
        if len(dgms) > 1:
            finite = np.isfinite(dgms[1][:, 1])
            if np.any(finite):
                tp = float(np.sum(dgms[1][finite, 1] - dgms[1][finite, 0]))

        results["rates"].append(rate)
        results["b0"].append(b0)
        results["b1"].append(b1)
        results["total_persistence_h1"].append(tp)
        results["welfare_imposer"].append(welfare_imp)
        results["welfare_world"].append(welfare_avg)

    return results
