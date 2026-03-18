"""
General Equilibrium Counterfactual Engine.

Python port + extension of Lashkaripour (2021, JIE) sufficient-statistics
approach for multi-country, multi-sector trade models.

Implements:
- Counterfactual equilibrium solver (2N system: exogenous tariff scenarios)
- Nash equilibrium tariff solver (3N system: all countries optimize)
- Optimal unilateral tariff solver (2N+1: single country optimizes)
- CES welfare computation (more general than ACR)
- Balanced trade preprocessing
- 8 trade elasticity specifications from the literature
- Tariff scenario construction with retaliation

Data: reads pre-built .mat files from Lashkaripour's tariffwar package
(WIOD 44×16, ICIO 81×28, ITPD 135×154).

References:
  Lashkaripour (2021) "The Cost of a Global Tariff War" JIE
  Caliendo & Parro (2015) "Trade and Welfare Effects of NAFTA" ReStud
  Arkolakis, Costinot, Rodriguez-Clare (2012) "New Trade Models" AER
"""

import numpy as np
from scipy.optimize import root
import h5py
import os
import streamlit as st

# ============================================================
#  Constants
# ============================================================

EPS = np.finfo(float).eps

# Path to .mat files: prefer local repo, fall back to bundled data/mat/
_LOCAL_MAT = os.path.expanduser("~/lashkaripour_repos/tariffwar/mat")
_BUNDLED_MAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mat")
MAT_DIR = _LOCAL_MAT if os.path.isdir(_LOCAL_MAT) else _BUNDLED_MAT

# EU-27 ISO3 codes for targeted scenarios
EU27 = [
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX",
    "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
]


# ============================================================
#  Trade Elasticity Registry (8 specifications)
# ============================================================

ELASTICITY_REGISTRY = {
    "IS": {
        "name": "In-Sample",
        "paper": "Lashkaripour (2021, JIE) / Caliendo-Parro estimator",
        "note": "Dataset-specific; read from .mat file",
    },
    "U4": {
        "name": "Uniform ε=4",
        "paper": "Simonovska & Waugh (2014, JIE)",
    },
    "CP": {
        "name": "Caliendo-Parro",
        "paper": "Caliendo & Parro (2015, ReStud)",
    },
    "BSY": {
        "name": "Bagwell-Staiger-Yurukoglu",
        "paper": "BSY (2021, Econometrica)",
    },
    "GYY": {
        "name": "Giri-Yi-Yilmazkuday",
        "paper": "GYY (2021, JIE)",
    },
    "Shap": {
        "name": "Shapiro",
        "paper": "Shapiro (2016, AEJ)",
    },
    "FGO": {
        "name": "Fontagné-Guimbard-Orefice",
        "paper": "Fontagné et al. (2022, JIE)",
    },
    "LL": {
        "name": "Lashkaripour-Lugovskyy",
        "paper": "Lashkaripour & Lugovskyy (2023, AER)",
    },
}


# ============================================================
#  Data Loading
# ============================================================

def _read_h5_string_array(f, dataset):
    """Read an array of object references (strings) from HDF5."""
    result = []
    for ref in dataset:
        obj = f[ref]
        chars = np.array(obj).flatten()
        result.append("".join(chr(int(c)) for c in chars))
    return result


@st.cache_data(ttl=7200)
def load_trade_data(dataset: str, year: int, mat_dir: str = MAT_DIR) -> dict:
    """
    Load pre-built trade data from a .mat file.

    Returns dict with:
        Xjik_3D:    (N, N, S) bilateral trade flows
        tjik_3D:    (N, N, S) applied tariff rates (decimal)
        sigma:      dict mapping elasticity abbrev -> sigma_S (S,) array
        countries:  list of N ISO3 codes
        sectors:    list of S sector names
        N, S:       int
        dataset:    str
        year:       int
    """
    fname = os.path.join(mat_dir, f"{dataset.upper()}{year}.mat")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Data file not found: {fname}")

    with h5py.File(fname, "r") as f:
        data = f["data"]

        N = int(data["N"][0, 0])
        S = int(data["S"][0, 0])

        # Trade flows: stored as (S, N, N) in HDF5, we want (N, N, S)
        Xjik_3D = np.array(data["Xjik_3D"]).transpose(1, 2, 0)  # (N, N, S)

        # Tariffs: same transpose
        tjik_3D = np.array(data["tjik_3D"]).transpose(1, 2, 0)  # (N, N, S)

        # Countries
        countries = _read_h5_string_array(f, data["countries"][0])

        # Sectors
        sectors = _read_h5_string_array(f, data["sectors"][0])

        # Elasticities: all 8 sources
        sigma_dict = {}
        sigma_grp = data["sigma"]
        for key in sigma_grp.keys():
            sigma_S = np.array(sigma_grp[key]["sigma_S"]).flatten()
            epsilon_S = np.array(sigma_grp[key]["epsilon_S"]).flatten()
            sigma_dict[key] = {
                "sigma_S": sigma_S,
                "epsilon_S": epsilon_S,
            }

    # For ICIO: add identity to diagonal (domestic sales)
    if dataset.lower() == "icio":
        eye_NS = np.tile(np.eye(N)[:, :, np.newaxis], (1, 1, S))
        Xjik_3D = Xjik_3D + eye_NS

    return {
        "Xjik_3D": Xjik_3D,
        "tjik_3D": tjik_3D,
        "sigma": sigma_dict,
        "countries": countries,
        "sectors": sectors,
        "N": N,
        "S": S,
        "dataset": dataset,
        "year": year,
    }


def get_available_datasets() -> list:
    """List available dataset-year combinations from the mat directory."""
    available = []
    if not os.path.isdir(MAT_DIR):
        return available
    for f in sorted(os.listdir(MAT_DIR)):
        if f.endswith(".mat"):
            name = f[:-4]
            # Parse e.g. "ICIO2022" -> ("icio", 2022)
            for prefix in ["ICIO", "WIOD", "ITPD"]:
                if name.startswith(prefix):
                    try:
                        year = int(name[len(prefix):])
                        available.append((prefix.lower(), year))
                    except ValueError:
                        pass
    return available


# ============================================================
#  Derived Cubes
# ============================================================

def compute_derived_cubes(Xjik_3D: np.ndarray, tjik_3D: np.ndarray, N: int, S: int):
    """
    Compute trade shares, income, revenue, and expenditure cubes.

    Returns:
        lambda_jik3D: (N, N, S) trade shares (sum over dim 0 = 1)
        Yi3D:         (N, N, S) national income (replicated)
        Ri3D:         (N, N, S) revenue / wage bill (replicated)
        e_ik3D:       (N, N, S) expenditure shares (Cobb-Douglas weights)
    """
    # Trade shares: lambda_jik = X_jik / sum_j(X_jik)
    denom = np.maximum(Xjik_3D.sum(axis=0, keepdims=True), EPS)  # (1, N, S)
    lambda_jik3D = Xjik_3D / np.tile(denom, (N, 1, 1))

    # National income: Yi = sum_j sum_s X_jik  (total spending by importer i)
    Yi = np.maximum(Xjik_3D.sum(axis=0).sum(axis=1), EPS)  # (N,)
    Yi3D = np.tile(Yi[:, np.newaxis, np.newaxis], (1, N, S))  # (N, N, S)

    # Revenue / wage bill: Ri = sum_j sum_s X_jik / (1 + t_jik)
    Ri = np.maximum((Xjik_3D / (1 + tjik_3D)).sum(axis=1).sum(axis=1), EPS)  # (N,)
    Ri3D = np.tile(Ri[:, np.newaxis, np.newaxis], (1, N, S))

    # Expenditure shares: e_ik = sum_j(X_jik) / Yi
    sector_spending = Xjik_3D.sum(axis=0)  # (N, S)
    e_ik = sector_spending / np.maximum(Yi[:, np.newaxis], EPS)  # (N, S)
    e_ik3D = np.tile(e_ik[np.newaxis, :, :], (N, 1, 1))  # (N, N, S)

    return lambda_jik3D, Yi3D, Ri3D, e_ik3D


# ============================================================
#  Balanced Trade (removes deficits)
# ============================================================

def _balanced_trade_equations(X, N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D, lambda_jik3D, tjik_3D):
    """2N system for balanced trade (no tariff change, just remove deficits)."""
    wi_h = np.abs(X[:N])
    Yi_h = np.abs(X[N:2*N])

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yi_h3D = np.tile(Yi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yj_h3D = Yi_h3D.transpose(1, 0, 2)
    Yj3D = Yi3D.transpose(1, 0, 2)

    # No tariff change in balanced trade step
    AUX0 = lambda_jik3D * (wi_h3D ** (1 - sigma_k3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    AUX2 = AUX0 / np.maximum(AUX1, EPS)
    AUX3 = AUX2 * e_ik3D * (Yj_h3D * Yj3D) / (1 + tjik_3D)

    ERR1 = AUX3.sum(axis=2).sum(axis=1) - wi_h * Ri3D[:, 0, 0]
    ERR1[-1] = (Ri3D[:, 0, 0] * (wi_h - 1)).sum()  # wage anchor

    AUX5 = AUX2 * e_ik3D * (tjik_3D / (1 + tjik_3D)) * Yj_h3D * Yj3D
    ERR2 = AUX5.sum(axis=2).sum(axis=0) + (wi_h * Ri3D[:, 0, 0]) - Yi_h * Yi3D[:, 0, 0]

    return np.concatenate([ERR1, ERR2])


def balance_trade(Xjik_3D, sigma_k3D, tjik_3D, N, S):
    """
    Solve for trade-balanced flows (zero deficits).

    Returns rebalanced Xjik_3D_new (N, N, S).
    """
    lambda_jik3D, Yi3D, Ri3D, e_ik3D = compute_derived_cubes(Xjik_3D, tjik_3D, N, S)

    X0 = np.ones(2 * N)
    result = root(
        _balanced_trade_equations, X0,
        args=(N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D, lambda_jik3D, tjik_3D),
        method="hybr",
        options={"maxfev": 10000, "xtol": 1e-8},
    )

    if not result.success:
        # Retry with Levenberg-Marquardt equivalent
        result = root(
            _balanced_trade_equations, X0 * (0.9 + 0.2 * np.random.rand(2 * N)),
            args=(N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D, lambda_jik3D, tjik_3D),
            method="lm",
            options={"maxiter": 200, "xtol": 1e-8},
        )

    wi_h = np.abs(result.x[:N])
    Yi_h = np.abs(result.x[N:2*N])

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yi_h3D = np.tile(Yi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yj_h3D = Yi_h3D.transpose(1, 0, 2)
    Yj3D = Yi3D.transpose(1, 0, 2)

    AUX0 = lambda_jik3D * (wi_h3D ** (1 - sigma_k3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    AUX2 = AUX0 / np.maximum(AUX1, EPS)
    Xjik_new = AUX2 * e_ik3D * (Yj_h3D * Yj3D)

    return Xjik_new


# ============================================================
#  Tariff Scenario Construction
# ============================================================

def build_tariff_cube(
    scenario: dict,
    countries: list,
    tjik_3D_factual: np.ndarray,
    N: int, S: int,
    retaliation: str = "none",
) -> tuple:
    """
    Build counterfactual tariff cube for a given scenario.

    scenario dict keys:
        type: "uniform" | "targeted" | "optimal" | "nash"
        rate: float (tariff rate, e.g. 0.10 for 10%)
        country: str ISO3 (who imposes, default "USA")
        partner: str ISO3 or "EU" (for targeted)

    Returns:
        tjik_3D_cf: (N, N, S) counterfactual tariff levels
        tjik_h3D:   (N, N, S) tariff hat = (1+cf)/(1+factual)
    """
    tjik_3D_cf = tjik_3D_factual.copy()

    imposer = scenario.get("country", "USA")
    imposer_idx = _find_country(countries, imposer)
    rate = scenario.get("rate", 0.10)
    scenario_type = scenario["type"]

    if scenario_type == "uniform":
        # Uniform tariff on all partners
        for j in range(N):
            if j == imposer_idx:
                continue
            tjik_3D_cf[j, imposer_idx, :] = rate

    elif scenario_type == "targeted":
        partner = scenario.get("partner", "CHN")
        partner_idxs = _resolve_partner(countries, partner)
        for j in partner_idxs:
            tjik_3D_cf[j, imposer_idx, :] = rate

    elif scenario_type == "free_trade":
        # Remove all tariffs
        tjik_3D_cf[:] = 0.0

    elif scenario_type == "custom":
        # Custom tariff schedule: scenario["rates"] = {ISO3: rate}
        rates_map = scenario.get("rates", {})
        for j, c in enumerate(countries):
            if j == imposer_idx:
                continue
            if c in rates_map:
                tjik_3D_cf[j, imposer_idx, :] = rates_map[c]

    # Reciprocal retaliation
    if retaliation == "reciprocal":
        if scenario_type == "uniform":
            for i in range(N):
                if i == imposer_idx:
                    continue
                tjik_3D_cf[imposer_idx, i, :] = rate
        elif scenario_type == "targeted":
            partner_idxs = _resolve_partner(countries, scenario.get("partner", "CHN"))
            for i in partner_idxs:
                tjik_3D_cf[imposer_idx, i, :] = rate
        elif scenario_type == "custom":
            rates_map = scenario.get("rates", {})
            for i, c in enumerate(countries):
                if i == imposer_idx:
                    continue
                if c in rates_map:
                    tjik_3D_cf[imposer_idx, i, :] = rates_map[c]

    # Tariff hat
    tjik_h3D = (1 + tjik_3D_cf) / (1 + tjik_3D_factual)

    return tjik_3D_cf, tjik_h3D


def _find_country(countries, iso3):
    for i, c in enumerate(countries):
        if c == iso3:
            return i
    raise ValueError(f"Country {iso3} not found in dataset")


def _resolve_partner(countries, partner):
    if partner == "EU":
        idxs = [i for i, c in enumerate(countries) if c in EU27]
        if not idxs:
            raise ValueError("No EU-27 members found in dataset")
        return idxs
    return [_find_country(countries, partner)]


# ============================================================
#  Counterfactual Equations (2N system)
# ============================================================

def _counterfactual_equations(X, N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D,
                               lambda_jik3D, tjik_3D_factual, tjik_h3D):
    """
    Equations 6 & 7 from Lashkaripour (2021).
    2N unknowns: N wages + N incomes, with exogenous tariff hat.
    """
    wi_h = np.abs(X[:N])
    Yi_h = np.abs(X[N:2*N])

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yi_h3D = np.tile(Yi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yj_h3D = Yi_h3D.transpose(1, 0, 2)
    Yj3D = Yi3D.transpose(1, 0, 2)

    tjik_3D_cf = tjik_h3D * (1 + tjik_3D_factual) - 1

    # Eq 6: Wage income = total sales net of tariffs
    AUX0 = lambda_jik3D * ((tjik_h3D * wi_h3D) ** (1 - sigma_k3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    AUX2 = AUX0 / np.maximum(AUX1, EPS)
    AUX3 = AUX2 * e_ik3D * (Yj_h3D * Yj3D) / (1 + tjik_3D_cf)

    ERR1 = AUX3.sum(axis=2).sum(axis=1) - wi_h * Ri3D[:, 0, 0]
    ERR1[-1] = (Ri3D[:, 0, 0] * (wi_h - 1)).sum()  # wage anchor

    # Eq 7: National income = wage income + tariff revenue
    AUX5 = AUX2 * e_ik3D * (tjik_3D_cf / (1 + tjik_3D_cf)) * Yj_h3D * Yj3D
    ERR2 = AUX5.sum(axis=2).sum(axis=0) + (wi_h * Ri3D[:, 0, 0]) - Yi_h * Yi3D[:, 0, 0]

    return np.concatenate([ERR1, ERR2])


# ============================================================
#  Nash Equations (3N system)
# ============================================================

def _nash_equations(X, N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D,
                     lambda_jik3D, tjik_3D_factual):
    """
    Equations 6, 7, 14 from Lashkaripour (2021).
    3N unknowns: N wages + N incomes + N optimal tariff levels.
    """
    wi_h = np.abs(X[:N])
    Yi_h = np.abs(X[N:2*N])
    tjik = np.abs(X[2*N:3*N])  # optimal tariff levels per country

    # Build tariff cube from N×1 vector
    tjik_2D = np.tile(tjik[np.newaxis, :], (N, 1))  # (N, N)
    eye_N = np.eye(N)
    tjik_3D = np.tile((eye_N + tjik_2D * (1 - eye_N))[:, :, np.newaxis], (1, 1, S)) - 1
    tjik_h3D = (1 + tjik_3D) / (1 + tjik_3D_factual)

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yi_h3D = np.tile(Yi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yj_h3D = Yi_h3D.transpose(1, 0, 2)
    Yj3D = Yi3D.transpose(1, 0, 2)

    # Eq 6
    AUX0 = lambda_jik3D * ((tjik_h3D * wi_h3D) ** (1 - sigma_k3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    AUX2 = AUX0 / np.maximum(AUX1, EPS)
    AUX3 = AUX2 * e_ik3D * (Yj_h3D * Yj3D) / (1 + tjik_3D)

    ERR1 = AUX3.sum(axis=2).sum(axis=1) - wi_h * Ri3D[:, 0, 0]
    ERR1[-1] = (Ri3D[:, 0, 0] * (wi_h - 1)).sum()

    # Eq 7
    AUX5 = AUX2 * e_ik3D * (tjik_3D / (1 + tjik_3D)) * Yj_h3D * Yj3D
    ERR2 = AUX5.sum(axis=2).sum(axis=0) + (wi_h * Ri3D[:, 0, 0]) - Yi_h * Yi3D[:, 0, 0]

    # Eq 14: Optimal tariff FOC
    off_diag = np.tile((1 - eye_N)[:, :, np.newaxis], (1, 1, S))
    AUX6 = AUX3 * off_diag
    AUX7_num = (AUX6 * (1 - AUX2)).sum(axis=1, keepdims=True)
    AUX7_den = np.maximum(np.tile(AUX6.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True), (1, 1, S)), EPS)
    AUX7 = AUX7_num / AUX7_den
    AUX8 = ((sigma_k3D[:, 0:1, :] - 1) * AUX7).sum(axis=2).flatten()
    ERR3 = tjik - (1 + 1 / np.maximum(AUX8, 1))

    return np.concatenate([ERR1, ERR2, ERR3])


# ============================================================
#  Optimal Unilateral Tariff (2N+1 system) — generalized
# ============================================================

def _optimal_country_equations(X, N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D,
                                lambda_jik3D, tjik_3D_factual, opt_idx):
    """
    Equations 6, 7, 14 — but Eq 14 applies ONLY to country opt_idx.
    2N+1 unknowns: N wages + N incomes + 1 optimal tariff rate.
    """
    wi_h = np.abs(X[:N])
    Yi_h = np.abs(X[N:2*N])
    t_opt = np.abs(X[2*N])  # scalar optimal tariff rate

    # Build tariff cube: opt_idx imports at t_opt, everything else factual
    tjik_3D = tjik_3D_factual.copy()
    for j in range(N):
        if j != opt_idx:
            tjik_3D[j, opt_idx, :] = t_opt
    tjik_h3D = (1 + tjik_3D) / (1 + tjik_3D_factual)

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yi_h3D = np.tile(Yi_h[:, np.newaxis, np.newaxis], (1, N, S))
    Yj_h3D = Yi_h3D.transpose(1, 0, 2)
    Yj3D = Yi3D.transpose(1, 0, 2)

    # Eq 6
    AUX0 = lambda_jik3D * ((tjik_h3D * wi_h3D) ** (1 - sigma_k3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    AUX2 = AUX0 / np.maximum(AUX1, EPS)
    AUX3 = AUX2 * e_ik3D * (Yj_h3D * Yj3D) / (1 + tjik_3D)

    ERR1 = AUX3.sum(axis=2).sum(axis=1) - wi_h * Ri3D[:, 0, 0]
    ERR1[-1] = (Ri3D[:, 0, 0] * (wi_h - 1)).sum()

    # Eq 7
    AUX5 = AUX2 * e_ik3D * (tjik_3D / (1 + tjik_3D)) * Yj_h3D * Yj3D
    ERR2 = AUX5.sum(axis=2).sum(axis=0) + (wi_h * Ri3D[:, 0, 0]) - Yi_h * Yi3D[:, 0, 0]

    # Eq 14 for opt_idx only
    off_diag = np.tile((1 - np.eye(N))[:, :, np.newaxis], (1, 1, S))
    AUX6 = AUX3 * off_diag
    AUX7_num = (AUX6 * (1 - AUX2)).sum(axis=1, keepdims=True)
    AUX7_den = np.maximum(np.tile(AUX6.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True), (1, 1, S)), EPS)
    AUX7 = AUX7_num / AUX7_den
    AUX8 = ((sigma_k3D[:, 0:1, :] - 1) * AUX7).sum(axis=2).flatten()
    ERR3 = np.array([t_opt - 1.0 / np.maximum(AUX8[opt_idx], 1)])

    return np.concatenate([ERR1, ERR2, ERR3])


# ============================================================
#  Welfare Computation
# ============================================================

def compute_welfare(X, N, S, e_ik3D, sigma_k3D, lambda_jik3D, tjik_h3D):
    """
    Compute welfare gains (% change in real income) per country.

    W_hat = E_hat / P_hat where P_hat = CES price index aggregation.
    More general than ACR (handles multi-sector + tariff revenue).
    """
    wi_h = np.abs(X[:N])
    Ei_h = np.abs(X[N:2*N])

    wi_h3D = np.tile(wi_h[:, np.newaxis, np.newaxis], (1, N, S))

    # CES price index change
    AUX0 = (tjik_h3D * wi_h3D) ** (1 - sigma_k3D)
    price_sum = np.maximum((lambda_jik3D * AUX0).sum(axis=0), EPS)  # (N, S)
    sigma_1d = sigma_k3D[0, 0, :]  # (S,) — same across countries
    Pjk_h = price_sum ** (1 / (1 - sigma_1d[np.newaxis, :]))
    Pjk_h = np.where(np.isfinite(Pjk_h), Pjk_h, 1.0)

    # Cobb-Douglas across sectors
    e_ik = e_ik3D[0, :, :]  # (N, S)
    Pi_h = np.exp((e_ik * np.log(np.maximum(Pjk_h, EPS))).sum(axis=1))  # (N,)

    Wi_h = Ei_h / Pi_h
    return 100 * (Wi_h - 1)


# ============================================================
#  Solver Wrappers
# ============================================================

def solve_counterfactual(data: dict, elasticity: str, scenario: dict,
                          retaliation: str = "none",
                          max_retries: int = 3) -> dict:
    """
    Full pipeline: load data -> balance trade -> build tariff cube -> solve -> welfare.

    Returns dict with:
        welfare:     (N,) percent welfare change per country
        countries:   list of ISO3
        sectors:     list of sector names
        X_sol:       solution vector
        converged:   bool
        max_residual: float
        scenario:    input scenario
        retaliation: str
        elasticity:  str
        N, S:        int
        tjik_h3D:    tariff hat
    """
    N, S = data["N"], data["S"]
    countries = data["countries"]

    # Get sigma
    if elasticity not in data["sigma"]:
        raise ValueError(f"Elasticity '{elasticity}' not available. Options: {list(data['sigma'].keys())}")
    sigma_S = data["sigma"][elasticity]["sigma_S"]
    sigma_k3D = np.tile(sigma_S[np.newaxis, np.newaxis, :], (N, N, 1))

    # Balance trade
    Xjik_balanced = balance_trade(data["Xjik_3D"], sigma_k3D, data["tjik_3D"], N, S)

    # Derived cubes
    lambda_jik3D, Yi3D, Ri3D, e_ik3D = compute_derived_cubes(
        Xjik_balanced, data["tjik_3D"], N, S)

    # Build tariff cube
    tjik_3D_cf, tjik_h3D = build_tariff_cube(
        scenario, countries, data["tjik_3D"], N, S, retaliation)

    # Solve
    X0 = np.concatenate([0.95 * np.ones(N), 1.05 * np.ones(N)])
    best_x, best_ef, best_resid = X0, False, np.inf

    for attempt in range(max_retries + 1):
        if attempt > 0:
            X0 = np.concatenate([
                (0.8 + 0.4 * np.random.rand()) * np.ones(N),
                (0.8 + 0.4 * np.random.rand()) * np.ones(N),
            ])

        result = root(
            _counterfactual_equations, X0,
            args=(N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D,
                  lambda_jik3D, data["tjik_3D"], tjik_h3D),
            method="lm" if attempt > 0 else "hybr",
            options={"maxfev": 10000, "xtol": 1e-8},
        )

        resid = np.max(np.abs(result.fun))
        if result.success or resid < best_resid:
            best_x = result.x
            best_ef = result.success
            best_resid = resid

        if result.success:
            break

    # Welfare
    welfare = compute_welfare(best_x, N, S, e_ik3D, sigma_k3D, lambda_jik3D, tjik_h3D)

    return {
        "welfare": welfare,
        "countries": countries,
        "sectors": data["sectors"],
        "X_sol": best_x,
        "converged": best_ef,
        "max_residual": best_resid,
        "scenario": scenario,
        "retaliation": retaliation,
        "elasticity": elasticity,
        "N": N,
        "S": S,
        "tjik_h3D": tjik_h3D,
        "lambda_jik3D": lambda_jik3D,
        "e_ik3D": e_ik3D,
        "sigma_k3D": sigma_k3D,
    }


def solve_nash_equilibrium(data: dict, elasticity: str,
                            max_retries: int = 5) -> dict:
    """
    Solve for Nash equilibrium tariffs (all countries optimize simultaneously).
    3N system: N wages + N incomes + N tariff levels.
    """
    N, S = data["N"], data["S"]
    countries = data["countries"]

    sigma_S = data["sigma"][elasticity]["sigma_S"]
    sigma_k3D = np.tile(sigma_S[np.newaxis, np.newaxis, :], (N, N, 1))

    Xjik_balanced = balance_trade(data["Xjik_3D"], sigma_k3D, data["tjik_3D"], N, S)
    lambda_jik3D, Yi3D, Ri3D, e_ik3D = compute_derived_cubes(
        Xjik_balanced, data["tjik_3D"], N, S)

    X0 = np.concatenate([0.9 * np.ones(N), 1.1 * np.ones(N), 1.25 * np.ones(N)])
    best_x, best_ef, best_resid = X0, False, np.inf

    for attempt in range(max_retries + 1):
        if attempt > 0:
            X0 = np.concatenate([
                (0.7 + 0.6 * np.random.rand()) * np.ones(N),
                (0.8 + 0.6 * np.random.rand()) * np.ones(N),
                (1.0 + 0.5 * np.random.rand()) * np.ones(N),
            ])

        result = root(
            _nash_equations, X0,
            args=(N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D,
                  lambda_jik3D, data["tjik_3D"]),
            method="lm" if attempt > 0 else "hybr",
            options={"maxfev": 50000, "xtol": 1e-8},
        )

        resid = np.max(np.abs(result.fun))
        if result.success or resid < best_resid:
            best_x = result.x
            best_ef = result.success
            best_resid = resid

        if result.success:
            break

    wi_h = np.abs(best_x[:N])
    Yi_h = np.abs(best_x[N:2*N])
    tjik_opt = np.abs(best_x[2*N:3*N])

    # Reconstruct tariff cube for welfare
    tjik_2D = np.tile(tjik_opt[np.newaxis, :], (N, 1))
    eye_N = np.eye(N)
    tjik_3D_nash = np.tile((eye_N + tjik_2D * (1 - eye_N))[:, :, np.newaxis], (1, 1, S)) - 1
    tjik_h3D = (1 + tjik_3D_nash) / (1 + data["tjik_3D"])

    welfare = compute_welfare(best_x[:2*N], N, S, e_ik3D, sigma_k3D, lambda_jik3D, tjik_h3D)

    return {
        "welfare": welfare,
        "optimal_tariffs": 100 * (tjik_opt - 1),  # percent tariff rates
        "countries": countries,
        "sectors": data["sectors"],
        "converged": best_ef,
        "max_residual": best_resid,
        "elasticity": elasticity,
        "N": N,
        "S": S,
        "wages": wi_h,
        "incomes": Yi_h,
    }


def solve_optimal_tariff(data: dict, elasticity: str, country: str = "USA",
                          max_retries: int = 3) -> dict:
    """
    Solve for optimal unilateral tariff for any country.
    2N+1 system: N wages + N incomes + 1 optimal tariff rate.

    This generalizes Lashkaripour's US-only solver to ANY country.
    """
    N, S = data["N"], data["S"]
    countries = data["countries"]
    opt_idx = _find_country(countries, country)

    sigma_S = data["sigma"][elasticity]["sigma_S"]
    sigma_k3D = np.tile(sigma_S[np.newaxis, np.newaxis, :], (N, N, 1))

    Xjik_balanced = balance_trade(data["Xjik_3D"], sigma_k3D, data["tjik_3D"], N, S)
    lambda_jik3D, Yi3D, Ri3D, e_ik3D = compute_derived_cubes(
        Xjik_balanced, data["tjik_3D"], N, S)

    X0 = np.concatenate([0.95 * np.ones(N), 1.05 * np.ones(N), [0.25]])
    best_x, best_ef, best_resid = X0, False, np.inf

    for attempt in range(max_retries + 1):
        if attempt > 0:
            X0 = np.concatenate([
                (0.8 + 0.4 * np.random.rand()) * np.ones(N),
                (0.8 + 0.4 * np.random.rand()) * np.ones(N),
                [0.10 + 0.40 * np.random.rand()],
            ])

        result = root(
            _optimal_country_equations, X0,
            args=(N, S, Yi3D, Ri3D, e_ik3D, sigma_k3D,
                  lambda_jik3D, data["tjik_3D"], opt_idx),
            method="lm" if attempt > 0 else "hybr",
            options={"maxfev": 20000, "xtol": 1e-8},
        )

        resid = np.max(np.abs(result.fun))
        if result.success or resid < best_resid:
            best_x = result.x
            best_ef = result.success
            best_resid = resid

        if result.success:
            break

    t_opt = np.abs(best_x[2*N])

    # Build tariff cube and compute welfare
    tjik_3D = data["tjik_3D"].copy()
    for j in range(N):
        if j != opt_idx:
            tjik_3D[j, opt_idx, :] = t_opt
    tjik_h3D = (1 + tjik_3D) / (1 + data["tjik_3D"])

    welfare = compute_welfare(best_x[:2*N], N, S, e_ik3D, sigma_k3D, lambda_jik3D, tjik_h3D)

    return {
        "welfare": welfare,
        "optimal_tariff_pct": 100 * t_opt,
        "country": country,
        "countries": countries,
        "converged": best_ef,
        "max_residual": best_resid,
        "elasticity": elasticity,
        "N": N,
        "S": S,
    }
