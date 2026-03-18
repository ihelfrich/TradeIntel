"""
Automated Research Pipeline for Trade Network Analysis.

Runs batch analyses across countries, elasticities, tariff rates, and
retaliation regimes, caching results for the Streamlit dashboard.

All heavy solvers are imported from utils.ge_counterfactual.  Functions
accept an optional ``progress_callback(current, total)`` for UI progress
bars.  Solver failures are handled gracefully: non-converged results are
included with ``Converged=False`` so downstream code can flag them.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from utils.ge_counterfactual import (
    ELASTICITY_REGISTRY,
    load_trade_data,
    solve_counterfactual,
    solve_optimal_tariff,
    solve_nash_equilibrium,
)


# ------------------------------------------------------------------
#  Internal helpers
# ------------------------------------------------------------------

def _data_fingerprint(data: dict) -> str:
    """Return a short hash that uniquely identifies a loaded dataset."""
    key_str = f"{data['dataset']}_{data['year']}_{data['N']}_{data['S']}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _safe_mean(arr: np.ndarray) -> float:
    """Mean that tolerates NaN / Inf from failed solves."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite))


# ------------------------------------------------------------------
#  1. Optimal-tariff survey across all countries x elasticities
# ------------------------------------------------------------------

@st.cache_data(ttl=7200, show_spinner=False)
def run_optimal_tariff_survey(
    _data: dict,
    elasticities: list[str] | None = None,
    *,
    _data_hash: str | None = None,
    _progress_callback=None,
) -> pd.DataFrame:
    """Compute each country's optimal unilateral tariff under every
    elasticity specification.

    Parameters
    ----------
    _data : dict
        Output of ``load_trade_data``.  Prefixed with underscore so
        Streamlit does not try to hash the large numpy arrays.
    elasticities : list[str] | None
        Subset of ELASTICITY_REGISTRY keys.  ``None`` means all eight.
    _data_hash : str | None
        Optional pre-computed fingerprint for caching.  If *None* one is
        derived automatically.
    progress_callback : callable | None
        ``callback(current_step, total_steps)`` for progress bars.

    Returns
    -------
    pd.DataFrame
        Columns: Country, Elasticity, Optimal_Tariff_Pct,
        Own_Welfare_Pct, World_Avg_Welfare_Pct, Converged.
    """
    data = _data
    available = list(data["sigma"].keys())
    if elasticities is None:
        elasticities = [e for e in ELASTICITY_REGISTRY if e in available]
    else:
        elasticities = [e for e in elasticities if e in available]

    countries = data["countries"]
    total = len(countries) * len(elasticities)
    step = 0

    rows: list[dict] = []
    for eidx, elas in enumerate(elasticities):
        for cidx, country in enumerate(countries):
            step += 1
            if _progress_callback is not None:
                _progress_callback(step, total)

            try:
                res = solve_optimal_tariff(data, elas, country=country)
                converged = bool(res["converged"])
                opt_tariff = float(res["optimal_tariff_pct"])
                own_welfare = float(res["welfare"][cidx])
                world_avg = _safe_mean(res["welfare"])
            except Exception:
                converged = False
                opt_tariff = np.nan
                own_welfare = np.nan
                world_avg = np.nan

            rows.append({
                "Country": country,
                "Elasticity": elas,
                "Optimal_Tariff_Pct": opt_tariff,
                "Own_Welfare_Pct": own_welfare,
                "World_Avg_Welfare_Pct": world_avg,
                "Converged": converged,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
#  2. Elasticity sensitivity for a single scenario
# ------------------------------------------------------------------

@st.cache_data(ttl=7200, show_spinner=False)
def run_elasticity_sensitivity(
    _data: dict,
    scenario: dict,
    elasticities: list[str] | None = None,
    retaliation: str = "none",
    *,
    _data_hash: str | None = None,
    _progress_callback=None,
) -> pd.DataFrame:
    """Run the *same* tariff scenario under every elasticity spec.

    Parameters
    ----------
    _data : dict
        Output of ``load_trade_data``.
    scenario : dict
        Tariff scenario (passed to ``solve_counterfactual``).
    elasticities : list[str] | None
        Subset of ELASTICITY_REGISTRY keys.  ``None`` means all eight.
    retaliation : str
        ``"none"`` or ``"reciprocal"``.
    progress_callback : callable | None
        ``callback(current_step, total_steps)``.

    Returns
    -------
    pd.DataFrame
        One row per (Elasticity, Country) with welfare and metadata.
    """
    data = _data
    available = list(data["sigma"].keys())
    if elasticities is None:
        elasticities = [e for e in ELASTICITY_REGISTRY if e in available]
    else:
        elasticities = [e for e in elasticities if e in available]

    countries = data["countries"]
    total = len(elasticities)
    rows: list[dict] = []

    for step, elas in enumerate(elasticities, 1):
        if _progress_callback is not None:
            _progress_callback(step, total)

        try:
            res = solve_counterfactual(data, elas, scenario, retaliation=retaliation)
            converged = bool(res["converged"])
            welfare = res["welfare"]
            max_residual = float(res["max_residual"])
        except Exception:
            converged = False
            welfare = np.full(len(countries), np.nan)
            max_residual = np.nan

        elas_name = ELASTICITY_REGISTRY.get(elas, {}).get("name", elas)
        world_avg = _safe_mean(welfare)

        for cidx, country in enumerate(countries):
            rows.append({
                "Elasticity": elas,
                "Elasticity_Name": elas_name,
                "Country": country,
                "Welfare_Pct": float(welfare[cidx]),
                "World_Avg_Welfare_Pct": world_avg,
                "Max_Residual": max_residual,
                "Converged": converged,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
#  3. Tariff-rate sweep (Laffer-style curve)
# ------------------------------------------------------------------

@st.cache_data(ttl=7200, show_spinner=False)
def run_tariff_rate_sweep(
    _data: dict,
    elasticity: str,
    country: str,
    rates: list[float] | None = None,
    retaliation: str = "none",
    *,
    _data_hash: str | None = None,
    _progress_callback=None,
) -> pd.DataFrame:
    """Sweep tariff rates for *country* and record welfare outcomes.

    Parameters
    ----------
    _data : dict
        Output of ``load_trade_data``.
    elasticity : str
        Key into ELASTICITY_REGISTRY.
    country : str
        ISO-3 code of the imposing country.
    rates : list[float] | None
        Tariff rates to sweep (as fractions, e.g. ``[0.0, 0.05, ...]``).
        Default: 0 to 1.0 in steps of 0.05 (0 %-100 %).
    retaliation : str
        ``"none"`` or ``"reciprocal"``.
    progress_callback : callable | None
        ``callback(current_step, total_steps)``.

    Returns
    -------
    pd.DataFrame
        Columns: Tariff_Rate_Pct, Imposer_Welfare_Pct,
        World_Avg_Welfare_Pct, Converged.
    """
    data = _data
    if rates is None:
        rates = [round(r * 0.05, 4) for r in range(21)]  # 0 .. 1.0

    countries = data["countries"]
    imposer_idx = countries.index(country)
    total = len(rates)
    rows: list[dict] = []

    for step, rate in enumerate(rates, 1):
        if _progress_callback is not None:
            _progress_callback(step, total)

        scenario = {
            "type": "uniform",
            "country": country,
            "rate": rate,
        }

        try:
            res = solve_counterfactual(
                data, elasticity, scenario, retaliation=retaliation,
            )
            converged = bool(res["converged"])
            imposer_welfare = float(res["welfare"][imposer_idx])
            world_avg = _safe_mean(res["welfare"])
        except Exception:
            converged = False
            imposer_welfare = np.nan
            world_avg = np.nan

        rows.append({
            "Tariff_Rate_Pct": round(100 * rate, 2),
            "Imposer_Welfare_Pct": imposer_welfare,
            "World_Avg_Welfare_Pct": world_avg,
            "Converged": converged,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
#  4. Retaliation comparison
# ------------------------------------------------------------------

@st.cache_data(ttl=7200, show_spinner=False)
def run_retaliation_comparison(
    _data: dict,
    elasticity: str,
    scenario: dict,
    *,
    _data_hash: str | None = None,
    _progress_callback=None,
) -> pd.DataFrame:
    """Run a scenario with and without retaliation and return a
    side-by-side comparison.

    Parameters
    ----------
    _data : dict
        Output of ``load_trade_data``.
    elasticity : str
        Key into ELASTICITY_REGISTRY.
    scenario : dict
        Tariff scenario (passed to ``solve_counterfactual``).
    progress_callback : callable | None
        ``callback(current_step, total_steps)``.

    Returns
    -------
    pd.DataFrame
        Columns: Country, Welfare_NoRetaliation_Pct,
        Welfare_Reciprocal_Pct, Welfare_Diff_Pct,
        Converged_NoRetaliation, Converged_Reciprocal.
    """
    data = _data
    countries = data["countries"]
    total = 2

    # --- No retaliation ---
    if _progress_callback is not None:
        _progress_callback(1, total)

    try:
        res_none = solve_counterfactual(data, elasticity, scenario, retaliation="none")
        w_none = res_none["welfare"]
        conv_none = bool(res_none["converged"])
    except Exception:
        w_none = np.full(len(countries), np.nan)
        conv_none = False

    # --- Reciprocal retaliation ---
    if _progress_callback is not None:
        _progress_callback(2, total)

    try:
        res_recip = solve_counterfactual(data, elasticity, scenario, retaliation="reciprocal")
        w_recip = res_recip["welfare"]
        conv_recip = bool(res_recip["converged"])
    except Exception:
        w_recip = np.full(len(countries), np.nan)
        conv_recip = False

    rows: list[dict] = []
    for cidx, country in enumerate(countries):
        wn = float(w_none[cidx])
        wr = float(w_recip[cidx])
        rows.append({
            "Country": country,
            "Welfare_NoRetaliation_Pct": wn,
            "Welfare_Reciprocal_Pct": wr,
            "Welfare_Diff_Pct": wr - wn,
            "Converged_NoRetaliation": conv_none,
            "Converged_Reciprocal": conv_recip,
        })

    return pd.DataFrame(rows)
