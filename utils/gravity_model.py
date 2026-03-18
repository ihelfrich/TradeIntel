"""
Structural gravity model estimation.

Estimates trade elasticities from the CEPII gravity dataset using:
1. OLS on log-linearized gravity equation (traditional)
2. PPML via iteratively reweighted least squares (Santos Silva & Tenreyro 2006)

The gravity equation:
  X_ij = exp(β₁·lnGDP_i + β₂·lnGDP_d + β₃·ln(dist_ij) + β₄·contig + β₅·comlang
             + β₆·col_dep + β₇·rta + β₈·τ_ij + FE_i + FE_j) × η_ij

Key references:
- Anderson & van Wincoop (2003): Theoretical foundations
- Santos Silva & Tenreyro (2006): PPML estimator
- Head & Mayer (2014): Survey of gravity estimates
- Caliendo & Parro (2015): Sectoral trade elasticities
- Yotov et al. (2016): Advanced guide to trade policy with gravity
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy import optimize
from pathlib import Path

GRAVITY_PATH = Path.home() / "trade_data_warehouse" / "gravity" / "gravity_v202211.parquet"


@st.cache_data(ttl=3600)
def load_gravity_for_estimation(year: int = 2019) -> pd.DataFrame:
    """Load and prepare gravity data for estimation."""
    cols = [
        "year", "iso3_o", "iso3_d",
        "distw_harmonic", "contig", "comlang_off", "col_dep_ever",
        "comleg_posttrans", "comrelig",
        "gdp_o", "gdp_d", "pop_o", "pop_d",
        "wto_o", "wto_d", "eu_o", "eu_d",
        "fta_wto", "rta_type",
        "entry_cost_d", "entry_tp_d",
        "tradeflow_baci",
        "diplo_disagreement", "scaled_sci_2021",
    ]
    gv = pd.read_parquet(GRAVITY_PATH, columns=cols)
    gv = gv[gv["year"] == year].copy()

    # Drop self-trade
    gv = gv[gv["iso3_o"] != gv["iso3_d"]]

    # Log transforms
    gv["ln_dist"] = np.log(gv["distw_harmonic"].clip(lower=1))
    gv["ln_gdp_o"] = np.log(gv["gdp_o"].clip(lower=1))
    gv["ln_gdp_d"] = np.log(gv["gdp_d"].clip(lower=1))
    gv["ln_trade"] = np.log(gv["tradeflow_baci"].clip(lower=0.001))
    gv["trade"] = gv["tradeflow_baci"].clip(lower=0)

    # Both WTO members
    gv["both_wto"] = ((gv["wto_o"] == 1) & (gv["wto_d"] == 1)).astype(float)
    gv["both_eu"] = ((gv["eu_o"] == 1) & (gv["eu_d"] == 1)).astype(float)

    # Clean
    gv = gv.dropna(subset=["ln_dist", "ln_gdp_o", "ln_gdp_d", "trade"])
    gv = gv.replace([np.inf, -np.inf], np.nan).dropna(subset=["ln_trade"])

    return gv


@st.cache_data(ttl=3600)
def estimate_ols_gravity(year: int = 2019) -> dict:
    """
    OLS estimation of the log-linearized gravity equation.
    log(X_ij) = α + β₁·ln(GDP_i) + β₂·ln(GDP_j) - β₃·ln(dist_ij)
                + β₄·contig + β₅·comlang + β₆·colonial + β₇·RTA + ε
    """
    gv = load_gravity_for_estimation(year)

    # Only keep positive trade flows for OLS (known bias — PPML fixes this)
    gv_pos = gv[gv["trade"] > 0].copy()

    y = gv_pos["ln_trade"].values

    # Design matrix
    X_data = gv_pos[[
        "ln_gdp_o", "ln_gdp_d", "ln_dist",
        "contig", "comlang_off", "col_dep_ever",
        "fta_wto", "both_wto", "both_eu",
    ]].fillna(0).values

    var_names = [
        "ln_GDP_origin", "ln_GDP_dest", "ln_distance",
        "contiguity", "common_language", "colonial_link",
        "FTA/RTA", "both_WTO", "both_EU",
    ]

    # Add constant
    ones = np.ones((X_data.shape[0], 1))
    X = np.hstack([ones, X_data])

    # OLS: β = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    n, k = X.shape

    # Standard errors (heteroskedasticity-robust HC1)
    sigma2 = residuals ** 2
    bread = np.linalg.inv(XtX)
    meat = X.T @ np.diag(sigma2) @ X
    V_robust = (n / (n - k)) * bread @ meat @ bread
    se_robust = np.sqrt(np.diag(V_robust))

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # t-statistics
    t_stats = beta / se_robust

    results = pd.DataFrame({
        "variable": ["constant"] + var_names,
        "coefficient": beta,
        "std_error": se_robust,
        "t_statistic": t_stats,
        "significant": np.abs(t_stats) > 1.96,
    })

    return {
        "results": results,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "n_obs": n,
        "n_vars": k,
        "year": year,
        "method": "OLS (log-linear, HC1 robust SE)",
        "distance_elasticity": float(beta[3]),  # ln_dist coefficient
        "note": "OLS on log trade drops zero flows — known downward bias on distance. PPML preferred.",
    }


@st.cache_data(ttl=3600)
def estimate_ppml_gravity(year: int = 2019, max_iter: int = 50) -> dict:
    """
    Poisson Pseudo-Maximum Likelihood estimation (Santos Silva & Tenreyro 2006).
    Solves: E[X_ij | Z] = exp(Z_ij · β)
    Via iteratively reweighted least squares (IRLS).
    Includes zero trade flows — unbiased under heteroskedasticity.
    """
    gv = load_gravity_for_estimation(year)

    y = gv["trade"].values.astype(np.float64)

    X_data = gv[[
        "ln_gdp_o", "ln_gdp_d", "ln_dist",
        "contig", "comlang_off", "col_dep_ever",
        "fta_wto", "both_wto", "both_eu",
    ]].fillna(0).values.astype(np.float64)

    var_names = [
        "ln_GDP_origin", "ln_GDP_dest", "ln_distance",
        "contiguity", "common_language", "colonial_link",
        "FTA/RTA", "both_WTO", "both_EU",
    ]

    ones = np.ones((X_data.shape[0], 1))
    X = np.hstack([ones, X_data])
    n, k = X.shape

    # Initialize with OLS on log(1+y)
    y_log = np.log(1 + y)
    try:
        beta = np.linalg.lstsq(X, y_log, rcond=None)[0]
    except Exception:
        beta = np.zeros(k)

    # IRLS iterations
    converged = False
    for iteration in range(max_iter):
        eta = X @ beta
        # Clip to prevent overflow
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)

        # Working response
        z = eta + (y - mu) / mu

        # Weights
        W = mu

        # Weighted least squares
        WX = X * W[:, np.newaxis]
        XtWX = WX.T @ X
        XtWz = WX.T @ z

        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(WX, z * np.sqrt(W), rcond=None)[0]

        # Check convergence
        change = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if change < 1e-8:
            converged = True
            break

    # Final fitted values
    eta = X @ beta
    eta = np.clip(eta, -30, 30)
    mu = np.exp(eta)

    # Robust standard errors (sandwich estimator)
    residuals = y - mu
    WX = X * mu[:, np.newaxis]
    bread = np.linalg.inv(WX.T @ X + np.eye(k) * 1e-10)
    e_scaled = X * (residuals[:, np.newaxis])
    meat = e_scaled.T @ e_scaled
    V_robust = bread @ meat @ bread
    se_robust = np.sqrt(np.abs(np.diag(V_robust)))

    # Pseudo R-squared (based on deviance)
    deviance = 2 * np.sum(y * np.log((y + 1e-10) / (mu + 1e-10)) - (y - mu))
    null_mu = np.mean(y)
    null_deviance = 2 * np.sum(y * np.log((y + 1e-10) / null_mu) - (y - null_mu))
    pseudo_r2 = 1 - deviance / null_deviance if null_deviance > 0 else 0

    t_stats = beta / (se_robust + 1e-10)

    results = pd.DataFrame({
        "variable": ["constant"] + var_names,
        "coefficient": beta,
        "std_error": se_robust,
        "t_statistic": t_stats,
        "significant": np.abs(t_stats) > 1.96,
    })

    return {
        "results": results,
        "pseudo_r_squared": pseudo_r2,
        "n_obs": n,
        "n_vars": k,
        "year": year,
        "method": f"PPML (IRLS, {'converged' if converged else 'max iter'}, {iteration+1} iterations)",
        "distance_elasticity": float(beta[3]),
        "converged": converged,
        "iterations": iteration + 1,
    }


@st.cache_data(ttl=3600)
def estimate_sector_elasticities() -> pd.DataFrame:
    """
    Estimate sector-specific trade elasticities using sectoral tariff data.
    Uses the relationship: ln(trade_ij_s) = ... - ε_s · τ_ij_s + ...
    where τ is the log tariff measure from the tau dataset.
    """
    from utils.data_loader import load_sector_flows, SECTOR_LABELS

    sector_flows = load_sector_flows()
    tau_sectoral = pd.read_parquet(
        Path.home() / "trade_data_warehouse" / "tau" / "tau_sectoral.parquet"
    )

    # Merge trade flows with tariffs
    merged = sector_flows.merge(
        tau_sectoral,
        on=["year", "iso_o", "iso_d", "oecd_sector"],
        how="inner",
    )
    merged["ln_trade"] = np.log(merged["value"].clip(lower=0.001))

    # Load gravity for controls
    gv = load_gravity_for_estimation(2019)
    gv_controls = gv[["iso3_o", "iso3_d", "ln_dist", "contig", "comlang_off", "ln_gdp_o", "ln_gdp_d"]].copy()
    gv_controls = gv_controls.rename(columns={"iso3_o": "iso_o", "iso3_d": "iso_d"})

    merged = merged.merge(gv_controls, on=["iso_o", "iso_d"], how="inner")
    merged = merged.dropna()

    results = []
    for sector in sorted(merged["oecd_sector"].unique()):
        sec_data = merged[merged["oecd_sector"] == sector]
        if len(sec_data) < 50:
            continue

        y = sec_data["ln_trade"].values
        X_data = sec_data[["tau_log", "ln_dist", "contig", "comlang_off", "ln_gdp_o", "ln_gdp_d"]].values
        ones = np.ones((X_data.shape[0], 1))
        X = np.hstack([ones, X_data])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ beta
            residuals = y - y_hat
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # The tariff elasticity is beta[1] (coefficient on tau_log)
            # In the gravity equation: ln(X) = ... - ε·ln(1+τ) + ...
            # tau_log ≈ ln(1+τ), so beta[1] ≈ -ε
            tariff_elasticity = -beta[1]

            results.append({
                "sector": sector,
                "sector_name": SECTOR_LABELS.get(sector, sector),
                "tariff_elasticity": tariff_elasticity,
                "distance_elasticity": -beta[2],
                "r_squared": r2,
                "n_obs": len(sec_data),
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("tariff_elasticity", ascending=False)
    return df


@st.cache_data(ttl=3600)
def compute_gravity_predicted_matrix(year: int = 2019, top_n: int = 50) -> dict:
    """
    Build matrices of actual vs. gravity-predicted bilateral trade flows.

    Returns W_actual, W_predicted, R_residual (= actual / predicted).
    The residual matrix is the key input for "gravity-residual topology" —
    persistent homology on deviations from gravity captures trade structure
    NOT explained by GDP, distance, language, colonial ties, or RTAs.
    """
    from utils.data_loader import load_bilateral_trade

    # Load gravity data and run PPML to get beta
    gv = load_gravity_for_estimation(year)

    # Get top_n countries by actual total trade
    trade_full = load_bilateral_trade()
    trade_year = trade_full[trade_full["year"] == year]
    country_trade = (
        trade_year.groupby("iso_o")["trade_value_usd_millions"].sum()
        .add(trade_year.groupby("iso_d")["trade_value_usd_millions"].sum(), fill_value=0)
    )
    top_countries = country_trade.nlargest(top_n).index.tolist()
    countries = sorted(top_countries)
    idx = {c: i for i, c in enumerate(countries)}
    n = len(countries)

    # Run PPML on full gravity data to get coefficients
    y = gv["trade"].values.astype(np.float64)
    X_cols = ["ln_gdp_o", "ln_gdp_d", "ln_dist",
              "contig", "comlang_off", "col_dep_ever",
              "fta_wto", "both_wto", "both_eu"]
    X_data = gv[X_cols].fillna(0).values.astype(np.float64)
    ones = np.ones((X_data.shape[0], 1))
    X = np.hstack([ones, X_data])

    # IRLS for PPML
    y_log = np.log(1 + y)
    try:
        beta = np.linalg.lstsq(X, y_log, rcond=None)[0]
    except Exception:
        beta = np.zeros(X.shape[1])

    for _ in range(50):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        WX = X * mu[:, np.newaxis]
        try:
            beta_new = np.linalg.solve(WX.T @ X, WX.T @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    # Compute predicted values for all pairs in gravity data
    eta_final = np.clip(X @ beta, -30, 30)
    mu_final = np.exp(eta_final)
    gv = gv.copy()
    gv["predicted"] = mu_final

    # Filter to top_n countries and build matrices
    gv_top = gv[gv["iso3_o"].isin(countries) & gv["iso3_d"].isin(countries)].copy()

    W_actual = np.zeros((n, n))
    W_predicted = np.zeros((n, n))

    for _, row in gv_top.iterrows():
        i, j = idx.get(row["iso3_o"]), idx.get(row["iso3_d"])
        if i is not None and j is not None and i != j:
            W_actual[i, j] = row["trade"]
            W_predicted[i, j] = row["predicted"]

    # Also fill actual from BACI (more complete coverage)
    trade_top = trade_year[
        trade_year["iso_o"].isin(countries) & trade_year["iso_d"].isin(countries)
    ]
    for _, row in trade_top.iterrows():
        i, j = idx.get(row["iso_o"]), idx.get(row["iso_d"])
        if i is not None and j is not None:
            W_actual[i, j] = row["trade_value_usd_millions"]

    # Residual matrix: actual / predicted (surprise factor)
    R_residual = np.where(
        W_predicted > 0,
        W_actual / W_predicted,
        np.where(W_actual > 0, 10.0, 0.0),  # actual > 0 but predicted ≈ 0 → big surprise
    )

    return {
        "W_actual": W_actual,
        "W_predicted": W_predicted,
        "R_residual": R_residual,
        "countries": countries,
        "beta": beta.tolist(),
        "year": year,
    }
