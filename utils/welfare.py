"""
Welfare calculations using the ACR (Arkolakis, Costinot, Rodriguez-Clare 2012)
sufficient statistic framework.

Key result: The welfare change from a trade shock is fully characterized by:
  Ŵ_j = λ̂_jj^(-1/ε)

where:
  λ_jj = domestic trade share (share of country j's expenditure on its own goods)
  ε = trade elasticity
  ̂  = proportional change

This remarkably general result holds across a wide class of trade models
(Armington, Eaton-Kortum, Melitz) and requires only two sufficient statistics:
the change in domestic trade share and the trade elasticity.

References:
- Arkolakis, Costinot, Rodriguez-Clare (2012) "New Trade Models, Same Old Gains?"
  American Economic Review.
- Costinot & Rodriguez-Clare (2014) "Trade Theory with Numbers"
"""
import numpy as np
import pandas as pd
import streamlit as st


def compute_domestic_share(trade_df: pd.DataFrame, country_iso: str, gdp: float = None) -> float:
    """
    Compute domestic trade share λ_jj.

    λ_jj = domestic_expenditure / total_expenditure
         ≈ 1 - import_penetration_ratio
         = 1 - (total_imports / (GDP or approx total absorption))

    If GDP not provided, approximate total absorption as imports + exports
    (rough proxy).
    """
    total_imports = trade_df[
        trade_df["iso_d"] == country_iso
    ]["trade_value_usd_millions"].sum()

    if gdp is not None and gdp > 0:
        # Convert GDP to millions to match trade data
        import_penetration = total_imports / gdp
        domestic_share = max(0.01, 1.0 - import_penetration)
    else:
        # Rough approximation using total trade
        total_exports = trade_df[
            trade_df["iso_o"] == country_iso
        ]["trade_value_usd_millions"].sum()
        # Approximate absorption = GDP ≈ production ≈ exports * (1/openness)
        # Use a simple heuristic: domestic share ≈ 1 - imports/(imports + 2*exports)
        total = total_imports + 2 * total_exports
        domestic_share = max(0.01, 1 - total_imports / total) if total > 0 else 0.5

    return domestic_share


def welfare_change_acr(
    lambda_jj_before: float,
    lambda_jj_after: float,
    elasticity: float,
) -> float:
    """
    ACR sufficient statistic for welfare change.

    Ŵ = (λ̂_jj)^(-1/ε)
      = (λ_jj_after / λ_jj_before)^(-1/ε)

    Returns percentage welfare change (positive = gain, negative = loss).
    """
    if lambda_jj_before <= 0 or lambda_jj_after <= 0 or elasticity <= 0:
        return 0.0

    lambda_hat = lambda_jj_after / lambda_jj_before
    welfare_hat = lambda_hat ** (-1.0 / elasticity)
    return (welfare_hat - 1.0) * 100  # percentage change


def compute_welfare_impact(
    trade_df: pd.DataFrame,
    country_iso: str,
    tariff_targets: dict,
    elasticity: float = 5.0,
    gdp: float = None,
) -> dict:
    """
    Compute welfare impact of tariff changes using ACR framework.

    Parameters
    ----------
    trade_df : bilateral trade data for a year
    country_iso : country imposing tariffs
    tariff_targets : dict of {partner_iso: tariff_pct}
    elasticity : trade elasticity
    gdp : GDP in millions (same units as trade data)

    Returns
    -------
    dict with welfare calculations
    """
    from utils.tariff_sim import simulate_tariff_shock

    # Before: current domestic share
    lambda_before = compute_domestic_share(trade_df, country_iso, gdp)

    # Compute total import change from tariff shocks
    total_import_change = 0.0
    total_imports_before = trade_df[
        trade_df["iso_d"] == country_iso
    ]["trade_value_usd_millions"].sum()

    for partner, tariff_pct in tariff_targets.items():
        result = simulate_tariff_shock(
            trade_df, partner, country_iso, tariff_pct, elasticity, bilateral=False
        )
        total_import_change += result["import_loss"]

    # After: new domestic share (imports decreased, so domestic share increased)
    if gdp is not None and gdp > 0:
        new_imports = total_imports_before - total_import_change
        lambda_after = max(0.01, 1.0 - new_imports / gdp)
    else:
        total_exports = trade_df[
            trade_df["iso_o"] == country_iso
        ]["trade_value_usd_millions"].sum()
        total_before = total_imports_before + 2 * total_exports
        new_imports = total_imports_before - total_import_change
        lambda_after = max(0.01, 1 - new_imports / total_before) if total_before > 0 else 0.5

    welfare_pct = welfare_change_acr(lambda_before, lambda_after, elasticity)

    # Convert to dollar terms if GDP available
    if gdp:
        welfare_dollars = welfare_pct / 100 * gdp
    else:
        welfare_dollars = None

    return {
        "country": country_iso,
        "lambda_before": lambda_before,
        "lambda_after": lambda_after,
        "lambda_change_pct": (lambda_after / lambda_before - 1) * 100,
        "welfare_change_pct": welfare_pct,
        "welfare_change_dollars_M": welfare_dollars,
        "total_import_reduction_M": total_import_change,
        "elasticity": elasticity,
        "method": "ACR (2012) sufficient statistic",
        "interpretation": (
            f"{'Welfare loss' if welfare_pct < 0 else 'Welfare gain'} of {abs(welfare_pct):.3f}% "
            f"({'≈ $' + f'{abs(welfare_dollars):,.0f}M' if welfare_dollars else 'GDP needed for $ estimate'})"
        ),
    }


def compute_multi_country_welfare(
    trade_df: pd.DataFrame,
    tariff_targets: dict,
    elasticity: float = 5.0,
) -> pd.DataFrame:
    """
    Compute welfare for both the imposing country AND all targeted countries.
    Shows who actually loses from tariffs.
    """
    results = []

    # For the imposing country (USA usually)
    us_welfare = compute_welfare_impact(
        trade_df, "USA", tariff_targets, elasticity,
        gdp=25_000_000,  # ~$25T US GDP in millions
    )
    results.append({
        "country": "USA",
        "role": "Imposing",
        "welfare_change_pct": us_welfare["welfare_change_pct"],
        "welfare_dollars_M": us_welfare["welfare_change_dollars_M"],
        "domestic_share_before": us_welfare["lambda_before"],
        "domestic_share_after": us_welfare["lambda_after"],
    })

    # For each targeted country (retaliation effect)
    # Approximate GDPs (2022, millions USD)
    approx_gdps = {
        "CHN": 18_000_000, "DEU": 4_100_000, "JPN": 4_200_000,
        "GBR": 3_100_000, "IND": 3_400_000, "FRA": 2_800_000,
        "CAN": 2_100_000, "MEX": 1_400_000, "KOR": 1_700_000,
        "BRA": 1_900_000, "AUS": 1_700_000, "IDN": 1_300_000,
        "VNM": 410_000, "THA": 500_000, "MYS": 410_000,
    }

    for partner, tariff_pct in tariff_targets.items():
        gdp = approx_gdps.get(partner)
        # The targeted country faces reduced exports (welfare loss from reduced market access)
        partner_welfare = compute_welfare_impact(
            trade_df, partner, {"USA": tariff_pct}, elasticity, gdp=gdp
        )
        results.append({
            "country": partner,
            "role": "Targeted",
            "welfare_change_pct": partner_welfare["welfare_change_pct"],
            "welfare_dollars_M": partner_welfare["welfare_change_dollars_M"],
            "domestic_share_before": partner_welfare["lambda_before"],
            "domestic_share_after": partner_welfare["lambda_after"],
        })

    return pd.DataFrame(results).sort_values("welfare_change_pct")
