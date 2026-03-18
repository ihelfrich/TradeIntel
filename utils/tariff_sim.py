"""
Tariff shock simulator using trade elasticity approach.

Based on the structural gravity literature:
- Caliendo & Parro (2015): sectoral trade elasticities ~4-6
- Head & Mayer (2014): average trade elasticity ~5
- Anderson & van Wincoop (2003): theoretical foundations

Model: When tariff on route (i->j) increases by Δτ,
  new_trade_ij = old_trade_ij × (1 + Δτ)^(-ε)
  where ε is the trade elasticity (default ~5)

Trade diversion: reduced trade is partially redirected to
alternative partners proportional to their existing trade shares.
"""
import pandas as pd
import numpy as np
import streamlit as st


DEFAULT_ELASTICITY = 5.0
DIVERSION_RATE = 0.6  # 60% of lost trade diverts to other partners


def simulate_tariff_shock(
    trade_df: pd.DataFrame,
    target_iso: str,
    imposing_iso: str,
    tariff_increase_pct: float,
    elasticity: float = DEFAULT_ELASTICITY,
    diversion_rate: float = DIVERSION_RATE,
    bilateral: bool = True,
) -> dict:
    """
    Simulate the impact of a tariff increase between two countries.

    Parameters
    ----------
    trade_df : bilateral trade for a given year
    target_iso : country being tariffed (e.g., "CHN")
    imposing_iso : country imposing tariff (e.g., "USA")
    tariff_increase_pct : tariff increase in percentage points (e.g., 25)
    elasticity : trade elasticity (default 5.0)
    diversion_rate : fraction of lost trade diverted to other partners
    bilateral : if True, also simulate retaliation

    Returns
    -------
    dict with simulation results
    """
    df = trade_df.copy()
    tariff_mult = tariff_increase_pct / 100.0

    results = {
        "imposing": imposing_iso,
        "target": target_iso,
        "tariff_pct": tariff_increase_pct,
        "elasticity": elasticity,
    }

    # --- Impact on imports: imposing_iso imports from target_iso ---
    mask_imports = (df["iso_o"] == target_iso) & (df["iso_d"] == imposing_iso)
    old_imports = df.loc[mask_imports, "trade_value_usd_millions"].sum()
    new_imports = old_imports * (1 + tariff_mult) ** (-elasticity)
    import_loss = old_imports - new_imports

    results["import_loss"] = import_loss
    results["old_imports_from_target"] = old_imports
    results["new_imports_from_target"] = new_imports

    # --- Trade diversion on import side ---
    # Other countries that export to imposing_iso (excluding target)
    other_exporters = df[
        (df["iso_d"] == imposing_iso) & (df["iso_o"] != target_iso)
    ].copy()

    if len(other_exporters) > 0 and other_exporters["trade_value_usd_millions"].sum() > 0:
        other_exporters["share"] = (
            other_exporters["trade_value_usd_millions"]
            / other_exporters["trade_value_usd_millions"].sum()
        )
        diverted_imports = import_loss * diversion_rate
        other_exporters["trade_gain"] = other_exporters["share"] * diverted_imports
        results["import_diversion"] = other_exporters[
            ["iso_o", "trade_value_usd_millions", "trade_gain"]
        ].nlargest(20, "trade_gain").copy()
        results["import_diversion"].columns = ["country", "current_trade", "trade_gain"]
    else:
        results["import_diversion"] = pd.DataFrame(columns=["country", "current_trade", "trade_gain"])

    # --- Impact on exports: imposing_iso exports to target_iso (retaliation) ---
    if bilateral:
        mask_exports = (df["iso_o"] == imposing_iso) & (df["iso_d"] == target_iso)
        old_exports = df.loc[mask_exports, "trade_value_usd_millions"].sum()
        new_exports = old_exports * (1 + tariff_mult) ** (-elasticity)
        export_loss = old_exports - new_exports
        results["export_loss"] = export_loss
        results["old_exports_to_target"] = old_exports
        results["new_exports_to_target"] = new_exports

        # Export diversion
        other_importers = df[
            (df["iso_o"] == imposing_iso) & (df["iso_d"] != target_iso)
        ].copy()

        if len(other_importers) > 0 and other_importers["trade_value_usd_millions"].sum() > 0:
            other_importers["share"] = (
                other_importers["trade_value_usd_millions"]
                / other_importers["trade_value_usd_millions"].sum()
            )
            diverted_exports = export_loss * diversion_rate
            other_importers["trade_gain"] = other_importers["share"] * diverted_exports
            results["export_diversion"] = other_importers[
                ["iso_d", "trade_value_usd_millions", "trade_gain"]
            ].nlargest(20, "trade_gain").copy()
            results["export_diversion"].columns = ["country", "current_trade", "trade_gain"]
        else:
            results["export_diversion"] = pd.DataFrame(columns=["country", "current_trade", "trade_gain"])
    else:
        results["export_loss"] = 0
        results["old_exports_to_target"] = 0
        results["new_exports_to_target"] = 0
        results["export_diversion"] = pd.DataFrame(columns=["country", "current_trade", "trade_gain"])

    # --- Total bilateral impact ---
    results["total_bilateral_loss"] = results["import_loss"] + results["export_loss"]
    results["total_trade_before"] = results["old_imports_from_target"] + results["old_exports_to_target"]
    results["total_trade_after"] = results["new_imports_from_target"] + results["new_exports_to_target"]
    results["pct_reduction"] = (
        results["total_bilateral_loss"] / results["total_trade_before"] * 100
        if results["total_trade_before"] > 0 else 0
    )

    return results


def simulate_multi_country_tariff(
    trade_df: pd.DataFrame,
    imposing_iso: str,
    targets: dict,
    elasticity: float = DEFAULT_ELASTICITY,
) -> pd.DataFrame:
    """
    Simulate tariffs on multiple countries simultaneously.

    Parameters
    ----------
    targets : dict of {iso3: tariff_pct}, e.g. {"CHN": 60, "MEX": 25, "CAN": 25}
    """
    records = []
    for target_iso, tariff_pct in targets.items():
        result = simulate_tariff_shock(
            trade_df, target_iso, imposing_iso, tariff_pct, elasticity,
        )
        records.append({
            "target": target_iso,
            "tariff_pct": tariff_pct,
            "old_bilateral_trade": result["total_trade_before"],
            "new_bilateral_trade": result["total_trade_after"],
            "trade_loss": result["total_bilateral_loss"],
            "pct_reduction": result["pct_reduction"],
            "import_loss": result["import_loss"],
            "export_loss": result["export_loss"],
        })

    df = pd.DataFrame(records)
    df["trade_loss_billions"] = df["trade_loss"] / 1000
    return df.sort_values("trade_loss", ascending=False)


# --- Current tariff scenarios (approximate as of early 2026) ---
SCENARIOS = {
    "Trump 2025-26 Tariffs (Approximate)": {
        "CHN": 60,   # 60% on China
        "MEX": 25,   # 25% on Mexico
        "CAN": 25,   # 25% on Canada
        "DEU": 20,   # EU tariffs ~20%
        "JPN": 20,   # Japan ~20%
        "KOR": 25,   # South Korea ~25%
        "VNM": 46,   # Vietnam ~46%
        "IND": 26,   # India ~26%
        "THA": 36,   # Thailand ~36%
        "IDN": 32,   # Indonesia ~32%
        "MYS": 24,   # Malaysia ~24%
    },
    "Universal 10% Baseline": {
        country: 10 for country in [
            "CHN", "MEX", "CAN", "DEU", "JPN", "KOR", "GBR", "FRA",
            "ITA", "IND", "BRA", "VNM", "THA", "IDN", "MYS",
        ]
    },
    "China Decoupling (100% on China)": {
        "CHN": 100,
    },
    "Pre-2025 Baseline (Low Tariffs)": {
        "CHN": 3,
        "MEX": 0,
        "CAN": 0,
        "DEU": 2,
        "JPN": 2,
    },
}
