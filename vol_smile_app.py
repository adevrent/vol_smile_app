import streamlit as st
import QuantLib as ql
from FX_option_pricer import OptionParams
import numpy as np
import scipy.stats as ss
import pandas as pd   # ← new

st.title("Volatility Smile Calculator")
# … all your input widgets remain unchanged …

if st.button("Compute"):
    # … parsing & params initialization unchanged …

    # calibrate smile
    params.optimize_sigma_S()
    params.set_K_C_P()

    # Compute for the chosen strike & option type
    sigma_K    = params.find_SPI_sigma_K(call_put, K)
    BS_results = params.BS(call_put, K, sigma_K)
    TV_greeks  = params.calc_TV_greeks(call_put, K)

    # ── Prices Table ─────────────────────────────────────────────────────────
    prices_df = pd.DataFrame({
        "Price Type": [f"{call_put} (domestic)", f"{call_put} (foreign)"],
        "Value":      [f"{BS_results['v_dom']*100:.4f}%", f"{BS_results['v_for']*100:.4f}%"]
    })
    st.subheader("Prices")
    st.table(prices_df)

    # ── Real Greeks ───────────────────────────────────────────────────────────
    real_df = pd.DataFrame({
        "Greek": [
            "Spot Δ", "Forward Δ",
            "Spot Δ (PA)", "Forward Δ (PA)"
        ],
        "Value": [
            f"{BS_results['delta_S']*100:.4f}%",
            f"{BS_results['delta_fwd']*100:.4f}%",
            f"{BS_results['delta_S_pa']*100:.4f}%",
            f"{BS_results['delta_fwd_pa']*100:.4f}%"
        ]
    })
    st.subheader("Real Greeks")
    st.table(real_df)

    # ── TV Greeks ─────────────────────────────────────────────────────────────
    tv_df = pd.DataFrame({
        "Greek": [
            "Spot Δ", "Forward Δ",
            "Spot Δ (PA)", "Forward Δ (PA)"
        ],
        "Value": [
            f"{TV_greeks['delta_S']*100:.4f}%",
            f"{TV_greeks['delta_fwd']*100:.4f}%",
            f"{TV_greeks['delta_S_pa']*100:.4f}%",
            f"{TV_greeks['delta_fwd_pa']*100:.4f}%"
        ]
    })
    st.subheader("TV Greeks")
    st.table(tv_df)
