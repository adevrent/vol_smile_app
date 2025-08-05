import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
from FX_Option_Pricer import OptionParams, calc_tx_with_spreads

st.title("SPI Vol Smile & TX Calculator")

# 1) Day-count bases
basis_map = {
    "ACTL360": ql.Actual360(),
    "ACTL365": ql.Actual365Fixed(),
}
col1, col2 = st.columns(2)
with col1:
    basis_for = st.selectbox("FOR Day-count Basis", list(basis_map.keys()), index=0)
with col2:
    basis_dom = st.selectbox("DOM Day-count Basis", list(basis_map.keys()), index=0)
basis_dict = {"FOR": basis_map[basis_for], "DOM": basis_map[basis_dom]}

# 2) Dates as text inputs
eval_date_str     = st.text_input("Eval Date (YYYY-MM-DD)", placeholder="2025-07-08")
expiry_date_str   = st.text_input("Expiry Date (YYYY-MM-DD)", placeholder="2025-07-23")
delivery_date_str = st.text_input("Delivery Date (YYYY-MM-DD)", placeholder="2025-07-24")

def parse_ql_date(s: str) -> ql.Date:
    try:
        y, m, d = map(int, s.split("-"))
        return ql.Date(d, m, y)
    except Exception:
        st.error(f"Invalid date format: {s}")
        st.stop()

# 3) Market inputs
col1, col2, col3 = st.columns(3)
with col1:
    x_str         = st.text_input("Spot Price", placeholder="e.g. 39.729")
with col2:
    rd_spread_str = st.text_input("Domestic Rate Spread (%)", placeholder="e.g. 45")
with col3:
    rf_spread_str = st.text_input("Foreign Rate Spread (%)", placeholder="e.g. 5")

col4, col5 = st.columns(2)
with col4:
    ATM_vol_spread_str = st.text_input("ATM vol Spread (%)", placeholder="e.g. 2.25")
with col5:
    sigma_ATM_str = st.text_input("ATM VOL (%)", placeholder="e.g. 12.00")
sigma_RR_str  = st.text_input("RR VOL (%)", placeholder="e.g. 13.00")
sigma_SQ_str  = st.text_input("BF VOL (%)", placeholder="e.g. 1.75")

# 4) Pillar delta
delta_tilde = st.selectbox("Pillar Delta", [0.10, 0.25], index=0)

# 5) BUY/SELL and Convention
buy_sell = st.selectbox("BUY/SELL", ["BUY", "SELL"], index=0)
convention = st.selectbox("Convention", ["Convention A", "Convention B"], index=0)
if convention == "Convention A":
    K_ATM_conv = "fwd_delta_neutral"
    delta_conv = "spot_pa"
else:
    K_ATM_conv = "fwd"
    delta_conv = "spot"

# 6) Strike & Call/Put
K_str      = st.text_input("Strike Price", placeholder="e.g. 41.00")
call_put   = st.selectbox("CALL/PUT", ["CALL","PUT"], index=1)

# 7) Compute
if st.button("Compute"):
    # parse dates
    eval_date     = parse_ql_date(eval_date_str)
    expiry_date   = parse_ql_date(expiry_date_str)
    delivery_date = parse_ql_date(delivery_date_str)

    # parse numbers and spreads
    try:
        x               = float(x_str)
        rd_spread       = float(rd_spread_str) / 100
        rf_spread       = float(rf_spread_str) / 100
        ATM_vol_spread  = float(ATM_vol_spread_str) / 100
        rd_simple       = float(sigma_ATM_str) / 100  # placeholder, will override below
        rf_simple       = float(sigma_RR_str) / 100   # placeholder
        sigma_ATM       = float(sigma_ATM_str) / 100
        sigma_RR        = float(sigma_RR_str)  / 100
        sigma_SQ        = float(sigma_SQ_str)  / 100
        K               = float(K_str)
    except Exception as e:
        st.error(f"Numeric input error: {e}")
        st.stop()

    # Calculate base simple rates from spreads
    # Using spreads as simple rates here for rd_simple/rf_simple
    rd_simple = rd_spread
    rf_simple = rf_spread

    # Run calculation
    # First, initialize parameters to extract forward parity
    params = OptionParams(
        calendar=ql.Turkey(),
        basis_dict=basis_dict,
        spot_bd=1,
        eval_date=eval_date,
        expiry_date=expiry_date,
        delivery_date=delivery_date,
        x=x,
        rd_simple=rd_simple,
        rf_simple=rf_simple,
        sigma_ATM=sigma_ATM,
        sigma_RR=sigma_RR,
        sigma_SQ=sigma_SQ,
        delta_tilde=delta_tilde,
        K_ATM_convention=K_ATM_conv,
        delta_convention=delta_conv
    )
    # Compute smile calibration
    params.optimize_sigma_S()
    params.set_K_C_P()

    # Now compute transaction with spreads
df = calc_tx_with_spreads(
    buy_sell, call_put, K,
    rd_spread, rf_spread, ATM_vol_spread,
    ql.Turkey(), basis_dict, 1,
    eval_date, expiry_date, delivery_date,
    x, rd_simple, rf_simple,
    sigma_ATM, sigma_RR, sigma_SQ,
    delta_tilde=delta_tilde,
    K_ATM_convention=K_ATM_conv,
    delta_convention=delta_conv
)
    # Display results
    st.text(f"MID Forward Parity: {np.round(params.f, 4)}")
    st.text("")
    st.text(f"ATM Strike Convention: {K_ATM_conv}")
    st.text(f"Delta convention: {delta_conv}")
    st.text(f"@{K:.3f} {call_put} results :")
    st.table(df)
