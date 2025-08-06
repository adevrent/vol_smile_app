import streamlit as st
import os, sys

import QuantLib as ql
from FX_option_pricer import calc_tx_with_spreads
import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt

st.title("SPI Vol Smile Calculator, by Atakan Devrent")

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
eval_date_str     = st.text_input("Eval Date (YYYY-MM-DD)", placeholder="e.g. 2025-07-08")
expiry_date_str   = st.text_input("Expiry Date (YYYY-MM-DD)", placeholder="e.g. 2025-07-23")
delivery_date_str = st.text_input("Delivery Date (YYYY-MM-DD)", placeholder="e.g. 2025-07-24")

def parse_ql_date(s: str) -> ql.Date:
    try:
        y, m, d = map(int, s.split("-"))
        return ql.Date(d, m, y)
    except Exception:
        st.error(f"Invalid date format: {s}")
        st.stop()

# 3) Numeric inputs as text
x_str         = st.text_input("Spot Price", placeholder="e.g. 39.729")
rd_str        = st.text_input("Simple Domestic Rate (%)", placeholder="e.g. 45.994")
rf_str        = st.text_input("Simple Foreign Rate (%)", placeholder="e.g. 4.320")
sigma_ATM_str = st.text_input("ATM VOL (%)", placeholder="e.g. 12.00")
sigma_RR_str  = st.text_input("RR VOL (%)", placeholder="e.g. 13.00")
sigma_SQ_str  = st.text_input("BF VOL (%)", placeholder="e.g. 1.75")

# 4) Pillar delta
pillar_choices = [0.10, 0.25]
delta_tilde   = st.selectbox("Pillar Delta", pillar_choices, index=0)

# 5) Conventions
conventions_map = {"Convention A":["fwd_delta_neutral", "spot_pa"], "Convention B":["fwd", "spot"]}
convention_label = st.selectbox("Convention", list(conventions_map.keys()), index=0)
conv = conventions_map[convention_label]

# Spreads
rd_spread_str = st.text_input("Domestic Rate spread (%)", placeholder="e.g. 0.0")
rf_spread_str  = st.text_input("Foreign Rate spread (%)", placeholder="e.g. 0.0")
ATM_vol_spread_str  = st.text_input("ATM vol spread (%)", placeholder="e.g. 2.25")


K_ATM_convention, delta_convention = conv

# 6) Strike & Call/Put & buy/sell
K_str = st.text_input("Strike Price", placeholder="e.g. 41.00")
call_put = st.selectbox("CALL/PUT", ["CALL","PUT"], index=0)
buy_sell = st.selectbox("BUY/SELL", ["BUY","SELL"], index=0)

# 7) Default values
spot_bd = 1
calendar = ql.Turkey()

# 8) Compute
if st.button("Compute"):
    # parse dates
    eval_date     = parse_ql_date(eval_date_str)
    expiry_date   = parse_ql_date(expiry_date_str)
    delivery_date = parse_ql_date(delivery_date_str)

    # parse numbers
    try:
        x = float(x_str)
        rd_simple = float(rd_str) / 100
        rf_simple = float(rf_str) / 100
        sigma_ATM = float(sigma_ATM_str) / 100
        sigma_RR  = float(sigma_RR_str)  / 100
        sigma_SQ  = float(sigma_SQ_str)  / 100
        K = float(K_str)
        rd_spread = float(rd_spread_str) / 100
        rf_spread = float(rf_spread_str) / 100
        ATM_vol_spread = float(ATM_vol_spread_str) / 100

    except Exception as e:
        st.error(f"Numeric input error: {e}")
        st.stop()

    df, mid_params = calc_tx_with_spreads(
        buy_sell, call_put, K, rd_spread, rf_spread, ATM_vol_spread,
        calendar, basis_dict, spot_bd, eval_date, expiry_date,
        delivery_date, x, rd_simple, rf_simple, sigma_ATM, sigma_RR,
        sigma_SQ, delta_tilde=delta_tilde, K_ATM_convention=K_ATM_convention,
        delta_convention=delta_convention)

    # display
    st.text(f"Forward Parity: {np.round(mid_params.f, 4)}")
    st.text("")
    st.text(f"ATM Strike Convention: {K_ATM_convention}")
    st.text(f"Delta convention: {delta_convention}")
    st.text(f"@{K:.3f} {buy_sell} {call_put} results :")
    st.table(df)
