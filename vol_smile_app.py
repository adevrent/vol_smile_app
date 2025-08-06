import streamlit as st
import QuantLib as ql
import numpy as np
import pandas as pd
import os, sys
import importlib.util

# 0) Ensure our modules are on PYTHONPATH
app_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(app_dir, os.pardir))
cwd = os.getcwd()
for p in (app_dir, parent_dir, cwd):
    if p not in sys.path:
        sys.path.insert(0, p)

# 1) Import FX_Option_Pricer with fallback from app or parent
try:
    from FX_Option_Pricer import OptionParams, calc_tx_with_spreads
except ImportError:
    spec_paths = [os.path.join(app_dir, "FX_Option_Pricer.py"), os.path.join(parent_dir, "FX_Option_Pricer.py")]
    fx_mod = None
    for spec_path in spec_paths:
        if os.path.exists(spec_path):
            spec = importlib.util.spec_from_file_location("FX_Option_Pricer", spec_path)
            fx_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fx_mod)
            break
    if not fx_mod:
        st.error("FX_Option_Pricer.py not found in app or parent directory.")
        st.stop()
    OptionParams = fx_mod.OptionParams
    calc_tx_with_spreads = fx_mod.calc_tx_with_spreads

# 2) Title
st.title("SPI Vol Smile & TX Calculator")

# 3) Day-count bases
basis_map = {"ACTL360": ql.Actual360(), "ACTL365": ql.Actual365Fixed()}
col1, col2 = st.columns(2)
with col1:
    basis_for = st.selectbox("FOR Day-count Basis", list(basis_map.keys()), index=0)
with col2:
    basis_dom = st.selectbox("DOM Day-count Basis", list(basis_map.keys()), index=0)
basis_dict = {"FOR": basis_map[basis_for], "DOM": basis_map[basis_dom]}

# 4) Dates
eval_date_str     = st.text_input("Eval Date (YYYY-MM-DD)", placeholder="2025-07-08")
expiry_date_str   = st.text_input("Expiry Date (YYYY-MM-DD)", placeholder="2025-07-23")
delivery_date_str = st.text_input("Delivery Date (YYYY-MM-DD)", placeholder="2025-07-24")

def parse_ql_date(s: str) -> ql.Date:
    try:
        y, m, d = map(int, s.split("-"))
        return ql.Date(d, m, y)
    except:
        st.error(f"Invalid date format: {s}")
        st.stop()

# 5) Market inputs
col1, col2, col3 = st.columns(3)
with col1:
    x_str = st.text_input("Spot Price", placeholder="e.g. 39.729")
with col2:
    rd_str = st.text_input("Domestic Rate Spread (%)", placeholder="e.g. 45")
with col3:
    rf_str = st.text_input("Foreign Rate Spread (%)", placeholder="e.g. 5")
col4, col5 = st.columns(2)
with col4:
    atm_spread_str = st.text_input("ATM vol Spread (%)", placeholder="e.g. 2.25")
with col5:
    sigma_atm_str = st.text_input("ATM VOL (%)", placeholder="e.g. 12.00")
sigma_rr_str = st.text_input("RR VOL (%)", placeholder="e.g. 13.00")
sigma_sq_str = st.text_input("BF VOL (%)", placeholder="e.g. 1.75")

# 6) Pillar delta
delta_tilde = st.selectbox("Pillar Delta", [0.10, 0.25], index=0)

# 7) BUY/SELL & Convention
buy_sell = st.selectbox("BUY/SELL", ["BUY", "SELL"], index=0)
convention = st.selectbox("Convention", ["Convention A", "Convention B"], index=0)
if convention == "Convention A":
    K_ATM_conv, delta_conv = "fwd_delta_neutral", "spot_pa"
else:
    K_ATM_conv, delta_conv = "fwd", "spot"

# 8) Strike & Call/Put
K_str = st.text_input("Strike Price", placeholder="e.g. 41.00")
call_put = st.selectbox("CALL/PUT", ["CALL","PUT"], index=1)

# 9) Compute
if st.button("Compute"):
    # parse
    try:
        eval_date, expiry_date, delivery_date = parse_ql_date(eval_date_str), parse_ql_date(expiry_date_str), parse_ql_date(delivery_date_str)
        x = float(x_str)
        rd_spread, rf_spread = float(rd_str)/100, float(rf_str)/100
        atm_vol_spread = float(atm_spread_str)/100
        sigma_atm, sigma_rr, sigma_sq = float(sigma_atm_str)/100, float(sigma_rr_str)/100, float(sigma_sq_str)/100
        K = float(K_str)
    except Exception as e:
        st.error(f"Numeric input error: {e}")
        st.stop()
    rd_simple, rf_simple = rd_spread, rf_spread
    # init params & calibrate
    params = OptionParams(
        calendar=ql.Turkey(), basis_dict=basis_dict, spot_bd=1,
        eval_date=eval_date, expiry_date=expiry_date, delivery_date=delivery_date,
        x=x, rd_simple=rd_simple, rf_simple=rf_simple,
        sigma_ATM=sigma_atm, sigma_RR=sigma_rr, sigma_SQ=sigma_sq,
        delta_tilde=delta_tilde,
        K_ATM_convention=K_ATM_conv, delta_convention=delta_conv
    )
    params.optimize_sigma_S()
    params.set_K_C_P()
    # calc transaction
df = calc_tx_with_spreads(
    buy_sell, call_put, K,
    rd_spread, rf_spread, atm_vol_spread,
    ql.Turkey(), basis_dict, 1,
    eval_date, expiry_date, delivery_date,
    x, rd_simple, rf_simple,
    sigma_atm, sigma_rr, sigma_sq,
    delta_tilde=delta_tilde,
    K_ATM_convention=K_ATM_conv,
    delta_convention=delta_conv
)

# display
st.text(f"MID Forward Parity: {np.round(params.f, 4)}")
st.text("")
st.text(f"ATM Strike Convention: {K_ATM_conv}")
st.text(f"Delta convention: {delta_conv}")
st.text(f"@{K:.3f} {call_put} results :")
st.table(df)
