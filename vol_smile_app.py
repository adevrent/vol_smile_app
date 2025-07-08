# vol_smile_app.py
import streamlit as st
import QuantLib as ql
from FX_option_pricer import OptionParams

st.title("Volatility Smile Calculator")

# ── 1) Day‑count bases ──────────────────────────────────────────────────────
basis_map = {
    "ACTL360": ql.Actual360(),
    "ACTL365": ql.Actual365Fixed(),
}
col1, col2 = st.columns(2)
with col1:
    basis_for_str = st.selectbox("FOR Day‑count Basis", list(basis_map.keys()), index=0)
with col2:
    basis_dom_str = st.selectbox("DOM Day‑count Basis", list(basis_map.keys()), index=0)
basis_dict = {"FOR": basis_map[basis_for_str], "DOM": basis_map[basis_dom_str]}

# ── 2) Dates as text inputs ─────────────────────────────────────────────────
eval_date_str     = st.text_input("Eval Date (YYYY-MM-DD)", placeholder="2025-07-08")
expiry_date_str   = st.text_input("Expiry Date (YYYY-MM-DD)", placeholder="2025-07-23")
delivery_date_str = st.text_input("Delivery Date (YYYY-MM-DD)", placeholder="2025-07-24")

def parse_ql_date(s: str) -> ql.Date:
    try:
        y, m, d = map(int, s.split("-"))
        return ql.Date(d, m, y)
    except Exception:
        st.error(f"Invalid date format: {s}")
        return None

# ── 3) Numeric inputs as text inputs ─────────────────────────────────────────
x_str         = st.text_input("Spot Price", placeholder="e.g. 39.729")
rd_simple_str = st.text_input("Simple Domestic Rate (%)", placeholder="e.g. 45.994")
rf_simple_str = st.text_input("Simple Foreign Rate (%)", placeholder="e.g. 4.320")
sigma_ATM_str = st.text_input("ATM VOL (%)", placeholder="e.g. 12.00")
sigma_RR_str  = st.text_input("RR VOL (%)", placeholder="e.g. 13.00")
sigma_SQ_str  = st.text_input("BF VOL (%)", placeholder="e.g. 1.75")

# ── 4) Pillar delta ─────────────────────────────────────────────────────────
pillar_delta_choices = [0.10, 0.25]
delta_tilde = st.selectbox("Pillar Delta", pillar_delta_choices, index=0)

# ── 5) Conventions ──────────────────────────────────────────────────────────
atm_conv_map = {
    "Forward": "fwd",
    "Forward Delta Neutral": "fwd_delta_neutral",
    "Spot": "spot"
}
K_ATM_label      = st.selectbox("ATM Strike Convention", list(atm_conv_map.keys()), index=0)
K_ATM_convention = atm_conv_map[K_ATM_label]

delta_conv_map   = {
    "Spot": "spot",
    "Spot Premium Adjusted": "spot_pa",
    "Forward": "fwd"
}
delta_label      = st.selectbox("Delta Convention", list(delta_conv_map.keys()), index=0)
delta_convention = delta_conv_map[delta_label]

# ── 6) Strike & Call/Put ───────────────────────────────────────────────────
K_str     = st.text_input("Strike Price", placeholder="e.g. 41.00")
call_put  = st.selectbox("CALL/PUT", ["CALL", "PUT"], index=0)

# ── 7) Compute button ───────────────────────────────────────────────────────
if st.button("Compute"):
    # Parse dates
    eval_date     = parse_ql_date(eval_date_str)
    expiry_date   = parse_ql_date(expiry_date_str)
    delivery_date = parse_ql_date(delivery_date_str)

    # Parse numeric inputs
    try:
        x         = float(x_str)
        rd_simple = float(rd_simple_str) / 100
        rf_simple = float(rf_simple_str) / 100
        sigma_ATM = float(sigma_ATM_str) / 100
        sigma_RR  = float(sigma_RR_str) / 100
        sigma_SQ  = float(sigma_SQ_str) / 100
        K         = float(K_str)
    except ValueError as e:
        st.error(f"Numeric input error: {e}")
        st.stop()

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
        K_ATM_convention=K_ATM_convention,
        delta_convention=delta_convention
    )

    # Run your full calibration / smile construction
    params.optimize_smile()
    params.set_K_C_P()

    # Compute for the chosen strike & option type
    iv        = params.implied_volatility(call_put, K)
    dom_price = params.BS(call_put, K, iv)["v_dom"]
    for_price= params.BS(call_put, K, iv)["v_for"]
    delta_v   = params.delta(call_put, K, iv)
    gamma_v   = params.gamma(call_put, K, iv)

    st.subheader("Results")
    st.write(f"• **Implied Vol @ K={K:.4f}:** {iv*100:.2f} %")
    st.write(f"• **{call_put} Price (dom):** {dom_price:.6f}")
    st.write(f"• **{call_put} Price (for):** {for_price:.6f}")
    st.write(f"• **Delta:** {delta_v:.6f}    • **Gamma:** {gamma_v:.6f}")
