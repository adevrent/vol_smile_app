from FX_option_pricer import calc_tx_with_spreads
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# DEBUG
buy_sell = "BUY"
call_put = "CALL"
K = 45.0

rd_spread = 0.0 / 100
rf_spread = 0.0 / 100
ATM_vol_spread = 3 / 100

calendar = ql.Turkey()
basis_dict = {"FOR": ql.Actual360(), "DOM": ql.Actual365Fixed()}
spot_bd = 1
eval_date = ql.Date(7, 10, 2025)
days_arr = np.array([30, 60, 90, 180, 270, 360])

x = 41.703
rd_simple = 39.797 / 100
rf_simple = 4.091 / 100

RR_ATM_ratios = np.array([0.8, 1.0, 1.2, 1.4])
sigma_RR_arr = np.arange(12, 26, 2) / 100  # Quoted Risk Reversal volatilities

sigma_SQ = 1.5 / 100  # Quoted Strangle volatility
convention = "Convention A"

if convention == "Convention B":
    K_ATM_convention = "fwd"
    delta_convention = "spot"
elif convention == "Convention A":
    K_ATM_convention = "fwd_delta_neutral"
    delta_convention = "spot_pa"

delta_tilde = 0.25  # pillar smile delta, e.g. 0.25 or 0.10

flag_tensor = np.empty(shape=(len(days_arr), len(sigma_RR_arr), len(RR_ATM_ratios)))

for i, num_days in enumerate(days_arr):
    expiry_date = eval_date + num_days
    delivery_date = expiry_date + 1
    for j, sigma_RR in enumerate(sigma_RR_arr):
        for k, ratio in enumerate(RR_ATM_ratios):
            sigma_ATM = sigma_RR / ratio
            try:
                df, mid_params = calc_tx_with_spreads(
                    buy_sell, call_put, K, rd_spread, rf_spread, ATM_vol_spread,
                    calendar, basis_dict, spot_bd, eval_date, expiry_date,
                    delivery_date, x, rd_simple, rf_simple, sigma_ATM, sigma_RR,
                    sigma_SQ, delta_tilde=delta_tilde, K_ATM_convention=K_ATM_convention,
                    delta_convention=delta_convention)

                flag_tensor[i, j, k] = 1
                print(f"Success for days={num_days}, sigma_RR={sigma_RR*100:.2f}%, ratio={ratio:.2f}")
                print("_"*50)
            except ValueError as e:
                flag_tensor[i, j, k] = 0
                print(f"Error: {e}")
                print(f"Failed for days={num_days}, sigma_RR={sigma_RR*100:.2f}%, ratio={ratio:.2f}")
                print("_"*50)
flags_i = (flag_tensor == 1).sum()
flags_0 = (flag_tensor == 0).sum()
print(f"Success (1): {flags_i} | Fail (0): {flags_0}")

# Create a MultiIndex from all combinations of the dimensions
index = pd.MultiIndex.from_product(
    [days_arr, sigma_RR_arr, RR_ATM_ratios],
    names=["MaturityDays", "RiskReversalVol", "RR_ATM_Ratio"]
)

# Flatten the 3D tensor into 1D (row-wise)
values = flag_tensor.flatten().astype(int)

# Build the DataFrame
df_flags = pd.DataFrame(values, index=index, columns=["Flag"])

print(df_flags)

df_flags.reset_index().to_excel("feasibility_flags_flat.xlsx", index=False)

X, Y, Z = np.meshgrid(days_arr, sigma_RR_arr, RR_ATM_ratios, indexing="ij")
x_vals, y_vals, z_vals = X.ravel(), Y.ravel(), Z.ravel()
flags = flag_tensor.ravel().astype(int)  # also helps downstream

cmap = ListedColormap(["red", "green"])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type('ortho')          # reduce perspective occlusion
ax.view_init(elev=20, azim=135)    # pick a clearer angle

# plot fails first (0), then successes (1)
mask0 = flags == 0
mask1 = flags == 1

ax.scatter(x_vals[mask0], y_vals[mask0], z_vals[mask0],
           c=flags[mask0], cmap=cmap, vmin=0, vmax=1,
           s=70, alpha=0.9, depthshade=False, edgecolors="k", linewidths=0.4)

ax.scatter(x_vals[mask1], y_vals[mask1], z_vals[mask1],
           c=flags[mask1], cmap=cmap, vmin=0, vmax=1,
           s=90, alpha=1.0, depthshade=False, edgecolors="k", linewidths=0.4)

ax.set_xlabel("Maturity Days")
ax.set_ylabel("Risk Reversal Vol")
ax.set_zlabel("RR/ATM Ratio")
ax.set_title("Feasibility Tensor (Green = Works, Red = Fails)")
plt.show()
