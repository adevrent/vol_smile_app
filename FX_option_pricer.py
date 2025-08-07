import numpy as np
import pandas as pd
import scipy.stats as ss
import QuantLib as ql
import datetime as dt
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from utils import convert_datetype, convert_simple_to_ccomp

valid_K_ATM_conventions = ["fwd", "fwd_delta_neutral", "spot"]
valid_delta_conventions = ["spot", "spot_pa", "fwd"]
valid_conventions = ["Convention A", "Convention B"]


class OptionParams:
    def __init__(self, calendar, basis_dict, spot_bd, eval_date, expiry_date, delivery_date, x, rd_simple, rf_simple, sigma_ATM, sigma_RR, sigma_SQ, delta_tilde=0.25, K_ATM_convention="fwd", delta_convention="fwd_pa"):
        """call_put, K, sigma missing on purpose, to be set later"""
        self.calendar = calendar
        self.basis_dict = basis_dict
        self.spot_bd = spot_bd

        self.eval_date = eval_date  # QL type
        self.expiry_date = expiry_date  # QL type
        self.delivery_date = delivery_date  # QL type
        self.x = x

        self.delta_convention = delta_convention  # "spot", "spot_pa", "fwd", "fwd_pa"

        # ##### RESTRICTION 1
        self.sigma_ATM = sigma_ATM  # ATM volatility
        """
        sigma(K_ATM) = self.sigma_ATM

        should be satisfied by the calibrated smile volatility function sigma(.)
        """

        # ##### RESTRICTION 2
        self.sigma_RR = sigma_RR  # Risk Reversal volatility
        """
        sigma(K_C) - sigma(K_P) = self.sigma_RR

        should be satisfied by the calibrated smile volatility function sigma(.)
        where K_C is the delta_tilde call strike and K_P is the delta_tilde put strike.
        However, at the beginning we do not know the values of K_C and K_P, so we will calibrate them later.
        """

        self.sigma_SQ = sigma_SQ  # Quoted Strangle volatility
        self.sigma_SM = (sigma_ATM + sigma_SQ)  # Market Strangle volatility

        self.eval_spot_date = calendar.advance(eval_date, ql.Period(spot_bd, ql.Days))
        self.for_basis = self.basis_dict["FOR"]
        self.dom_basis = self.basis_dict["DOM"]
        self.tau = self.dom_basis.yearFraction(eval_date, expiry_date)
        self.tau_for = self.for_basis.yearFraction(eval_date, delivery_date)
        self.tau_dom = self.dom_basis.yearFraction(eval_date, delivery_date)
        self.tau_spot_for = self.for_basis.yearFraction(self.eval_spot_date, self.delivery_date)
        self.tau_spot_dom = self.dom_basis.yearFraction(self.eval_spot_date, self.delivery_date)

        # New tau values
        self.tau_365 = ql.Actual365Fixed().yearFraction(self.eval_date, self.expiry_date)
        self.tau_360 = ql.Actual360().yearFraction(self.eval_date, self.expiry_date)
        self.tau_spot_360 = ql.Actual360().yearFraction(self.eval_spot_date, self.delivery_date)
        self.tau_spot_365 = ql.Actual365Fixed().yearFraction(self.eval_spot_date, self.delivery_date)

        self.rd = convert_simple_to_ccomp(rd_simple, self.tau_spot_360)
        self.rf = convert_simple_to_ccomp(rf_simple, self.tau_spot_360)

        self.f = self.x * np.exp((self.rd - self.rf) * self.tau_spot_360)

        if K_ATM_convention.lower() not in valid_K_ATM_conventions:
            raise ValueError(f"Invalid K_ATM_convention: {K_ATM_convention}. Must be one of {valid_K_ATM_conventions}.")
        if K_ATM_convention.lower() == "fwd":
            self.K_ATM = self.f
        elif K_ATM_convention.lower() == "fwd_delta_neutral":
            if delta_convention.lower() == "spot":
                self.K_ATM = self.f * np.exp(0.5 * self.sigma_ATM**2 * self.tau_365)
            elif delta_convention.lower() == "spot_pa":
                self.K_ATM = self.f * np.exp(-0.5 * self.sigma_ATM**2 * self.tau_365)
        elif K_ATM_convention.lower() == "spot":
            self.K_ATM = self.x

        self.delta_ATM = self.BS("CALL", self.K_ATM, self.sigma_ATM)["delta_S"]  # spot delta of ATM call option, to be used in strangle calculation

        # SPI params
        self.delta_tilde = delta_tilde  # delta_tilde is the pillar smile delta, e.g. 0.25 or 0.10
        print("...." * 50)
        print("Calculating K_CSM")
        self.K_CSM = self.calc_strike("CALL", self.sigma_SM, self.delta_tilde)  # Call strike at delta pillar with MARKET STRANGLE VOL !
        print("K_CSM:", self.K_CSM)
        print("K_CSM pa delta:", np.round(self.BS("CALL", self.K_CSM, self.sigma_SM)["delta_S_pa"] * 100, 4))
        print()
        print("Calculating K_PSM")

        self.K_PSM = self.calc_strike("PUT", self.sigma_SM, -self.delta_tilde)  # Put strike at delta pillar with MARKET STRANGLE VOL !
        print("K_PSM:", self.K_PSM)
        print("K_PSM pa delta:", np.round(self.BS("PUT", self.K_PSM, self.sigma_SM)["delta_S_pa"] * 100, 4))
        print("...." * 50)

        # ##### RESTRICTION 3
        self.v_SM = self.BS("CALL", self.K_CSM, self.sigma_SM)["v_dom"] + self.BS("PUT", self.K_PSM, self.sigma_SM)["v_dom"]  # Market Strangle value in domestic currency with MARKET STRANGLE VOL !
        """
        v(K_CSM, sigma(K_CSM), CALL) + v(K_PSM, sigma(K_PSM), PUT) = self.v_SM

        should be satisfied by the calibrated smile volatility function sigma(.)
        Deltas of these options will not yield the delta_tilde, because vol is changed from sigma_SM to sigma(.). This is okay.
        """

        if delta_convention.lower() not in valid_delta_conventions:
            raise ValueError(f"Invalid delta_convention: {delta_convention}. Must be one of {valid_delta_conventions}.")

        if delta_convention.lower() == "spot":
            self.a = np.exp(-self.rf * self.tau_360)  # equal to foreign DF from put-call delta parity
            #                                       because we use spot delta
        elif delta_convention.lower() == "fwd":
            self.a = 1
        elif delta_convention.lower() == "spot_pa":
            self.a = None  # to be calibrated later inside `optimize_sigma_S` method

        self.K_C = None  # K_C is the delta_tilde call strike, found via optimization.
        self.K_P = None  # K_P is the delta_tilde put strike, found via optimization.

        self.sigma_S = self.sigma_SQ  # self.sigma_S is the smile strangle volatility, found via optimization.
        #                      self.sigma_S = (sigma(K_C) + sigma(K_P)) / 2 - self.sigma_ATM

    def simple_call_put(self, K):
        if K >= self.K_ATM:
            return "CALL"
        else:
            return "PUT"

    def calc_strike(self, call_put, sigma, delta, eps=1e-6, max_iter=10000):
        if call_put.lower() == "call":
            phi = 1
        else:
            phi = -1
        theta_plus = (self.rd - self.rf)/sigma + sigma/2
        if self.delta_convention.lower() == "spot":
            delta_S = delta
            K = self.x * np.exp(-phi * ss.norm.ppf(phi * np.exp(self.rf*self.tau_360) * delta_S) * sigma * np.sqrt(self.tau_365) + sigma * theta_plus * self.tau_365)
            return K
        elif self.delta_convention.lower() == "fwd":
            delta_S = np.exp(-self.rf * self.tau_360) * delta
            K = self.x * np.exp(-phi * ss.norm.ppf(phi * np.exp(self.rf*self.tau_360) * delta_S) * sigma * np.sqrt(self.tau_365) + sigma * theta_plus * self.tau_365)
            return K
        elif self.delta_convention.lower() == "spot_pa":
            # print("Inside calc_strike with spot_pa delta convention")
            delta_S = delta
            """using delta_S_pa as vanilla delta_S to calculate K_max
            because premium-adjusted delta for a strike K is always
            SMALLER than the non-adjusted delta corresponding to
            the same strike"""
            K_npa = self.x * np.exp(-phi * ss.norm.ppf(phi * np.exp(self.rf*self.tau_360) * delta_S) * sigma * np.sqrt(self.tau_365) + sigma * theta_plus * self.tau_365)
            # print("K_npa:", K_npa)
            K_max = K_npa
            K_min = self.solve_K_min(sigma, K_max, eps=eps, max_iter=1000)

            # print("K_min:", K_min)
            # print("K_max:", K_max)

            def f(K):
                print()
                # print("####### Inside calc_strike objective function #######")
                d2 = self.calc_d2(K, sigma)
                delta_S_pa = phi * np.exp(-self.rf * self.tau_360) * K/self.f * ss.norm.cdf(phi * d2)
                # print("delta_S_pa:", delta_S_pa)
                # print("calc strike objective: %", np.round(delta_S_pa - delta, 6))
                # print("K =", K)
                # print("#######################################################")
                # print()
                return delta_S_pa - delta

            # Adaptive bracket expansion
            max_expansions = 10
            expansions = 0

            while expansions < max_expansions:
                try:
                    f_min = f(K_min)
                    f_max = f(K_max)

                    # print(f"Bracket: [{K_min:.6f}, {K_max:.6f}]")
                    # print(f"f(min)={f_min:.6f}, f(max)={f_max:.6f}")

                    if np.sign(f_min) != np.sign(f_max):
                        res = root_scalar(f, method='brentq', bracket=[K_min, K_max], xtol=eps, maxiter=100)
                        K = res.root
                        print(f"Found K={K:.6f} after {expansions} expansions")
                        return K
                except Exception as e:
                    print(f"Root finding failed: {e}")

                # Expand bracket for K_min
                K_min *= 0.8  # Lower min for calls

                expansions += 1
                # print(f"Expanded bracket to [{K_min:.6f}, {K_max:.6f}]")

            # Final attempt with wider bounds
            try:
                res = root_scalar(f, method='brentq', bracket=[0.1*self.f, 10*self.f], xtol=eps, maxiter=max_iter)
                return res.root
            except:
                # Fallback to non-premium-adjusted strike
                return K_npa

    def calc_d1(self, K, sigma):
        """
        Calculate d1 for Black-Scholes formula.

        Args:
            K (float): strike price
            sigma (float): volatility of the option, in 0.33 format.
        """
        d1 = (np.log(self.f/K) + 0.5*(sigma**2)*self.tau_365) / (sigma*np.sqrt(self.tau_365))
        return d1

    def calc_d2(self, K, sigma):
        """
        Calculate d2 for Black-Scholes formula.

        Args:
            K (float): strike price
            sigma (float): volatility of the option, in 0.33 format.
        """
        d2 = (np.log(self.f/K) - 0.5*(sigma**2)*self.tau_365) / (sigma*np.sqrt(self.tau_365))
        return d2

    def solve_K_min(self, sigma, K_max, eps=1e-6, max_iter=1000):
        def f(K_min):
            d2 = self.calc_d2(K_min, sigma)
            obj = sigma * np.sqrt(self.tau_365) * ss.norm.cdf(d2) - ss.norm.pdf(d2)
            return obj

        try:
            # find root
            res = root_scalar(f,
                              method='brentq',
                              bracket=[0.001, K_max],
                              xtol=eps,
                              maxiter=max_iter)
            K_min = res.root
            if K_min is np.nan or K_min <= 0:
                raise ValueError("K_min is not a valid positive number.")

            # print(f"K_min found: {K_min}")
            return K_min

        except ValueError:
            # Fallback to a reasonable lower bound
            return 0.001 * self.f

    def BS(self, call_put, K, sigma):
        """
        Args:
            call_put (str): "CALL" or "PUT"
            K (float): strike price
            sigma (float): volatility of the option, in 0.33 format.
        """
        if call_put.lower() == "call":
            phi = 1
        else:
            phi = -1

        d1 = self.calc_d1(K, sigma)
        d2 = self.calc_d2(K, sigma)

        v_dom = phi * np.exp(-self.rd * self.tau_360) * (self.f * ss.norm.cdf(phi * d1) - K * ss.norm.cdf(phi * d2))
        v_for = v_dom/self.x

        delta_S = phi * np.exp(-self.rf * self.tau_360) * ss.norm.cdf(phi * d1)
        delta_S_pa = delta_S - v_for
        delta_dual = -phi * np.exp(-self.rd * self.tau_360) * ss.norm.cdf(phi * d2)
        delta_fwd = phi * ss.norm.cdf(phi * d1)
        delta_fwd_pa = phi * K/self.f * ss.norm.cdf(phi * d2)

        return {"v_dom": v_dom,
                "v_for": v_for,
                "delta_S": delta_S,
                "delta_S_pa": delta_S_pa,
                "delta_dual": delta_dual,
                "delta_fwd": delta_fwd,
                "delta_fwd_pa": delta_fwd_pa}

    def get_vol_from_price(self, v_dom, K, call_put, eps=1e-9, max_iter=10000):
        """
        Find the implied volatility for a given price using the Black-Scholes model.
        """

        def f(sigma):
            return self.BS(call_put, K, sigma)["v_dom"] - v_dom

        a, b = 1e-6, 2.0

        res = root_scalar(f, method="brentq", bracket=[a, b], xtol=eps, maxiter=max_iter)
        return res.root


    def calc_c1(self, sigma_S, a=None):
        """
        Args:
            sigma_S (float): Smile Strangle volatility, in 0.33 format. Iteratively calibrated.
        """
        if (a is None) and self.delta_convention.lower() in ["spot", "fwd"]:
            a = self.a

        numerator = a**2 * (2*sigma_S+self.sigma_RR) - 2*a*(2*sigma_S+self.sigma_RR)*(self.delta_tilde + self.delta_ATM) + 2*(self.delta_tilde**2 * self.sigma_RR + 4*sigma_S*self.delta_tilde*self.delta_ATM + self.sigma_RR * self.delta_ATM**2)
        denominator = 2 * (2*self.delta_tilde - a) * (self.delta_tilde - self.delta_ATM) * (self.delta_tilde - a + self.delta_ATM)
        c1 = numerator / denominator
        return c1

    def calc_c2(self, sigma_S, a=None):
        """
        Args:
            sigma_S (float): Smile Strangle volatility, in 0.33 format. Iteratively calibrated.
        """
        if (a is None) and self.delta_convention.lower() in ["spot", "fwd"]:
            a = self.a

        numerator = 4*self.delta_tilde*sigma_S - a * (2*sigma_S + self.sigma_RR) + 2*self.sigma_RR*self.delta_ATM
        denominator = 2 * (2*self.delta_tilde - a) * (self.delta_tilde - self.delta_ATM) * (self.delta_tilde - a + self.delta_ATM)
        c2 = numerator / denominator
        return c2

    def calc_sigma_from_delta(self, delta_call, sigma_S, a=None):
        """
        Args:
            delta (float): Delta of the implied volatility to be calculated, in 0.33 format.
            This delta is ALWAYS the call delta, even if the option is a put.
            sigma_S (float): Smile Strangle volatility, in 0.33 format. Iteratively calibrated.
        """
        if a is None:
            a = self.a
        c1 = self.calc_c1(sigma_S, a)
        c2 = self.calc_c2(sigma_S, a)
        sigma = self.sigma_ATM + c1*(delta_call - self.delta_ATM) + c2*(delta_call - self.delta_ATM)**2
        return sigma

    def optimize_sigma_S(self, eps=1e-9, max_iter=10000):
        """
        Calibrate sigma_S (smile-strangle vol) so that the
        SPI strangle price matches the market strangle value.

        Args:
            eps (float): Tolerance for the optimization, default is 1e-9.
            max_iter (int): Maximum number of iterations, default is 10000.
        """
        if self.delta_convention in ["spot", "fwd"]:
            def f(sigma_S):
                sigma_CSM = self.find_SPI_sigma_K("CALL", self.K_CSM, sigma_S)
                sigma_PSM = self.find_SPI_sigma_K("PUT", self.K_PSM, sigma_S)

                v_call = self.BS("CALL", self.K_CSM, sigma_CSM)["v_dom"]
                v_put = self.BS("PUT",  self.K_PSM, sigma_PSM)["v_dom"]
                print("    sigma_S optimization objective: %", np.round((v_call + v_put) - self.v_SM, 6)*100)
                return (v_call + v_put) - self.v_SM
        elif self.delta_convention == "spot_pa":
            if self.a is None:
                print("||||| Initial K_P optimization: |||||")
                K_P = self.calc_strike("PUT", self.sigma_SM, -self.delta_tilde)  # Initial value
                self.a = np.exp(-self.rf * self.tau_360) * K_P/self.f
                print("        -- Initial K_P =", K_P)

            def f(sigma_S):
                sigma_P = np.maximum(self.sigma_ATM - 0.5*self.sigma_RR + sigma_S, 1e-3)  # Ensure sigma_P is positive
                print("        -- sigma_P = %", sigma_P*100)
                K_P = self.calc_strike("PUT", sigma_P, -self.delta_tilde)  # Update K_P based on sigma_P
                print("        -- K_P =", K_P)
                self.a = np.exp(-self.rf * self.tau_360) * K_P/self.f

                sigma_CSM = self.find_SPI_sigma_K("CALL", self.K_CSM, sigma_S)
                sigma_PSM = self.find_SPI_sigma_K("PUT", self.K_PSM, sigma_S)

                v_call = self.BS("CALL", self.K_CSM, sigma_CSM)["v_dom"]
                v_put = self.BS("PUT",  self.K_PSM, sigma_PSM)["v_dom"]
                print("    sigma_S optimization objective: %", np.round((v_call + v_put) - self.v_SM, 6)*100)
                return (v_call + v_put) - self.v_SM

        lowest_vol_sum = np.minimum(self.sigma_ATM + self.sigma_RR/2, self.sigma_ATM - self.sigma_RR/2)
        # find root
        res = root_scalar(
            f,
            method='brentq',  #TODO
            bracket=[-lowest_vol_sum + 1e-3, self.sigma_SQ + 0.1],
            xtol=eps,
            x0=self.sigma_SQ,  # initial guess
            maxiter=max_iter
        )
        sigma_S_opt = res.root
        print(f"sigma_S_opt: %{sigma_S_opt*100:.2f}")
        self.sigma_S = sigma_S_opt  # update the instance variable

        print("v_SM calc: %", np.round(self.BS("CALL", self.K_CSM, self.find_SPI_sigma_K("CALL", self.K_CSM))["v_dom"] + self.BS("PUT", self.K_PSM, self.find_SPI_sigma_K("PUT", self.K_PSM))["v_dom"], 4)*100)

        return sigma_S_opt

    def find_SPI_sigma_K(self, call_put, K, sigma_S=None, eps=1e-6, max_iter=10000):
        """
        Find the implied volatility for a given strike K using the SPI model.

        Args:
            K (float): Strike price for which to find the implied volatility.
        """
        if not sigma_S:
            sigma_S = self.sigma_S

        # Define the objective function
        if self.delta_convention.lower() == "spot":
            delta_type = "delta_S"
            a = self.a
            c1 = self.calc_c1(sigma_S, a)
            c2 = self.calc_c2(sigma_S, a)
        elif self.delta_convention.lower() == "fwd":
            delta_type = "delta_fwd"
            a = self.a
            c1 = self.calc_c1(sigma_S, a)
            c2 = self.calc_c2(sigma_S, a)
        elif self.delta_convention.lower() == "spot_pa":
            delta_type = "delta_S_pa"
            a = np.exp(-self.rf * self.tau_360) * K/self.f
            c1 = self.calc_c1(sigma_S, a)
            c2 = self.calc_c2(sigma_S, a)
        else:
            raise NotImplementedError(f"Delta convention {self.delta_convention} not implemented.")

        def f(sigma):
            if call_put.lower() == "put":
                delta = self.BS("PUT", K, sigma)[delta_type]
                call_delta = delta + a
            else:
                call_delta = self.BS("CALL", K, sigma)[delta_type]
            obj = self.calc_sigma_from_delta(call_delta, sigma_S) - sigma
            # obj = self.sigma_ATM + c1 * (call_delta - self.delta_ATM) + c2 * (call_delta - self.delta_ATM)**2 - sigma
            return obj

        rr_coef = 1.5
        
        print("self.sigma_ATM:", self.sigma_ATM)
        print("self.sigma_RR:", self.sigma_RR)
        print("sigma_S:", sigma_S)
        vol_min = np.maximum(self.sigma_ATM - rr_coef * self.sigma_RR + sigma_S, 1e-6)  # Ensure vol_min is positive
        vol_max = self.sigma_ATM + rr_coef * self.sigma_RR + sigma_S
        print("vol_min:", vol_min)
        print("vol_max:", vol_max)
        
        expansions = 0
        max_expansions = 10

        while expansions < max_expansions:
            print(f"Expansion #{expansions + 1}")
            f_min = f(vol_min)
            f_max = f(vol_max)

            print(f"Bracket: [{vol_min:.6f}, {vol_max:.6f}]")
            print(f"f(min)={f_min:.6f}, f(max)={f_max:.6f}")


            if np.sign(f_min) != np.sign(f_max):
                print("if block entered")
                # find root
                res = root_scalar(
                    f,
                    method='brentq',  # or 'brentq' if you prefer
                    bracket=[vol_min, vol_max],  # reasonable bounds for the volatility
                    x0=self.sigma_ATM,  # initial guess
                    xtol=eps,
                    maxiter=max_iter
                )
                sigma_K = res.root
                # print(f"sigma_{K:.2f}: %{sigma_K*100:.2f}")
                return sigma_K

            else:
                ("else block entered")
                vol_min = np.maximum(vol_min * 0.8, 1e-6)  # Lower bound for vol_min
                vol_max *= 1.2  # Expand upper bound for vol_max
                expansions += 1

        # If we reach here, we didn't find a root in the initial bracket
        print(f"Failed to find brentq root sigma_K after {expansions} expansions.")
        # find root
        res = root_scalar(
            f,
            method='secant',  # or 'brentq' if you prefer
            x0=self.sigma_ATM,  # initial guess
            xtol=eps,
            maxiter=max_iter
        )
        sigma_K = res.root
        print(f"Secant method used, sigma for K={K}: %", np.round(sigma_K*100, 4))
        return sigma_K

        # # Adaptive bracket expansion
        # max_expansions = 10
        # expansions = 0

        # while expansions < max_expansions:
        #     try:
        #         f_min = f(vol_min)
        #         f_max = f(vol_max)

        #         print(f"Bracket: [{vol_min:.6f}, {vol_max:.6f}]")
        #         print(f"f(min)={f_min:.6f}, f(max)={f_max:.6f}")

        #         if np.sign(f_min) != np.sign(f_max):
        #             res = root_scalar(f, method='brentq', bracket=[vol_min, vol_max], xtol=eps, maxiter=100)
        #             sigma_K = res.root
        #             print(f"Found sigma_K={sigma_K:.6f} after {expansions} expansions")
        #             return sigma_K
        #     except Exception as e:
        #         print(f"Root finding failed: {e}")

        #     # Expand brackets
        #     vol_min = np.maximum(vol_min * 0.8, 1e-6)
        #     vol_max *= 1.2

        #     expansions += 1
        #     print(f"Expanded bracket to [{vol_min:.6f}, {vol_max:.6f}]")

    def set_K_C_P(self):
        self.sigma_C = self.sigma_ATM + 0.5 * self.sigma_RR + self.sigma_S
        self.sigma_P = self.sigma_ATM - 0.5 * self.sigma_RR + self.sigma_S

        self.K_C = self.calc_strike("CALL", self.sigma_C, self.delta_tilde)  # Call strike at delta pillar with smile vol
        self.K_P = self.calc_strike("PUT", self.sigma_P, -self.delta_tilde)  # Put strike at delta pillar with smile vol

    def print_init(self):
        print("OptionParams initialized with:")
        print(f"  eval_date: {self.eval_date}")
        print(f"  expiry_date: {self.expiry_date}")
        print(f"  delivery_date: {self.delivery_date}")
        print(f"  x: {self.x}")
        print(f"  rd: {self.rd}")
        print(f"  rf: {self.rf}")
        print(f"  sigma_ATM: {self.sigma_ATM}")
        print(f"  sigma_RR: {self.sigma_RR}")
        print(f"  sigma_SQ: {self.sigma_SQ}")
        print(f"  sigma_SM: {self.sigma_SM}")
        print(f"  delta_tilde: {self.delta_tilde}")
        print(f"  f: {self.f}")
        print(f"  K_ATM: {self.K_ATM}")
        print(f"  K_CSM: {self.K_CSM}")
        print(f"  K_PSM: {self.K_PSM}")
        print(f"  v_SM: {self.v_SM}")
        print(f"  a: {self.a}")

    def plot_smile_K(self):
        """
        Plot the implied volatility smile for the SPI model.
        """
        if not self.sigma_S:
            print("sigma_S not set, optimizing...")
            self.optimize_sigma_S()

        K_arr = np.linspace(self.K_ATM - self.K_ATM/4, self.K_ATM + self.K_ATM/4, 20)
        sigmas = np.array([self.find_SPI_sigma_K(self.simple_call_put(K), K)*100 for K in K_arr])

        plt.figure(figsize=(10, 6))
        plt.plot(K_arr, sigmas, label='SPI Smile (%)', color='blue')
        plt.axhline(self.sigma_ATM*100, color='red', linestyle='--', label='ATM Volatility')
        plt.axhline(self.sigma_SM*100, color='green', linestyle='--', label='Market Strangle Volatility')
        plt.title('SPI Implied Volatility Smile')
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_smile_delta(self):  # TODO
        """
        Plot the implied volatility smile for the SPI model.
        """
        if not self.sigma_S:
            print("sigma_S not set, optimizing...")
            self.optimize_sigma_S()

        delta_arr_put = np.linspace(-0.01, -0.50, 20)
        delta_arr_call = np.linspace(0.01, 0.50, 20)
        sigmas_put = np.array([self.calc_sigma_from_delta(delta, self.sigma_S)*100 if delta > 0 else self.calc_sigma_from_delta(delta + self.a, self.sigma_S)*100 for delta in delta_arr_put])
        sigmas_call = np.array([self.calc_sigma_from_delta(delta, self.sigma_S)*100 if delta > 0 else self.calc_sigma_from_delta(delta + self.a, self.sigma_S)*100 for delta in delta_arr_call])
        delta_arr = np.concatenate((delta_arr_put, delta_arr_call))
        sigmas = np.concatenate((sigmas_put, sigmas_call))

        plt.figure(figsize=(10, 6))
        plt.plot(sigmas, label='SPI Smile (%)', color='blue')
        plt.axhline(self.sigma_ATM*100, color='red', linestyle='--', label='ATM Volatility')
        plt.axhline(self.sigma_SM*100, color='green', linestyle='--', label='Market Strangle Volatility')
        plt.title('SPI Implied Volatility Smile')
        plt.xlabel('Delta')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid()
        plt.show()

    def print_results(self):
        print("-" * 50)
        print("Delta pillars:")
        print("K_C:", self.K_C)
        print("K_P:", self.K_P)
        print("sigma(K_C) calc: %", np.round(self.sigma_C * 100, 4))
        print("delta(K_C) calc: %", np.round(self.BS("CALL", self.K_C, self.sigma_C)["delta_S"] * 100, 4))
        print("sigma(K_P) calc: %", np.round(self.sigma_P * 100, 4))
        print("delta(K_P) calc: %", np.round(self.BS("PUT", self.K_P, self.sigma_P)["delta_S"] * 100, 4))
        print("-" * 50)
        print("sigma_S calc: %", np.round(self.sigma_S*100, 4))
        print("sigma_S calc via P, C: %", ((self.find_SPI_sigma_K("CALL", K=self.K_C) + self.find_SPI_sigma_K("PUT", K=self.K_P))/2 - self.sigma_ATM) * 100)
        print("-" * 50)
        print("RESTRICTION 1:")
        print("sigma(K_ATM) calc: %", np.round(self.find_SPI_sigma_K("CALL", K=self.K_ATM) * 100, 4))
        print("sigma(K_ATM) given: %", self.sigma_ATM * 100)
        print("-" * 50)
        print("RESTRICTION 2:")
        print("sigma_RR calc: %", np.round((self.find_SPI_sigma_K("CALL", K=self.K_C) - self.find_SPI_sigma_K("PUT", K=self.K_P))*100, 4))
        print("sigma_RR given: %", np.round(self.sigma_RR*100, 4))
        print("-" * 50)
        print("RESTRICTION 3:")
        print("v_SM calc: %", np.round((self.BS("CALL", self.K_CSM, self.find_SPI_sigma_K("CALL", self.K_CSM))["v_dom"] + self.BS("PUT", self.K_PSM, self.find_SPI_sigma_K("PUT", self.K_PSM))["v_dom"])*100, 4))
        print("v_SM given: %", np.round(self.v_SM*100, 4))
        print("-" * 50)

        print("K_C =", np.round(self.K_C, 4))
        print("delta K_C = %", np.round(self.BS("CALL", self.K_C, self.sigma_C)["delta_S"] * 100, 4))
        print("K_P =", np.round(self.K_P, 4))
        print("delta K_P = %", np.round(self.BS("PUT", self.K_P, self.sigma_P)["delta_S"] * 100, 4))

def calc_tx_with_spreads(buy_sell, call_put, K, rd_spread, rf_spread, ATM_vol_spread, calendar, basis_dict, spot_bd, eval_date, expiry_date, delivery_date, x, rd_simple, rf_simple, sigma_ATM, sigma_RR, sigma_SQ, delta_tilde=0.25, K_ATM_convention="fwd", delta_convention="fwd_pa"):
    rd_bid = rd_simple - rd_spread / 2
    rd_ask = rd_simple + rd_spread / 2

    rf_bid = rf_simple - rf_spread / 2
    rf_ask = rf_simple + rf_spread / 2

    if call_put.lower() == "call":
        if buy_sell.lower() == "buy":
            rf = rf_ask
            rd = rd_bid
        elif buy_sell.lower() == "sell":
            rf = rf_bid
            rd = rd_ask
    elif call_put.lower() == "put":
        if buy_sell.lower() == "buy":
            rf = rf_bid
            rd = rd_ask
        elif buy_sell.lower() == "sell":
            rf = rf_ask
            rd = rd_bid

    mid_params = OptionParams(
        calendar=calendar,
        basis_dict=basis_dict,
        spot_bd=spot_bd,
        eval_date=eval_date,
        expiry_date=expiry_date,
        delivery_date=delivery_date,
        x=x,
        rd_simple=rd,
        rf_simple=rf,
        sigma_ATM=sigma_ATM,
        sigma_RR=sigma_RR,
        sigma_SQ=sigma_SQ,
        delta_tilde=delta_tilde,
        K_ATM_convention=K_ATM_convention,
        delta_convention=delta_convention
    )

    mid_params.optimize_sigma_S()  # This will calibrate sigma_S
    mid_params.set_K_C_P()  # This will set K_C and K_P based
    mid_params.print_results()  # Print the results of the calibration

    K_ATM = mid_params.K_ATM
    sigma_ATM_bid = sigma_ATM - ATM_vol_spread / 2
    sigma_ATM_ask = sigma_ATM + ATM_vol_spread / 2

    bid_ATM_v_for = mid_params.BS(call_put, K_ATM, sigma_ATM_bid)["v_for"]
    ask_ATM_v_for = mid_params.BS(call_put, K_ATM, sigma_ATM_ask)["v_for"]
    ATM_v_for_diff = ask_ATM_v_for - bid_ATM_v_for

    sigma_K_mid = mid_params.find_SPI_sigma_K(call_put, K)

    v_for_mid = mid_params.BS(call_put, K, sigma_K_mid)["v_for"]
    v_for_bid = np.maximum(v_for_mid - ATM_v_for_diff / 2, 1e-6)  # Ensure v_for_bid is not negative
    v_for_ask = v_for_mid + ATM_v_for_diff / 2

    v_dom_bid = v_for_bid * mid_params.x
    v_dom_ask = v_for_ask * mid_params.x

    sigma_K_bid = mid_params.get_vol_from_price(v_dom_bid, K, call_put)
    sigma_K_ask = mid_params.get_vol_from_price(v_dom_ask, K, call_put)

    # print("_" * 40)
    # print()
    # print(f"TX results for {buy_sell.upper()} {call_put.upper()} @ K = {K}:")
    # print()
    # print("Domestic Rate (rd):", np.round(rd * 100, 4), "%")
    # print("Foreign Rate (rf):", np.round(rf * 100, 4), "%")
    # print()
    # print(f"MID Forward Parity: {np.round(mid_params.f, 4)}")
    # print()
    # print(f"ATM Strike Convention: {K_ATM_convention}\nDelta convention: {delta_convention}\n@{K:.3f} {call_put} results :")

    df_dict = {"BID": [f"%{np.round(sigma_K_bid * 100, 5)}", f"%{np.round(v_for_bid * 100, 5)}"],
               "ASK": [f"%{np.round(sigma_K_ask * 100, 5)}", f"%{np.round(v_for_ask * 100, 5)}"],
               "MID": [f"%{np.round(sigma_K_mid * 100, 5)}", f"%{np.round(v_for_mid * 100, 5)}"]}

    df = pd.DataFrame(df_dict, index=["sigma", "v_for"])
    print(df)
    print()
    # print("bid_ATM_v_for: %", np.round(bid_ATM_v_for*100, 5))
    # print("ask_ATM_v_for: %", np.round(ask_ATM_v_for*100, 5))
    # print("ATM_v_for_diff: %", np.round(ATM_v_for_diff*100, 5))
    print("v_for diff: %", np.round((v_for_ask - v_for_bid)*100, 4))
    return df, mid_params