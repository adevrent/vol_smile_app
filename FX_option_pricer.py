import numpy as np
import scipy.stats as ss
import QuantLib as ql
import datetime as dt
from scipy.optimize import brentq
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt


valid_datetypes = ["ISO", "datetime", "QL"]
valid_K_ATM_conventions = ["fwd", "fwd_delta_neutral", "spot"]

def convert_datetype(date, to_type):
    assert to_type in valid_datetypes, "Invalid to_date type"

    # If the input is already the target type, return it.
    if to_type == "ISO" and isinstance(date, str):
        return date
    elif to_type == "datetime" and isinstance(date, dt.date) and not isinstance(date, dt.datetime):
        return date
    elif to_type == "QL" and isinstance(date, ql.Date):
        return date

    if isinstance(date, str):  # date is ISO
        assert len(date) == 10, "Date is not in valid ISO format"
        if to_type == "datetime":
            return dt.datetime.strptime(date, "%Y-%m-%d").date()
        elif to_type == "QL":
            return ql.DateParser.parseISO(date)
    elif isinstance(date, dt.date) and not isinstance(date, dt.datetime):  # date is datetime.date
        if to_type == "ISO":
            return date.strftime("%Y-%m-%d")
        elif to_type == "QL":
            return ql.Date.from_date(date)
    elif isinstance(date, dt.datetime):  # if date is datetime.datetime, convert to date first
        # Convert to date before further processing
        date_only = date.date()
        return convert_datetype(date_only, to_type)
    elif isinstance(date, ql.Date):  # date is QuantLib Date
        if to_type == "ISO":
            return date.ISO()
        elif to_type == "datetime":
            return date.to_date()

    # If none of the above conditions match, raise an error.
    raise TypeError("Unsupported date type provided.")


valid_delta_conventions = ["spot", "spot_pa", "fwd"]


class OptionParams:
    def __init__(self, calendar, basis_dict, spot_bd, eval_date, expiry_date, delivery_date, x, rd_simple, rf_simple, sigma_ATM, sigma_RR, sigma_SQ, delta_tilde=0.25, K_ATM_convention="fwd", delta_convention="fwd_pa"):
        """call_put, K, sigma missing on purpose, to be set later"""
        self.calendar = calendar
        self.basis_dict = basis_dict
        self.spot_bd = spot_bd

        self.eval_date = eval_date
        self.expiry_date = expiry_date
        self.delivery_date = delivery_date
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

        self.rd = convert_simple_to_ccomp(rd_simple, self.tau_spot_dom)
        self.rf = convert_simple_to_ccomp(rf_simple, self.tau_spot_for)

        self.f = self.x * np.exp((self.rd - self.rf) * self.tau_spot_for)

        if K_ATM_convention.lower() not in valid_K_ATM_conventions:
            raise ValueError(f"Invalid K_ATM_convention: {K_ATM_convention}. Must be one of {valid_K_ATM_conventions}.")
        if K_ATM_convention.lower() == "fwd":
            self.K_ATM = self.f
        elif K_ATM_convention.lower() == "fwd_delta_neutral":
            self.K_ATM = self.f * np.exp(0.5 * self.sigma_ATM**2 * self.tau)
        elif K_ATM_convention.lower() == "spot":
            self.K_ATM = self.x

        # self.K_ATM = x
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
            self.a = np.exp(-self.rf * self.tau)  # equal to foreign DF from put-call delta parity
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
            K = self.x * np.exp(-phi * ss.norm.ppf(phi * np.exp(self.rf*self.tau) * delta_S) * sigma * np.sqrt(self.tau) + sigma * theta_plus * self.tau)
            return K
        elif self.delta_convention.lower() == "fwd":
            delta_S = np.exp(-self.rf * self.tau) * delta
            K = self.x * np.exp(-phi * ss.norm.ppf(phi * np.exp(self.rf*self.tau) * delta_S) * sigma * np.sqrt(self.tau) + sigma * theta_plus * self.tau)
            return K
        elif self.delta_convention.lower() == "spot_pa":
            print("Inside calc_strike with spot_pa delta convention")
            delta_S = delta
            """using delta_S_pa as vanilla delta_S to calculate K_max
            because premium-adjusted delta for a strike K is always
            SMALLER than the non-adjusted delta corresponding to
            the same strike"""
            K_npa = self.x * np.exp(-phi * ss.norm.ppf(phi * np.exp(self.rf*self.tau) * delta_S) * sigma * np.sqrt(self.tau) + sigma * theta_plus * self.tau)
            print("K_npa:", K_npa)
            K_max = K_npa
            K_min = self.solve_K_min(sigma, K_max, eps=eps, max_iter=1000)

            print("K_min:", K_min)
            print("K_max:", K_max)

            def f(K):
                print()
                print("####### Inside calc_strike objective function #######")
                d2 = self.calc_d2(K, sigma)
                delta_S_pa = phi * np.exp(-self.rf * self.tau) * K/self.f * ss.norm.cdf(phi * d2)
                print("delta_S_pa:", delta_S_pa)
                print("calc strike objective: %", np.round(delta_S_pa - delta, 6))
                print("K =", K)
                print("#######################################################")
                print()
                return delta_S_pa - delta

            # Adaptive bracket expansion
            max_expansions = 10
            expansions = 0

            while expansions < max_expansions:
                try:
                    f_min = f(K_min)
                    f_max = f(K_max)

                    print(f"Bracket: [{K_min:.6f}, {K_max:.6f}]")
                    print(f"f(min)={f_min:.6f}, f(max)={f_max:.6f}")

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
                print(f"Expanded bracket to [{K_min:.6f}, {K_max:.6f}]")

            # Final attempt with wider bounds
            try:
                res = root_scalar(f, method='brentq', bracket=[0.1*self.f, 10*self.f], xtol=eps, maxiter=max_iter)
                return res.root
            except:
                # Fallback to non-premium-adjusted strike
                return K_npa


    def calc_strike_jaeckel(self, call_put, sigma, delta_fwd_pa):
        if call_put.lower() == "call":
            phi = 1
        else:
            phi = -1

        alpha = phi * sigma
        y = np.log(K/self.f) + alpha/2

        q = ss.norm.pdf(y) / ss.norm.cdf(-y)
        f_1 = lambda y: alpha - q
        f_2 = lambda y: q * (y - q)
        f_3 = lambda y: q * (q * (3*y - 2*q) + 1 - y**2)
        f_4 = lambda y: q * (q * (q * (12*y - 6*q) + 4 - 7*y**2) + y * (y**2 - 3))
        f_5 = lambda y: q * (q * (q * (q * (60*y - 24*q) + 20 - 50*y**2) + y * (15*y**2 - 25)) + y**2 * (6 - y**2) - 3)
        f_6 = lambda y: q * (q * (q * (q * (q * (360*y - 120*q) + 120 - 390*y**2) + y * (180*y**2 - 210)) + y**2 * (101 - 31*y**2) - 28) + y * (15 + y**2 * (y**2 - 10)))

        def f(y):
            return np.log(np.abs(2*delta_fwd_pa)) + (alpha**2)/y

        # Initial guess function
        def y0(f_star):  #TODO
            return

        K = self.f * np.exp(alpha*y - (alpha**2)/2)

    def calc_d1(self, K, sigma):
        """
        Calculate d1 for Black-Scholes formula.

        Args:
            K (float): strike price
            sigma (float): volatility of the option, in 0.33 format.
        """
        d1 = (np.log(self.f/K) + 0.5*(sigma**2)*self.tau) / (sigma*np.sqrt(self.tau))
        return d1

    def calc_d2(self, K, sigma):
        """
        Calculate d2 for Black-Scholes formula.

        Args:
            K (float): strike price
            sigma (float): volatility of the option, in 0.33 format.
        """
        d2 = (np.log(self.f/K) - 0.5*(sigma**2)*self.tau) / (sigma*np.sqrt(self.tau))
        return d2

    def solve_K_min(self, sigma, K_max, eps=1e-6, max_iter=1000):
        def f(K_min):
            d2 = self.calc_d2(K_min, sigma)
            obj = sigma * np.sqrt(self.tau) * ss.norm.cdf(d2) - ss.norm.pdf(d2)
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

            print(f"K_min found: {K_min}")
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

        v_dom = phi * np.exp(-self.rd * self.tau) * (self.f * ss.norm.cdf(phi * d1) - K * ss.norm.cdf(phi * d2))
        v_for = v_dom/self.x

        delta_S = phi * np.exp(-self.rf * self.tau) * ss.norm.cdf(phi * d1)
        delta_S_pa = delta_S - v_for
        delta_dual = -phi * np.exp(-self.rd * self.tau) * ss.norm.cdf(phi * d2)
        delta_fwd = phi * ss.norm.cdf(phi * d1)
        delta_fwd_pa = phi * K/self.f * ss.norm.cdf(phi * d2)

        return {"v_dom": v_dom,
                "v_for": v_for,
                "delta_S": delta_S,
                "delta_S_pa": delta_S_pa,
                "delta_dual": delta_dual,
                "delta_fwd": delta_fwd,
                "delta_fwd_pa": delta_fwd_pa}

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
                self.a = np.exp(-self.rf * self.tau) * K_P/self.f
                print("        -- Initial K_P =", K_P)

            def f(sigma_S):
                sigma_P = np.maximum(self.sigma_ATM - 0.5*self.sigma_RR + sigma_S, 1e-3)  # Ensure sigma_P is positive
                print("        -- sigma_P = %", sigma_P*100)
                K_P = self.calc_strike("PUT", sigma_P, -self.delta_tilde)  # Update K_P based on sigma_P
                print("        -- K_P =", K_P)
                self.a = np.exp(-self.rf * self.tau) * K_P/self.f

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
            a = np.exp(-self.rf * self.tau) * K/self.f
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

        # find root
        res = root_scalar(
            f,
            method='brentq',  # or 'brentq' if you prefer
            bracket=[self.sigma_ATM - 0.3, self.sigma_ATM + 0.3],  # reasonable bounds for the volatility
            x0=self.sigma_ATM,  # initial guess
            xtol=eps,
            maxiter=max_iter
        )
        sigma_K = res.root
        # print(f"sigma_{K:.2f}: %{sigma_K*100:.2f}")
        return sigma_K

    def set_K_C_P(self):
        self.sigma_C = self.sigma_ATM + 0.5 * self.sigma_RR + self.sigma_S
        self.sigma_P = self.sigma_ATM - 0.5 * self.sigma_RR + self.sigma_S

        self.K_C = self.calc_strike("CALL", self.sigma_C, self.delta_tilde)  # Call strike at delta pillar with smile vol
        self.K_P = self.calc_strike("PUT", self.sigma_P, -self.delta_tilde)  # Put strike at delta pillar with smile vol

    def calc_TV_greeks(self, call_put, K):
        sigma = self.sigma_ATM
        BS_results = self.BS(call_put, K, sigma)
        return BS_results

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

        K_arr = np.linspace(self.K_ATM - self.K_ATM/5, self.K_ATM + self.K_ATM/5, 50)
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
        fig = plt.gcf()
        return fig

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

def convert_simple_to_ccomp(rate, tau):
    """
    Convert simple interest rate to continuously compounded rate.
    """
    return np.log(1 + rate * tau) / tau


myparams = OptionParams(
    calendar=ql.Turkey(),
    basis_dict={"FOR": ql.Actual360(), "DOM": ql.Actual365Fixed()},
    spot_bd=1,
    eval_date=ql.Date(23, 6, 2025),
    expiry_date=ql.Date(23, 7, 2025),
    delivery_date=ql.Date(24, 7, 2025),
    x=39.729,
    rd_simple=45.994/100,
    rf_simple=4.32/100,
    sigma_ATM=12/100,  # ATM volatility
    sigma_RR=13/100,  # Risk Reversal volatility
    sigma_SQ=1.75/100,  # Quoted Strangle volatility
    delta_tilde=0.25,  # pillar smile delta, e.g. 0.25 or 0.10
    K_ATM_convention="fwd",  # "fwd", "fwd_delta_neutral", "spot"
    delta_convention="spot_pa"  # "spot", "spot_pa", "fwd", "fwd_pa"
    )

myparams.optimize_sigma_S()  # This will calibrate sigma_S
myparams.set_K_C_P()  # This will set K_C and K_P based on the calibrated sigma_S
myparams.print_results()  # Print the results of the calibration

K = 42.0935  # Example strike price for testing
call_put = "CALL"
sigma_K = myparams.find_SPI_sigma_K(call_put, K)


v_for = myparams.BS(call_put, K, sigma_K)["v_for"]
# print("forward parity:", np.round(myparams.f, 4))
print(f"strike {K} sigma: %", np.round(sigma_K * 100, 4))
print(f"strike {K} v_for: %", np.round(v_for * 100, 4))
print(myparams.calc_TV_greeks(call_put, K))  # Calculate and print the TV greeks for the given strike
fig = myparams.plot_smile_K()  # Plot the implied volatility smile for the SPI model
fig.show()