#!/usr/bin/env python
"""
Parallelised Sequential SMC² + Posterior-Averaged Tabular Q-learning (ε-greedy)
===============================================================================

This script implements Policy π² (Algorithm 2 in the paper):

  • National-scale SEIR–ICU–VU epidemic model with vaccination.
  • Bayesian filtering via SMC² with 500 parameter particles.
  • Domain-randomised ‘true’ β trajectory per replicate.
  • Posterior-averaged tabular Q-learning, slice-based (Δ = 10 days):
      - G = 200 ICU bins on [1, 8000]
      - K = 25 θ-draws per decision time
      - E = 80,000 episodes per decision time
      - Planning horizon H = 100 days
      - ε-greedy: ε decays from 0.20 → 0.05
      - Learning rate α_k = C / (C + k), with C = 45
  • Decision horizon T_HORIZON = 300 days, split into B = 30 blocks of length Δ = 10.
  • Reward:
        r_t(y_t, a_t) =
            -P                     if y_t > T_crash
            -κ_icu y_t - κ_so-ec C(a_t, ℓ_t)   otherwise
    with:
        T_crash = 5000 beds,
        P = -1e6,
        κ_icu = 1,
        κ_so-ec ∈ {0.2, 0.5, 0.8} (looped over in main).
"""

# ------------------------------------------------------------------
# 0.  Imports and global config
# ------------------------------------------------------------------
import os, gc, time, warnings, sys
import multiprocessing
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from tqdm.auto import tqdm
from typing import Optional
import pandas as pd
from scipy.stats import ttest_rel

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------
# 1.  particles library imports
# ------------------------------------------------------------------
import particles
from particles import state_space_models as ssms
from particles import distributions as dists
from particles import smc_samplers as ssp

# ------------------------------------------------------------------
# 2.  Hyper-parameters (fixed to match the paper)
# ------------------------------------------------------------------
# Reward weights (κ_icu, κ_so-ec)
ICU_SCALE       = 1.0      # κ_icu = 1
COST_SCALE      = 0.0      # κ_so-ec, overridden in main loop

# Discount factor
GAMMA           = 0.95

# Decision structure
DECISION_PERIOD = 10       # Δ = 10 days
T_HORIZON       = 300      # total horizon
n_actions       = 4
n_replicates    = 10       # R = 10 independent replicates

# Q-learning hyperparameters for π² (Algorithm 2)
n_episodes      = 80000    # E = 80,000 episodes
PLAN_HORIZON    = 100      # H = 100 days
CRASH_THRESHOLD = 5000     # T_crash
CRASH_PENALTY   = -1e6    # P = -10^6

# Robbins–Monro constant C in α_k = C / (C + k)
C_const               = 45
epsilon_start_default = 0.20
N_bins_icu            = 200               # G = 200 ICU bins
icu_bins              = np.geomspace(1, 6000, N_bins_icu + 1)
EPS_START_BASE        = epsilon_start_default
EPISODES_MAX          = n_episodes
C_BASE                = C_const

# K = 25 posterior θ draws per block (set in main via argument)
# ------------------------------------------------------------------
# 3.  Real ICU data, vaccinations & prior (from SMC squared)
# ------------------------------------------------------------------
obs_icu = np.array([
    74,72,67,65,63,61,57,55,56,60,63,68,64,63,68,69,67,64,63,64,
    64,66,65,62,62,55,62,52,56,52,54,59,59,58,54,50,52,62,64,64,
    62,129,70,74,88,101,107,108,115,123,142,154,179,192,209,227,223,
    233,245,259,282,285,311,307,310,331,349,376,368,367,396,401,426,
    441,468,482,482,495,504,529,560,572,564,602,632,663,682,743,788,
    803,801,816,804,883,954,997,986,984,1003,1017,1048,1010,1081,1088,
    1158,1162,1194,1198,1228,1242,1241,1241,1248,1259,1299,1306,1300,
    1264,1242,1204,1214,1185,1182,1149,1094,1113,1086,1087,1109,1118,
    1094,1110,1117,1087,1123,1127,1159,1163,1188,1190,1239,1267,1327,
    1339,1374,1427,1437,1498,1556,1641,1728,1854,1899,1940,2017,2181,
    2310,2378,2550,2654,2814,2860,2963,3055,3175,3307,3351,3464,3525,
    3533,3570,3603,3603,3607,3727,3731,3736,3694,3634,3602,3585,3506,
    3486,3366,3414,3324,3328,3275,3217,3099,3047,2980,2911,2872,2780,
    2688,2592,2611,2577,2484,2393,2316,2251,2142,2122,2072,1956,1931,
    1866,1808,1747,1630,1658,1556,1507,1454,1417,1326,1273,1236,1187,
    1137,1101,1023,1000,963,930,882,846,797,749,710,692,676,647,613,
    566,559,537,532,532,518,501,470,436,433,422,416,406,393,377,362,
    342,343,337,333,315,294,298,292,283,291,273,249,237,221,220,211,
    208,196,188,178,170,167,172,168,162,161,144,145,135,131,135,126,
    123,119,115,115,116,117,114,113,115,117,113,111,113,117,115,110
])[:T_HORIZON]

# Vaccination data extension 
extra_first = np.array([
176489,176489,176489,176489,176489,176489,176489,
167707,167707,167707,167707,167707,167707,167707,
134286,134286,134286,134286,134286,134286,134286,
117143,117143,117143,117143,117143,117143,117143,
100000,100000,100000,100000,100000,100000,100000,
 88571, 88571, 88571, 88571, 88571, 88571, 88571,
 77143, 77143, 77143, 77143, 77143, 77143, 77143,
 65714, 65714, 65714, 65714, 65714, 65714, 65714,
 55714, 55714, 55714, 55714, 55714, 55714, 55714,
 47143, 47143, 47143, 47143, 47143, 47143, 47143,
 40000, 40000, 40000, 40000, 40000, 40000, 40000,
 31429, 31429, 31429, 31429, 31429, 31429, 31429,
 25714, 25714, 25714, 25714, 25714, 25714, 25714,
 22857, 22857, 22857, 22857, 22857, 22857, 22857,
 21429, 21429, 21429, 21429, 21429, 21429, 21429,
 20000, 20000, 20000, 20000, 20000, 20000, 20000,
 18571, 18571, 18571, 18571, 18571, 18571, 18571,
 28286, 28286, 28286, 28286, 28286, 28286, 28286,
 34373, 34373, 34373, 34373, 34373, 34373, 34373,
 31707, 31707, 31707, 31707, 31707, 31707, 31707
], dtype=int)

extra_second = np.array([
114286,114286,114286,114286,114286,114286,114286,
126063,126063,126063,126063,126063,126063,126063,
137143,137143,137143,137143,137143,137143,137143,
141429,141429,141429,141429,141429,141429,141429,
134286,134286,134286,134286,134286,134286,134286,
128571,128571,128571,128571,128571,128571,128571,
122857,122857,122857,122857,122857,122857,122857,
114286,114286,114286,114286,114286,114286,114286,
107143,107143,107143,107143,107143,107143,107143,
100000,100000,100000,100000,100000,100000,100000,
 92857, 92857, 92857, 92857, 92857, 92857, 92857,
 85714, 85714, 85714, 85714, 85714, 85714, 85714,
 80000, 80000, 80000, 80000, 80000, 80000, 80000,
 75714, 75714, 75714, 75714, 75714, 75714, 75714,
 72857, 72857, 72857, 72857, 72857, 72857, 72857,
 70000, 70000, 70000, 70000, 70000, 70000, 70000,
 67143, 67143, 67143, 67143, 67143, 67143, 67143,
 23173, 23173, 23173, 23173, 23173, 23173, 23173,
 20151, 20151, 20151, 20151, 20151, 20151, 20151,
 17360, 17360, 17360, 17360, 17360, 17360, 17360
], dtype=int)

daily_first_core = np.concatenate([
    np.zeros(149),
    np.full(7,142_857), np.full(7,214_286), np.full(7,285_714),
    np.full(21,285_714), np.full(49,214_286), np.full(49,142_857),
    np.full(7,285_714), np.full(49,214_286), np.full(7,160_189)
]).astype(int)

daily_second_core = np.concatenate([
    np.zeros(149),
    np.zeros(35), np.full(7, 71_429), np.full(49,214_286),
    np.full(7,285_714), np.full(49,214_286), np.full(7,160_189)
]).astype(int)

Vacc_horizon = T_HORIZON + 140
daily_first  = np.concatenate([daily_first_core,  extra_first])[:Vacc_horizon].astype(int)
daily_second = np.concatenate([daily_second_core, extra_second])[:Vacc_horizon].astype(int)

# ------------------------------------------------------------------
# Posterior-centred log-normal surrogate for β's (prior)
# ------------------------------------------------------------------
beta_nominal = np.array([0.318, 0.216, 0.060, 0.032])

def sample_beta_surrogate(day: int) -> np.ndarray:
    """
    Day-dependent log-normal surrogate for (β1,...,β4).

    The coefficient of variation shrinks from 0.50 at day 0 to 0.05 at day 300,
    mimicking the posterior contraction over time.
    """
    CV_max = np.array([0.50, 0.50, 0.50, 0.50])
    CV_min = np.array([0.05, 0.05, 0.05, 0.05])
    frac   = day / T_HORIZON
    cv_vec = CV_max * (1 - frac) + CV_min * frac
    sigma  = np.sqrt(np.log(1 + cv_vec**2))
    raw    = np.random.lognormal(np.log(beta_nominal), sigma)
    # enforce β1 >= β2 >= β3 >= β4
    return np.sort(raw)[::-1]

# ------------------------------------------------------------------
# 4.  Mapping RL-action → β parameter name
# ------------------------------------------------------------------
action2param = {0: 'beta1', 1: 'beta2', 2: 'beta3', 3: 'beta4'}

# ------------------------------------------------------------------
# 5.  SEIR-ICU RL environment (generator for a given θ)
# ------------------------------------------------------------------
class SEIR_ICU_RL:
    """
    SEIR-ICU-Vacc RL environment (daily step).
    Only difference from the SSM is that this uses a single β (current_beta)
    set by the RL agent, rather than the fixed historical schedule.
    """
    def __init__(self, daily_first, daily_second):
        self.Npop      = 60_000_000
        self.gamma_icu = 1/10
        self.p_icu     = 0.0005
        self.k_obs     = 10.0
        self.beta1, self.beta2, self.beta3, self.beta4 = 0.318, 0.216, 0.060, 0.032
        self.daily_first, self.daily_second = daily_first, daily_second
        self.pE = 1 - np.exp(-1/9.86)
        self.pI = 1 - np.exp(-1/10.41)
        self.gamma_v1 = self.gamma_v2 = self.gamma_v3 = 1/20.
        self.eps = np.array([0.50, 0.80, 0.70, 0.60, 0.95])
        self.psi = np.array([0.50, 0.75, 0.70, 0.60, 0.89])
        self.I_d1 = self.I_d2 = None
        self.current_beta = self.beta1

    def set_intervention(self, a: int):
        """Map RL-action a ∈ {0,1,2,3} to β-value."""
        self.current_beta = [self.beta1, self.beta2, self.beta3, self.beta4][a]

    def PX0(self) -> np.ndarray:
        """Initial state for the generator."""
        x0 = np.zeros(14, dtype=float)
        x0[0] = 59_947_215
        x0[1:5] = [8_108, 6_932, 37_665, 80]
        return x0

    def _ensure_buffers(self, N: int):
        if self.I_d1 is None or len(self.I_d1) != N:
            self.I_d1 = np.full(N, 5.)
            self.I_d2 = np.full(N, 5.)

    def step(self, xp: np.ndarray, t: int):
        """
        One-day transition with combined force of infection:

            λ_t = β * (I + I_V) / (total population)
        """
        S, E, I, R, C = xp[0], xp[1], xp[2], xp[3], xp[4]
        V              = xp[5:10].copy()
        EV, IV, RV, ICUv = xp[10], xp[11], xp[12], xp[13]

        self._ensure_buffers(1)

        # ---- vaccinations
        # 1) first doses: susceptibles → V[0]
        v1 = min(self.daily_first[t], int(S))
        S  -= v1
        V[0] += v1

        # 2) second doses (boosters) – always send to V[4], draining V[3], V[2], V[1]
        rem2 = self.daily_second[t]
        f3 = min(rem2, int(V[3])); rem2 -= f3
        f2 = min(rem2, int(V[2])); rem2 -= f2
        f1 = min(rem2, int(V[1])); rem2 -= f1
        v2 = f1 + f2 + f3
        V[3] -= f3
        V[2] -= f2
        V[1] -= f1
        V[4] += v2

        # ---- combined-force infection
        β   = self.current_beta
        N0  = S + E + I + R + C
        N1  = V.sum() + EV + IV + RV + ICUv
        total_inf = I + IV
        total_pop = N0 + N1
        lam = β * total_inf / total_pop if total_pop > 0 else 0.0

        # Unvaccinated new exposures
        newE0 = np.random.binomial(int(S), 1 - np.exp(-lam))

        # Vaccinated new exposures across all V compartments
        newEV = 0
        for i in range(5):
            prob = 1.0 - np.exp(-lam * (1 - self.eps[i]))
            ne   = np.random.binomial(int(V[i]), prob)
            V[i] -= ne
            newEV += ne

        S  -= newE0
        E  += newE0
        EV += newEV

        # ---- waning
        w12 = np.random.binomial(int(V[0]), self.gamma_v1)
        w23 = np.random.binomial(int(V[1]), self.gamma_v2)
        w34 = np.random.binomial(int(V[2]), self.gamma_v3)
        V[0] -= w12
        V[1] += w12 - w23
        V[2] += w23 - w34
        V[3] += w34

        S  = np.maximum(S , 0)
        V  = np.maximum(V , 0)

        # ---- progression (unvaccinated)
        dE0  = np.random.binomial(int(E), self.pE)
        dI0  = np.random.binomial(int(I), self.pI)
        adm0 = np.random.binomial(int(self.I_d2[0]), self.p_icu)
        rec0 = np.random.binomial(int(C), self.gamma_icu)

        E += -dE0
        I += dE0 - dI0
        R += dI0 - adm0
        C += adm0 - rec0

        # ---- progression (vaccinated)
        dEV = np.random.binomial(int(EV), self.pE)
        EV -= dEV
        IV += dEV

        num = (V * (self.p_icu * (1 - self.psi))).sum()
        den = V.sum()
        p_icu_v = num / den if den > 0 else self.p_icu
        adm1 = np.random.binomial(int(IV), p_icu_v)
        IV  -= adm1
        ICUv += adm1

        dIV = np.random.binomial(int(IV), self.pI)
        IV -= dIV
        RV += dIV

        rec1 = np.random.binomial(int(ICUv), self.gamma_icu)
        ICUv -= rec1
        RV   += rec1

        # ---- book-keeping
        self.I_d2[0], self.I_d1[0] = self.I_d1[0], I

        xp_next = np.array([
            S, E, I, R, C,
            *V, EV, IV, RV, ICUv
        ], dtype=float)
        return xp_next, xp_next[4] + xp_next[13]

# ------------------------------------------------------------------
# 6.  SEIR-ICU state-space model for SMC²
# ------------------------------------------------------------------
class SEIR_ICU(ssms.StateSpaceModel):
    """
    SEIR-ICU-Vacc state-space model for SMC².
    Uses the fixed (historical) β schedule, but with combined-force infection.
    """

    default_params = dict(
        Npop      = 60_000_000,
        gamma_icu = 1/10,
        p_icu     = 0.0005,
        k_obs     = 10.0,
        beta1     = 0.318,
        beta2     = 0.216,
        beta3     = 0.060,
        beta4     = 0.032
    )

    def __init__(self, **kw):
        super().__init__(**kw)
        # Updated piecewise-constant β schedule
        self.updated_bounds = [
            (0,   18,  self.beta1),
            (18,  36,  self.beta2),
            (36,  49,  self.beta3),
            (49,  88,  self.beta2),
            (88,  120, self.beta3),
            (120, 156, self.beta2),
            (156, 212, self.beta4),
            (212, 249, self.beta4),
            (249, 283, self.beta3),
            (283, 300, self.beta2)
        ]
        self.schedule = np.concatenate([
            np.full(hi - lo, b) for lo, hi, b in self.updated_bounds
        ])

        self.pE = 1 - np.exp(-1/9.86)
        self.pI = 1 - np.exp(-1/10.41)
        self.gamma_v1 = self.gamma_v2 = self.gamma_v3 = 1/20.

        self.daily_first  = daily_first
        self.daily_second = daily_second

        self.eps = np.array([0.50, 0.80, 0.70, 0.60, 0.95])
        self.psi = np.array([0.50, 0.75, 0.70, 0.60, 0.89])

        self.I_d1 = self.I_d2 = None

    def PX0(self):
        """Initial distribution over latent states (Dirac)."""
        x0 = np.zeros(14)
        x0[0] = 59_976_749
        x0[1:5] = [3_321, 9_120, 24_707, 80]
        return dists.Dirac(loc=x0)

    def _ensure_buffers(self, N):
        if self.I_d1 is None or self.I_d1.size != N:
            self.I_d1 = np.full(N, 5.)
            self.I_d2 = np.full(N, 5.)

    def PX(self, t, xp):
        """Expose the transition kernel so particles.SMC can call it."""
        return SEIR_ICU._Step(self, xp, t)

    class _Step(dists.ProbDist):
        def __init__(self, m, xp, t):
            self.m, self.xp, self.t = m, xp, t
            self.dim = len(xp)

        def logpdf(self, x):
            return np.zeros(np.atleast_2d(x).shape[0])

        def rvs(self, size=None):
            m, t   = self.m, self.t
            x      = np.atleast_2d(self.xp).copy()
            S, E, I, R, C = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]
            V = x[:,5:10].T
            EV, IV, RV, ICUv = x[:,10], x[:,11], x[:,12], x[:,13]

            m._ensure_buffers(S.size)

            # ---- vaccinations
            v1 = np.minimum(m.daily_first[t], S).astype(int)
            S  -= v1
            V[0] += v1

            rem = m.daily_second[t]
            f3  = np.minimum(rem, V[3].astype(int));  rem -= f3
            f2  = np.minimum(rem, V[2].astype(int));  rem -= f2
            f1  = np.minimum(rem, V[1].astype(int));  rem -= f1
            V[3] -= f3
            V[2] -= f2
            V[1] -= f1
            V[4] += (f1 + f2 + f3)

            # ---- combined-force infection
            β  = m.schedule[t]
            N0 = S + E + I + R + C
            N1 = V.sum(axis=0) + EV + IV + RV + ICUv
            lam = np.where(N0 + N1 > 0, β * (I + IV) / (N0 + N1), 0.)

            newE0 = np.random.binomial(S.astype(int), 1 - np.exp(-lam))
            S    -= newE0
            E    += newE0

            newEV = np.zeros_like(S)
            for i in range(5):
                prob = 1.0 - np.exp(-lam * (1.0 - m.eps[i]))
                ne   = np.random.binomial(V[i].astype(int), prob)
                V[i] -= ne
                newEV += ne
                EV += newEV  # (kept as in your code, even if a bit unusual)

            # ---- waning
            w12 = np.random.binomial(V[0].astype(int), m.gamma_v1)
            w23 = np.random.binomial(V[1].astype(int), m.gamma_v2)
            w34 = np.random.binomial(V[2].astype(int), m.gamma_v3)
            V[0] -= w12
            V[1] += w12 - w23
            V[2] += w23 - w34
            V[3] += w34

            S  = np.maximum(S , 0)
            V  = np.maximum(V , 0)

            # ---- progression (unvaccinated)
            dE0  = np.random.binomial(E.astype(int), m.pE)
            dI0  = np.random.binomial(I.astype(int), m.pI)
            adm0 = np.random.binomial(m.I_d2.astype(int), m.p_icu)
            rec0 = np.random.binomial(C.astype(int), m.gamma_icu)

            E -= dE0
            I += dE0 - dI0
            R += dI0 - adm0
            C += adm0 - rec0

            # ---- progression (vaccinated)
            dEV = np.random.binomial(EV.astype(int), m.pE)
            EV -= dEV
            IV += dEV

            num = (V * (m.p_icu * (1 - m.psi)[:, None])).sum(axis=0)
            den = V.sum(axis=0)
            p_icu_v = np.where(den > 0, num / den, m.p_icu)
            adm1 = np.random.binomial(IV.astype(int), p_icu_v)
            IV  -= adm1
            ICUv += adm1

            dIV = np.random.binomial(IV.astype(int), m.pI)
            IV -= dIV
            RV += dIV

            rec1 = np.random.binomial(ICUv.astype(int), m.gamma_icu)
            ICUv -= rec1
            RV   += rec1

            m.I_d2, m.I_d1 = m.I_d1.copy(), I.copy()

            return np.column_stack([
                S, E, I, R, C,
                V[0], V[1], V[2], V[3], V[4],
                EV, IV, RV, ICUv
            ])

    def PY(self, t, xp, x):
        """Negative binomial observation model for ICU counts."""
        ICU = x[4] + x[-1] if x.ndim == 1 else x[:,4] + x[:,-1]
        mu  = np.maximum(1., ICU)
        p   = self.k_obs / (self.k_obs + mu)

        class NB(dists.ProbDist):
            dim, dtype = 1, int
            def __init__(self, k, p): self.k, self.p = k, p
            def logpdf(self, y):      return nbinom.logpmf(y, self.k, self.p)
            def rvs(self, size=None): return nbinom.rvs(self.k, self.p, size=size)

        return NB(self.k_obs, p)

# ------------------------------------------------------------------
# 7.  Helper functions (state discretisation, costs, streak updates)
# ------------------------------------------------------------------
def discretize(x: np.ndarray) -> int:
    """Map a state x to its ICU bin index g ∈ {0,...,G-1}."""
    icu = x[4] + x[13] if x.ndim == 1 else x[:,4] + x[:,-1]
    return np.clip(np.digitize(icu, icu_bins) - 1, 0, N_bins_icu - 1)

def intervention_cost(a: int, streak: int) -> float:
    """
    Socio-economic intervention cost C(a, ℓ):

      - action 0: no cost
      - action 1: logarithmic (mild, diminishing)
      - action 2: linear (moderate)
      - action 3: linear with large slope (strict lockdown)
    """
    if a == 1:
        return 50 * np.log1p(streak)
    elif a == 2:
        return 200 * streak
    elif a == 3:
        return 800 * streak
    else:
        return 0.0

def action_tier(a: int) -> int:
    """
    Tier mapping:
        action 0 → tier 0
        action 1 → tier 1
        action 2 or 3 → tier 2
    Used for the 'de-escalation reset' rule on streaks.
    """
    if a < 2:
        return a
    return 2

def update_streak(prev_a: Optional[int], prev_streak: int, new_a: int) -> int:
    """
    Custom streak logic (as described in the paper):

      - action 0 breaks any streak
      - first non-zero action after 0 starts streak at 1
      - any de-escalation (tier goes down) resets to 1
      - otherwise, streak increments
    """
    if new_a == 0:
        return 0
    if prev_a is None or prev_a == 0:
        return 1
    if action_tier(new_a) < action_tier(prev_a):
        return 1
    return prev_streak + 1

def block_planning_params(day0: int):
    """
    For the paper, we keep constant planning hyperparameters across blocks:
      - n_eps_blk = E = 30,000
      - C_blk     = 45
      - eps_blk   = 0.20
    """
    n_eps_blk = EPISODES_MAX
    C_blk     = C_BASE
    eps_blk   = EPS_START_BASE
    return n_eps_blk, C_blk, eps_blk

# ------------------------------------------------------------------
# 8.  ONE replicate with posterior-averaged Q-learning (π²)
# ------------------------------------------------------------------
def run_one_replicate(rep_id: int, seed: Optional[int] = None, K: int = 25):
    """
    Bayesian-averaged Q-learning replicate (Policy π²):

      • At each 10-day block draw K θ’s from the current posterior.
      • Run n_episodes of Q-learning for those K models (in parallel).
      • Each worker returns (Q, ΔQ-per-episode vector).
      • Average the K Q-tables → Q_avg; average ΔQ traces → deltas_avg.
      • Deploy the greedy action from Q_avg for the next 10 days in the
        real environment, collect ICU data, update the posterior (SMC²).
      • Repeat for all 30 blocks (T_HORIZON = 300, Δ = 10).

    Returns
    -------
    rl_icu : list[float]
        Realised ICU trajectory.
    week_actions : list[int]
        Deployed action per 10-day block.
    total_reward : float
        Total discounted reward over 300 days.
    Q_avg_list : list[np.ndarray]
        K-averaged Q-table per block.
    deltas_avg_list : list[np.ndarray]
        Mean ΔQ per episode per block.
    total_icu_load : float
        Sum of daily ICU occupancy over the horizon.
    total_intervention_cost : float
        Sum of socio-economic costs (scaled by κ_so-ec).
    crash_count : int
        Number of days where ICU > T_crash.
    ci_widths : dict[str, list[float]]
        90% CI width per β per block (posterior contraction diagnostic).
    """
    import time, gc
    from joblib import Parallel, delayed, parallel_backend

    # ------------------------------------------------------------------
    # 0.  Seeding
    # ------------------------------------------------------------------
    if seed is None:
        seed = (int(time.time() * 1000) + rep_id) % 2**32
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1.  Ground-truth environment (domain-randomised θ*)
    # ------------------------------------------------------------------
    theta_star = sample_beta_surrogate(T_HORIZON)
    env_true   = SEIR_ICU_RL(daily_first, daily_second)
    env_true.beta1, env_true.beta2, env_true.beta3, env_true.beta4 = theta_star
    x_real     = env_true.PX0()

    # ------------------------------------------------------------------
    # 2.  Empirical-Bayes priors and posterior storage for β's
    # ------------------------------------------------------------------
    CV_max   = np.array([0.50, 0.50, 0.50, 0.50])
    beta_0   = sample_beta_surrogate(0)

    def gamma_from_mean_cv(m, cv):
        shape, rate = 1. / cv**2, (1. / cv**2) / m
        return dists.Gamma(a=shape, b=rate)

    beta_priors = {
        p: gamma_from_mean_cv(m, c)
        for p, m, c in zip(action2param.values(), beta_0, CV_max)
    }
    p_icu_prior       = dists.Beta(50., 50. / 0.00106 - 50.)
    posterior_storage = {p: [] for p in action2param.values()}

    # ------------------------------------------------------------------
    # 3.  Real-world bookkeeping
    # ------------------------------------------------------------------
    rl_icu                   = []
    week_actions             = []
    total_reward             = 0.0
    total_icu_load           = 0.0
    total_intervention_cost  = 0.0
    crash_count              = 0
    prev_action, real_streak = None, 0
    ci_widths                = {p: [] for p in action2param.values()}

    # ------------------------------------------------------------------
    # 4.  Per-block storage for diagnostics
    # ------------------------------------------------------------------
    Q_avg_list      = []
    deltas_avg_list = []

    # ------------------------------------------------------------------
    # 5.  Outer loop over B = 30 decision blocks
    # ------------------------------------------------------------------
    for block_idx, t0 in enumerate(range(0, T_HORIZON, DECISION_PERIOD)):
        n_ep_blk, C_blk, eps_start_blk = block_planning_params(t0)
        plan_end = t0 + PLAN_HORIZON                        # exclusive upper bound

        # --------------------------------------------------------------
        # 5.1  Draw K θ-samples from current posterior
        # --------------------------------------------------------------
        thetas = []
        for _ in range(K):
            θ = []
            for pname in action2param.values():
                if posterior_storage[pname]:
                    s, w = posterior_storage[pname][-1]
                    θ.append(np.random.choice(s, p=w / w.sum()))
                else:
                    θ.append(beta_priors[pname].rvs())
            thetas.append(np.sort(θ)[::-1])                 # sort β in descending order

        # --------------------------------------------------------------
        # 5.2  Parallel Q-learning workers: one per θ
        # --------------------------------------------------------------
        def learn_one_theta(theta, rng_seed):
            """
            Run Q-learning for a single θ; return (Q, ΔQ-per-episode).

            Here we implement Algorithm 2:
              - slice-based rewards (Δ = 10 days)
              - ε-greedy exploration with exponential decay
              - α_k = C / (C + k) using visit-count N_sa as k
              - crash penalty P applied in planning if ICU > T_crash
            """
            np.random.seed(rng_seed)

            Q     = np.zeros((N_bins_icu, n_actions))
            N_sa  = np.zeros_like(Q, dtype=int)
            Q_old = Q.copy()
            slice_deltas = []

            eps_end    = 0.05
            decay_rate = -np.log(eps_end / eps_start_blk) / (n_ep_blk - 1)

            for ep in range(n_ep_blk):
                epsilon   = eps_start_blk * np.exp(-decay_rate * ep)
                env_sim   = SEIR_ICU_RL(daily_first, daily_second)
                env_sim.beta1, env_sim.beta2, env_sim.beta3, env_sim.beta4 = theta

                x = x_real.copy()
                sim_prev_a, sim_streak = prev_action, real_streak

                R_block = 0.0

                for day in range(t0, plan_end):
                    # At the start of each Δ-slice pick action
                    if (day - t0) % DECISION_PERIOD == 0:
                        s = discretize(x)
                        sim_prev_s = s
                        if np.random.rand() < epsilon:
                            a = np.random.randint(n_actions)
                        else:
                            best = np.flatnonzero(Q[s] == Q[s].max())
                            a = np.random.choice(best)
                        sim_streak = update_streak(sim_prev_a, sim_streak, a)
                        sim_prev_a = a
                        R_block = 0.0
                    else:
                        # Within slice: same action, streak evolves
                        sim_streak = update_streak(sim_prev_a, sim_streak, sim_prev_a)

                    env_sim.set_intervention(a)
                    x, icu = env_sim.step(x, day)

                    if icu > CRASH_THRESHOLD:
                        R_block += CRASH_PENALTY
                    else:
                        R_block += -(ICU_SCALE * icu
                                     + COST_SCALE * intervention_cost(a, sim_streak))

                    end_slice = ((day - t0) % DECISION_PERIOD == DECISION_PERIOD - 1) \
                                or (day == T_HORIZON - 1)
                    if end_slice:
                        s_next = discretize(x)
                        td     = R_block + GAMMA * Q[s_next].max()
                        N_sa[sim_prev_s, sim_prev_a] += 1
                        α = C_blk / (C_blk + float(N_sa[sim_prev_s, sim_prev_a]))
                        Q[sim_prev_s, sim_prev_a] += α * (td - Q[sim_prev_s, sim_prev_a])

                        # store ΔQ for this slice
                        slice_deltas.append(np.max(np.abs(Q - Q_old)))
                        Q_old = Q.copy()

            # Compress (episodes × slices) → (episodes) by max over slices
            n_slices = PLAN_HORIZON // DECISION_PERIOD
            rep_arr  = np.asarray(slice_deltas).reshape(n_ep_blk, n_slices).max(axis=1)
            if n_ep_blk < EPISODES_MAX:
                rep_arr = np.pad(rep_arr, (0, EPISODES_MAX - n_ep_blk))

            return Q, rep_arr

        # ----- launch the K workers in parallel -----------------------
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=K, prefer="processes")(
                delayed(learn_one_theta)(theta, np.random.randint(0, 2**32))
                for theta in thetas
            )

        # accumulate Monte Carlo sums
        Q_sum      = np.zeros((N_bins_icu, n_actions))
        deltas_sum = np.zeros(EPISODES_MAX)
        for Q, rep_arr in results:
            Q_sum      += Q
            deltas_sum += rep_arr

        # --------------------------------------------------------------
        # 5.3  Monte Carlo averages over the K θ’s
        # --------------------------------------------------------------
        Q_avg      = Q_sum / K
        deltas_avg = deltas_sum / K

        Q_avg_list.append(Q_avg)
        deltas_avg_list.append(deltas_avg)

        # --------------------------------------------------------------
        # 5.4  Deploy greedy action under Q_avg for the next 10 days
        # --------------------------------------------------------------
        s0 = discretize(x_real)
        a0 = int(np.argmax(Q_avg[s0]))
        print(f"[Rep {rep_id}][Block {block_idx}] Chosen action: {a0} (state bin={s0})")
        week_actions.append(a0)

        # Update streak with chosen action
        real_streak = update_streak(prev_action, real_streak, a0)
        prev_action = a0

        for day in range(t0, t0 + DECISION_PERIOD):
            env_true.set_intervention(a0)
            x_real, icu_real = env_true.step(x_real, day)
            real_streak = update_streak(prev_action, real_streak, a0)

            rl_icu.append(icu_real)
            total_icu_load += icu_real

            cost = intervention_cost(a0, real_streak)
            total_intervention_cost += COST_SCALE * cost

            if icu_real > CRASH_THRESHOLD:
                crash_count += 1
                inst_reward = CRASH_PENALTY
            else:
                inst_reward = -(ICU_SCALE * icu_real + COST_SCALE * cost)

            total_reward += inst_reward

        # --------------------------------------------------------------
        # 5.5  Posterior update via SMC² (N = 500 parameter particles)
        # --------------------------------------------------------------
        data_seg = np.asarray(rl_icu[-DECISION_PERIOD:]).reshape(-1, 1)
        param    = action2param[a0]

        prior = dists.StructDist({
            param:   beta_priors[param],
            'p_icu': p_icu_prior
        })

        fk  = ssp.SMC2(
            ssm_cls=SEIR_ICU,
            prior=prior,
            data=data_seg,
            init_Nx=100,
            len_chain=3,
            ar_to_increase_Nx=0.10,
            smc_options=dict(qmc=False)
        )
        alg = particles.SMC(fk=fk, N=500, verbose=False)  # N = 500 as in the paper
        for _ in alg:
            pass

        θ, W = alg.X.theta[param], alg.W
        posterior_storage[param].append((θ.copy(), W.copy()))

        def wq(vals, weights, q):
            sorter = np.argsort(vals)
            v, w   = vals[sorter], weights[sorter]
            cw     = np.cumsum(w)
            return np.interp(q * cw[-1], cw, v)

        # store 90 % CI width for every β, even if it wasn’t updated this block
        for pname in action2param.values():
            if posterior_storage[pname]:
                s, w = posterior_storage[pname][-1]
                lo, hi = wq(s, w, 0.05), wq(s, w, 0.95)
            else:
                prior_samps = beta_priors[pname].rvs(size=2_000)
                lo, hi      = np.percentile(prior_samps, [5, 95])
            ci_widths[pname].append(hi - lo)

        # Empirical-Bayes update of prior for the parameter just updated
        m = np.sum(θ * W)
        v = np.sum(W * (θ - m) ** 2)
        beta_priors[param] = dists.Gamma(a=m * m / v, b=m / v)

        gc.collect()

    # ------------------------------------------------------------------
    # 6.  Return everything needed by the driver
    # ------------------------------------------------------------------
    return (rl_icu,
            week_actions,
            total_reward,
            Q_avg_list,
            deltas_avg_list,
            total_icu_load,
            total_intervention_cost,
            crash_count,
            ci_widths)

# ------------------------------------------------------------------
# B1 & B2: Baseline functions (MPC, Random) — kept for future use
# ------------------------------------------------------------------
# (Unchanged except for using N=500 in SMC; not called in main now.)

def run_baseline_mpc(rep_id: int, seed: Optional[int] = None):
    """
    Deterministic MPC baseline with MAP-parameter planning (kept for completeness).
    Not executed in the current main block.
    """
    import time, gc
    if seed is None:
        seed = (int(time.time() * 1000) + rep_id) % 2**32
    np.random.seed(seed)

    theta_star = sample_beta_surrogate(T_HORIZON)
    env_true   = SEIR_ICU_RL(daily_first, daily_second)
    env_true.beta1, env_true.beta2, env_true.beta3, env_true.beta4 = theta_star
    x_real     = env_true.PX0()

    CV_max   = np.array([0.50, 0.50, 0.50, 0.50])
    beta_0   = sample_beta_surrogate(0)

    def gamma_from_mean_cv(m, cv):
        shape, rate = 1. / cv**2, (1. / cv**2) / m
        return dists.Gamma(a=shape, b=rate)

    beta_priors = {
        p: gamma_from_mean_cv(m, c)
        for p, m, c in zip(action2param.values(), beta_0, CV_max)
    }
    p_icu_prior       = dists.Beta(50., 50. / 0.00106 - 50.)
    posterior_storage = {p: [] for p in action2param.values()}

    rl_icu                  = []
    week_actions            = []
    total_reward            = 0.0
    total_icu_load          = 0.0
    total_intervention_cost = 0.0
    crash_count             = 0
    prev_action             = None
    real_streak             = 0

    Q_map_list      = []
    deltas_map_list = []

    for block_idx, t0 in enumerate(range(0, T_HORIZON, DECISION_PERIOD)):
        n_ep_blk, C_blk, eps_start_blk = block_planning_params(t0)
        plan_end = t0 + PLAN_HORIZON

        theta_map = []
        for pname in action2param.values():
            if posterior_storage[pname]:
                samples, weights = posterior_storage[pname][-1]
                theta_map.append(samples[np.argmax(weights)])
            else:
                m = beta_priors[pname].a / beta_priors[pname].b
                theta_map.append(m)
        b1, b2, b3, b4 = np.sort(theta_map)[::-1]

        Q     = np.zeros((N_bins_icu, n_actions))
        N_sa  = np.zeros_like(Q, dtype=int)
        Q_old = Q.copy()
        rep_deltas = []
        eps_end    = 0.05
        decay_rate = -np.log(eps_end / eps_start_blk) / (n_ep_blk - 1)

        for ep in range(n_ep_blk):
            epsilon = eps_start_blk * np.exp(-decay_rate * ep)
            env_sim = SEIR_ICU_RL(daily_first, daily_second)
            env_sim.beta1, env_sim.beta2, env_sim.beta3, env_sim.beta4 = b1, b2, b3, b4

            x = x_real.copy()
            sim_prev_a, sim_streak = prev_action, real_streak
            R_block = 0.0

            for day in range(t0, plan_end):
                if (day - t0) % DECISION_PERIOD == 0:
                    s = discretize(x)
                    sim_prev_s = s
                    if np.random.rand() < epsilon:
                        a = np.random.randint(n_actions)
                    else:
                        best = np.flatnonzero(Q[s] == Q[s].max())
                        a = np.random.choice(best)
                    sim_streak = update_streak(sim_prev_a, sim_streak, a)
                    sim_prev_a = a
                    R_block = 0.0
                else:
                    sim_streak = update_streak(sim_prev_a, sim_streak, sim_prev_a)

                env_sim.set_intervention(a)
                x, icu = env_sim.step(x, day)
                if icu > CRASH_THRESHOLD:
                    R_block += CRASH_PENALTY
                else:
                    R_block += -(ICU_SCALE * icu + COST_SCALE * intervention_cost(a, sim_streak))

                end_slice = ((day - t0) % DECISION_PERIOD == DECISION_PERIOD - 1) or (day == T_HORIZON - 1)
                if end_slice:
                    s_next = discretize(x)
                    td = R_block + GAMMA * Q[s_next].max()
                    N_sa[sim_prev_s, sim_prev_a] += 1
                    α = C_blk / (C_blk + float(N_sa[sim_prev_s, sim_prev_a]))
                    Q[sim_prev_s, sim_prev_a] += α * (td - Q[sim_prev_s, sim_prev_a])
                    delta = np.max(np.abs(Q - Q_old))
                    rep_deltas.append(delta)
                    Q_old = Q.copy()

        if len(rep_deltas) < EPISODES_MAX:
            rep_deltas.extend([0.0] * (EPISODES_MAX - len(rep_deltas)))
        deltas_map_list.append(np.asarray(rep_deltas[:EPISODES_MAX]))
        Q_map_list.append(Q)

        s0 = discretize(x_real)
        a0 = int(np.argmax(Q[s0]))
        week_actions.append(a0)

        real_streak = update_streak(prev_action, real_streak, a0)
        prev_action = a0

        for day in range(t0, t0 + DECISION_PERIOD):
            env_true.set_intervention(a0)
            x_real, icu_real = env_true.step(x_real, day)
            real_streak = update_streak(prev_action, real_streak, a0)

            rl_icu.append(icu_real)
            total_icu_load += icu_real
            cost = intervention_cost(a0, real_streak)
            total_intervention_cost += COST_SCALE * cost
            if icu_real > CRASH_THRESHOLD:
                crash_count += 1
                inst_reward = CRASH_PENALTY
            else:
                inst_reward = -(ICU_SCALE * icu_real + COST_SCALE * cost)
            total_reward += inst_reward

        data_seg = np.asarray(rl_icu[-DECISION_PERIOD:]).reshape(-1, 1)
        param = action2param[a0]
        prior = dists.StructDist({param: beta_priors[param], 'p_icu': p_icu_prior})
        fk = ssp.SMC2(ssm_cls=SEIR_ICU, prior=prior, data=data_seg,
                      init_Nx=100, len_chain=3, ar_to_increase_Nx=0.10,
                      smc_options=dict(qmc=False))
        alg = particles.SMC(fk=fk, N=500, verbose=False)
        for _ in alg:
            pass
        θ, W = alg.X.theta[param], alg.W
        posterior_storage[param].append((θ.copy(), W.copy()))
        m = np.sum(θ * W)
        v = np.sum(W * (θ - m) ** 2)
        beta_priors[param] = dists.Gamma(a=m * m / v, b=m / v)
        gc.collect()

    return (
        rl_icu,
        week_actions,
        total_reward,
        Q_map_list,
        deltas_map_list,
        total_icu_load,
        total_intervention_cost,
        crash_count
    )

def run_baseline_random(rep_id: int, seed: Optional[int] = None):
    """
    Random-uniform baseline (kept for completeness).
    Not executed in the current main block.
    """
    import time, gc
    if seed is None:
        seed = (int(time.time() * 1000) + rep_id) % 2**32
    np.random.seed(seed)

    theta_star = sample_beta_surrogate(T_HORIZON)
    env_true   = SEIR_ICU_RL(daily_first, daily_second)
    env_true.beta1, env_true.beta2, env_true.beta3, env_true.beta4 = theta_star
    x_real     = env_true.PX0()

    CV_max   = np.array([0.50, 0.50, 0.50, 0.50])
    beta_0   = sample_beta_surrogate(0)

    def gamma_from_mean_cv(m, cv):
        shape, rate = 1./cv**2, (1./cv**2)/m
        return dists.Gamma(a=shape, b=rate)

    beta_priors = {
        p: gamma_from_mean_cv(m, c)
        for p,m,c in zip(action2param.values(), beta_0, CV_max)
    }
    p_icu_prior       = dists.Beta(50., 50./0.00106 - 50.)
    posterior_storage = {p: [] for p in action2param.values()}

    rl_icu                  = []
    week_actions            = []
    total_reward            = 0.0
    total_icu_load          = 0.0
    total_intervention_cost = 0.0
    crash_count             = 0
    prev_action             = None
    real_streak             = 0

    block_deltas = []

    for block_idx, t0 in enumerate(range(0, T_HORIZON, DECISION_PERIOD)):
        block_deltas.append(np.zeros(EPISODES_MAX))

        a0 = np.random.randint(n_actions)
        week_actions.append(a0)

        real_streak = update_streak(prev_action, real_streak, a0)
        prev_action = a0

        for day in range(t0, t0 + DECISION_PERIOD):
            env_true.set_intervention(a0)
            x_real, icu_real = env_true.step(x_real, day)
            real_streak = update_streak(prev_action, real_streak, a0)

            rl_icu.append(icu_real)
            total_icu_load += icu_real

            cost = intervention_cost(a0, real_streak)
            total_intervention_cost += COST_SCALE * cost

            if icu_real > CRASH_THRESHOLD:
                crash_count += 1
                inst_reward = CRASH_PENALTY
            else:
                inst_reward = -(ICU_SCALE * icu_real + COST_SCALE * cost)
            total_reward += inst_reward

        data_seg = np.asarray(rl_icu[-DECISION_PERIOD:]).reshape(-1, 1)
        param    = action2param[a0]
        prior    = dists.StructDist({param: beta_priors[param],
                                     'p_icu': p_icu_prior})
        fk       = ssp.SMC2(ssm_cls=SEIR_ICU, prior=prior, data=data_seg,
                            init_Nx=100, len_chain=3, ar_to_increase_Nx=0.10,
                            smc_options=dict(qmc=False))
        alg = particles.SMC(fk=fk, N=500, verbose=False)
        for _ in alg:
            pass
        θ, W = alg.X.theta[param], alg.W
        posterior_storage[param].append((θ.copy(), W.copy()))
        m = np.sum(θ * W)
        v = np.sum(W * (θ - m) ** 2)
        beta_priors[param] = dists.Gamma(a=m * m / v, b=m / v)
        gc.collect()

    return (
        rl_icu,
        week_actions,
        total_reward,
        block_deltas,
        total_icu_load,
        total_intervention_cost,
        crash_count
    )

# ------------------------------------------------------------------
# 9.  Parallel driver & plotting (Algorithm 2 only)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from itertools import product

    # Hyperparameter grids (kept trivial so they match the paper exactly)
    epsilons    = [epsilon_start_default]  # ε0 = 0.20
    N_eps       = [n_episodes]            # E = 80,000
    N_bins_lst  = [N_bins_icu]           # G = 200
    Cs          = [C_const]              # C = 45
    cost_scales = [0.2, 0.5, 0.8]        # κ_so-ec ∈ {0.2, 0.5, 0.8}

    n_replicates   = 10
    K = 25  # number of θ draws per block

    for comb_id, (ne, eps0, Cc, nb, cost_scale) in enumerate(
            product(N_eps, epsilons, Cs, N_bins_lst, cost_scales)):

        label = f"cost_{cost_scale}"

        # Override globals for this combination (matches the paper)
        n_episodes            = ne
        epsilon_start_default = eps0
        C_const               = Cc
        N_bins_icu            = nb
        COST_SCALE            = cost_scale
        icu_bins              = np.geomspace(1, 6000, N_bins_icu + 1)
        EPISODES_MAX          = n_episodes
        C_BASE                = C_const
        EPS_START_BASE        = epsilon_start_default
        PLAN_HORIZON          = 100  # H = 100

        # Generate reproducible seeds for RL
        root_rl   = np.random.SeedSequence(10000 + comb_id)
        seeds_rl  = [int(s.generate_state(1)[0]) for s in root_rl.spawn(n_replicates)]

        # Run RL replicas in parallel (Algorithm 2 only)
        jobs = [
            delayed(run_one_replicate)(i, seeds_rl[i], K)
            for i in range(n_replicates)
        ]

        with parallel_backend("loky", inner_max_num_threads=1):
            all_results = Parallel(
                n_jobs=n_replicates,
                prefer="processes"
            )(jobs)

        # Unpack RL results
        (all_rl_icus,
         all_actions,
         total_rewards,
         all_q_avg_lists,
         all_block_deltas,
         all_icu_loads,
         all_interv_costs,
         all_crash_counts,
         all_ci_widths) = map(list, zip(*all_results))

        # ---------- 1) Posterior credible-interval shrinkage ----------
        ci_arrs = {
            pname: np.stack([ci_dict[pname] for ci_dict in all_ci_widths], axis=0)
            for pname in action2param.values()
        }
        mean_ci = {p: arr.mean(axis=0) for p, arr in ci_arrs.items()}
        n_blocks = len(next(iter(mean_ci.values())))

        plt.figure(figsize=(8, 5))
        for pname, widths in mean_ci.items():
            plt.plot(range(n_blocks), widths, label=pname)
        plt.xlabel('Decision block index')
        plt.ylabel('90% CI width (β)')
        plt.title(f'CI Shrinkage (π², {label})')
        plt.legend(title='Parameter')
        plt.tight_layout()
        plt.savefig(f'{label}_CI_shrinkage_pi2.png', dpi=300)
        plt.close()

        # ---------- 2) RL ensemble vs observed ICU ----------
        plt.figure(figsize=(12, 6))
        plt.scatter(range(T_HORIZON), obs_icu, s=15, color='k', alpha=0.6, label='Observed ICU')
        for traj in all_rl_icus:
            plt.plot(range(T_HORIZON), traj, color='C1', alpha=0.2)
        mean_rl = np.mean(all_rl_icus, axis=0)
        plt.plot(range(T_HORIZON), mean_rl, color='C1', lw=2, label='Mean RL ICU')
        std_rl = np.std(all_rl_icus, axis=0, ddof=1)
        ci95_rl = 1.96 * std_rl / np.sqrt(len(all_rl_icus))
        lower_rl = np.clip(mean_rl - ci95_rl, 0, None)
        upper_rl = mean_rl + ci95_rl

        mean_reward_rl = np.mean(total_rewards)
        std_reward_rl  = np.std(total_rewards, ddof=1)
        ci95_reward_rl = 1.96 * std_reward_rl / np.sqrt(len(total_rewards))
        mean_total_icu  = np.mean(all_icu_loads)
        std_total_icu   = np.std(all_icu_loads, ddof=1)
        ci95_total_icu  = 1.96 * std_total_icu / np.sqrt(len(all_icu_loads))
        real_total_icu  = obs_icu.sum()

        plt.fill_between(range(T_HORIZON), lower_rl, upper_rl,
                         color='C1', alpha=0.3, label='95% CI')
        plt.xlabel('Day')
        plt.ylabel('ICU occupancy')
        plt.title(f'SMC² + Q-learning (π²): ICU Trajectories vs Observed ({label})')
        ax = plt.gca()
        textstr = (
            f"Mean total reward:       {mean_reward_rl:,.0f} ± {ci95_reward_rl:,.0f}\n"
            f"Mean total ICU load:     {mean_total_icu:,.0f} ± {ci95_total_icu:,.0f}\n"
            f"Observed total ICU load: {real_total_icu:,.0f}"
        )
        ax.text(
            0.98, 0.98, textstr,
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
        )
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{label}_RL_ensemble_ICU_pi2.png', dpi=300)
        plt.close()

        # ---------- 3) RL Q-convergence heatmap ----------
        data = np.array(all_block_deltas)            # shape: (reps, blocks, episodes)
        mean_data = data.mean(axis=0)                # shape: (blocks, episodes)
        episodes = np.arange(1, mean_data.shape[1] + 1)
        log_mean = np.log10(mean_data + 1e-12)
        plt.figure(figsize=(10, 6))
        plt.imshow(log_mean, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label=r'$\log_{10}\max|\Delta Q|$')
        plt.xlabel('Episode')
        plt.ylabel('Block index')
        plt.title(f'RL Q-Convergence Heatmap (π², {label})')
        plt.tight_layout()
        plt.savefig(f'{label}_Q_convergence_heatmap_pi2.png', dpi=300)
        plt.close()

        # ---------- 4) RL representative convergence slices ----------
        reps = [0, n_blocks // 2, n_blocks - 1]
        plt.figure(figsize=(8, 5))
        for b in reps:
            mean_block  = data[:, b, :].mean(axis=0)
            plt.plot(episodes, mean_block, label=f'Block {b}')
        plt.yscale('log')
        plt.xlabel('Episode')
        plt.ylabel(r'$\max|\Delta Q|$')
        plt.title(f'RL Convergence Representative Blocks (π², {label})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{label}_Q_convergence_repr_pi2.png', dpi=300)
        plt.close()

        # ---------- 5) RL empirical vs PAC bound (example block) ----------
        j = np.arange(1, mean_data.shape[1] + 1)
        delta = 0.05
        S, A = N_bins_icu, n_actions
        eps_pac = np.sqrt(2 * np.log(2 * S * A * j**2 / delta) / j)
        plt.figure(figsize=(8, 5))
        # use block index 24 ~ “block 25” as in the paper
        plt.plot(j, mean_data[24], label='Empirical (Block 25)')
        plt.plot(j, eps_pac, linestyle='--', label='PAC bound')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Planning episode')
        plt.ylabel(r'$\max|\Delta Q|$')
        plt.title(f'Empirical vs. PAC Bound (π², {label})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{label}_PAC_bound_pi2.png', dpi=300)
        plt.close()

        # ---------- 6) Print summary statistics ----------
        print(f"\n=== κ_so-ec = {cost_scale} (Ne={ne}, ε₀={eps0}, C={Cc}, bins={nb}, H={PLAN_HORIZON}) ===")
        mean_rl_reward = mean_reward_rl
        std_rl_reward  = std_reward_rl
        print(f"  RL reward: {mean_rl_reward:.0f} ± {ci95_reward_rl:.0f}")
        print(f"  Mean crash days per replicate: {np.mean(all_crash_counts):.2f}")
        print("  RL actions per replicate:")
        for i, acts in enumerate(all_actions, start=1):
            print(f"    Replicate {i}: {acts}")
