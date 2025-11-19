

"""
Algorithm 1 – Finite-Horizon MDP for ICU-Thresholded Policy π_I
=================================================================

This script implements the threshold policy optimisation (Algorithm 1):

  • National-scale SEIR–ICU–VU model with vaccination (same as Algorithm 2).
  • Bayesian filtering via SMC² with N = 500 parameter particles.
  • Threshold policy π_I^φ mapping ICU level to actions a ∈ {0,1,2,3}.
  • Coarse-to-fine grid search over ICU thresholds φ = (τ1, τ2, τ3).
  • Planning horizon H = 100 days, decision period Δ = 10 days.
  • Discount factor γ = 0.95 inside planning.
  • Reward:
        r_t(y_t, a_t) =
            -P                     if y_t > T_crash
            -κ_icu y_t - κ_so-ec C(a_t, ℓ_t)   otherwise
    with:
        T_crash = 5000 beds,
        P = -1e5,
        κ_icu = 1,
        κ_so-ec ∈ {0.2, 0.5, 0.8} (looped over in main).
"""

import os, gc, time, warnings, sys, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import Optional, Tuple, Dict, List

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
# 2.  Hyper-parameters (aligned with Section 6.2)
# ------------------------------------------------------------------
ICU_SCALE       = 1.0          # κ_icu = 1
COST_SCALE      = 0.2          # κ_so-ec (overridden per run in main)
GAMMA           = 0.95         # discount factor γ
DECISION_PERIOD = 10           # Δ = 10

n_actions    = 4
T_HORIZON    = 300             # total horizon
n_replicates = 10              # R = 10


# Crash penalty settings (match Section 6.2)
CRASH_THRESHOLD = 5000         # T_crash = 5000 ICU beds
CRASH_PENALTY   = -1e5         # P = -10^5

# Hard lower-bounds on thresholds 
MIN_T1 = 50
MIN_T2 = 220
MIN_T3 = 900

# Threshold search grid
ICU_MAX_EST      = 6_000
GRID_POINTS      = 20
PLAN_HORIZON     = 100         # H = 100 days (planning horizon)
POST_SAMPLES_BASE = 25         # K = 25 posterior draws per optimisation round
MIN_POST_SAMPLES  = 25

# Local-search & SMC² tuning
NEIGH_PCT         = 0.1
CANDIDATE_MIN     = 20
N_PARTICLES_FULL  = 500        # N = 500 SMC² particles (early horizon)
N_PARTICLES_HALF  = 500        # N = 500 SMC² particles (later horizon)
BATCHSIZE_TRIPLES = 25

grid    = np.geomspace(10, ICU_MAX_EST, GRID_POINTS)
triples = list(itertools.combinations(grid, 3))

# enforce the hard lower-bounds on τ₂ and τ₃
triples = [
    (t1, t2, t3)
    for (t1, t2, t3) in triples
    if t1 >= MIN_T1 and t2 >= MIN_T2 and t3 >= MIN_T3
]

# ------------------------------------------------------------------
# 3.  Real data, vaccinations & surrogate generator
# ------------------------------------------------------------------
T_HORIZON = 300

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
# Posterior-centred log-normal surrogate for β's (Prior from SMC^2)
# ------------------------------------------------------------------
beta_nominal = np.array([0.318, 0.216, 0.060, 0.032])

def sample_beta_surrogate(day: int):
    CV_max = np.array([0.50,0.50,0.50,0.50])
    CV_min = np.array([0.05,0.05,0.05,0.05])
    frac   = day / T_HORIZON
    cv_vec = CV_max*(1-frac) + CV_min*frac
    sigma  = np.sqrt(np.log(1 + cv_vec**2))
    raw    = np.random.lognormal(np.log(beta_nominal), sigma)
    return np.sort(raw)[::-1]

# ------------------------------------------------------------------
# 4.  Mapping RL-action → β parameter name
# ------------------------------------------------------------------
action2param = {0: 'beta1', 1: 'beta2', 2: 'beta3', 3: 'beta4'}

# ------------------------------------------------------------------
# 5.  SEIR-ICU RL environment
# ------------------------------------------------------------------
class SEIR_ICU_RL:
    """SEIR-ICU-Vacc RL environment with combined-force infection."""
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
        self.current_beta = [self.beta1, self.beta2, self.beta3, self.beta4][a]

    def PX0(self) -> np.ndarray:
        x0 = np.zeros(14, dtype=float)
        x0[0] = 59_947_215
        x0[1:5] = [8_108, 6_932, 37_665, 80]
        return x0

    def _ensure_buffers(self, N: int):
        if self.I_d1 is None or len(self.I_d1) != N:
            self.I_d1 = np.full(N, 5.)
            self.I_d2 = np.full(N, 5.)

    def step(self, xp: np.ndarray, t: int):
        """One-day transition with combined-force infection."""
        S, E, I, R, C = xp[0], xp[1], xp[2], xp[3], xp[4]
        V              = xp[5:10].copy()
        EV, IV, RV, ICUv = xp[10], xp[11], xp[12], xp[13]

        self._ensure_buffers(1)

        # ---- vaccinations
        v1 = min(self.daily_first[t], int(S))
        S  -= v1
        V[0] += v1

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
        lam = β * total_inf / total_pop if total_pop > 0 else 0.

        newE0 = np.random.binomial(int(S), 1 - np.exp(-lam))

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
    """SEIR-ICU-Vacc state-space model for SMC² (combined-force infection)."""

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
        x0 = np.zeros(14)
        x0[0] = 59_976_749
        x0[1:5] = [3_321, 9_120, 24_707, 80]
        return dists.Dirac(loc=x0)

    def _ensure_buffers(self, N):
        if self.I_d1 is None or self.I_d1.size != N:
            self.I_d1 = np.full(N, 5.)
            self.I_d2 = np.full(N, 5.)

    def PX(self, t, xp):
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
            lam = np.where(N0+N1 > 0, β * (I + IV) / (N0 + N1), 0.)

            newE0 = np.random.binomial(S.astype(int), 1 - np.exp(-lam))
            S    -= newE0
            E    += newE0

            newEV = np.zeros_like(S)
            for i in range(5):
                prob = 1.0 - np.exp(-lam * (1.0 - m.eps[i]))
                ne   = np.random.binomial(V[i].astype(int), prob)
                V[i] -= ne
                newEV += ne
                EV += newEV

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
# 7.  Helper functions
# ------------------------------------------------------------------
N_bins_icu = 200
icu_bins   = np.geomspace(1, 7_000, N_bins_icu + 1)

def discretize(x: np.ndarray) -> int:
    icu = x[4] + x[13] if x.ndim == 1 else x[:,4] + x[:,-1]
    return np.clip(np.digitize(icu, icu_bins) - 1, 0, N_bins_icu - 1)

def intervention_cost(a: int, streak: int) -> float:
    if a == 1:
        return 50 * np.log1p(streak)
    elif a == 2:
        return 200 * streak
    elif a == 3:
        return 800 * streak
    else:
        return 0.0

def choose_action(icu_now: float, thr: Tuple[float,float,float]) -> int:
    """Map current ICU occupancy to an action index ∈ {0,1,2,3}."""
    t1, t2, t3 = thr
    if icu_now < t1:
        return 0
    elif icu_now < t2:
        return 1
    elif icu_now < t3:
        return 2
    else:
        return 3

def action_tier(a: int) -> int:
    return 0 if a < 2 else 1

def update_streak(prev_a: Optional[int], prev_streak: int, new_a: int) -> int:
    if new_a == 0:
        return 0
    if prev_a is None or prev_a == 0:
        return 1
    if action_tier(new_a) < action_tier(prev_a):
        return 1
    return prev_streak + 1

# ------------------------------------------------------------------
# 7a.  Threshold evaluation & SMC² beta-posterior helper
# ------------------------------------------------------------------
def evaluate_thresholds(
    thr: Tuple[float,float,float],
    t0: int,
    x_init: np.ndarray,
    beta_draws: np.ndarray
) -> float:
    """
    Monte-Carlo estimate of discounted return over the next planning window:

        C_hat_ϕ = (1/K) Σ_k Σ_{t=t0}^{t0+H-1} γ^{t-t0} r_t

    with r_t defined using CRASH_THRESHOLD and CRASH_PENALTY.
    """
    R_tot = 0.0
    for b1, b2, b3, b4 in beta_draws:
        env_sim = SEIR_ICU_RL(daily_first, daily_second)
        env_sim.beta1, env_sim.beta2, env_sim.beta3, env_sim.beta4 = b1, b2, b3, b4

        x      = x_init.copy()
        prev_a = None
        streak = 0
        R_blk  = 0.0

        for day in range(t0, min(t0 + PLAN_HORIZON, T_HORIZON)):
            icu_now = x[4] + x[13]
            a = choose_action(icu_now, thr)
            streak = update_streak(prev_a, streak, a)
            prev_a = a

            env_sim.set_intervention(a)
            x, icu = env_sim.step(x, day)

            if icu > CRASH_THRESHOLD:
                r_t = CRASH_PENALTY
            else:
                r_t = -(ICU_SCALE * icu + COST_SCALE * intervention_cost(a, streak))

            R_blk += (GAMMA ** (day - t0)) * r_t

        R_tot += R_blk

    return R_tot / len(beta_draws)

def evaluate_thresholds_batch(
    thr_batch: List[Tuple[float,float,float]],
    t0: int,
    x_init: np.ndarray,
    beta_draws: np.ndarray
) -> np.ndarray:
    rewards = np.empty(len(thr_batch))
    for i, thr in enumerate(thr_batch):
        rewards[i] = evaluate_thresholds(thr, t0, x_init, beta_draws)
    return rewards

def smc2_draw_beta(
    data_seg: np.ndarray,
    beta_priors: Dict[str, dists.ProbDist],
    p_icu_prior: dists.ProbDist,
    n_particles: int
):
    """Run one SMC² update per β-parameter and return updated priors."""
    bp_new     = {}
    post_dict  = {}

    for pname, prior_dist in beta_priors.items():
        prior = dists.StructDist({pname: prior_dist, 'p_icu': p_icu_prior})
        fk = ssp.SMC2(
            ssm_cls=SEIR_ICU,
            prior=prior,
            data=data_seg,
            init_Nx=150,
            len_chain=3,
            ar_to_increase_Nx=0.10,
            smc_options=dict(qmc=False)
        )
        alg = particles.SMC(fk=fk, N=n_particles, verbose=False)
        for _ in alg:
            pass

        θ = alg.X.theta[pname]
        W = alg.W
        post_dict[pname] = (θ.copy(), W.copy())

        m = np.sum(θ * W)
        v = np.sum(W * (θ - m) ** 2)
        bp_new[pname] = dists.Gamma(a=m*m / v, b=m / v)

    return None, bp_new, p_icu_prior, post_dict

# ------------------------------------------------------------------
# 8.  ONE replicate with threshold search & custom streak logic
# ------------------------------------------------------------------
def action_group(a: int) -> int:
    return 0 if a < 2 else 1

def run_one_replicate(rep_id: int, seed: Optional[int] = None):
    if seed is None:
        seed = (int(time.time() * 1000) + rep_id) % 2**32
    np.random.seed(seed)

    # Hidden ground-truth environment
    theta_star = sample_beta_surrogate(T_HORIZON)
    env_true   = SEIR_ICU_RL(daily_first, daily_second)
    env_true.beta1, env_true.beta2, env_true.beta3, env_true.beta4 = theta_star
    x_real = env_true.PX0()

    # Priors & posterior storage (simple Gamma(1,1) for β; Beta for p_icu)
    beta_priors = {p: dists.Gamma(a=1.0, b=1.0) for p in action2param.values()}
    p_icu_prior = dists.Beta(50., 50. / 0.00106 - 50.)
    posterior_storage = {p: [] for p in action2param.values()}

    thr_icu_path: List[float] = []
    epoch_thrs: List[Tuple[float, float, float]] = []
    weekly_actions: List[int] = []
    total_reward = 0.0
    prev_action, real_streak = None, 0
    used_actions_prev: set[int] = set()
    pnames = ['beta1', 'beta2', 'beta3', 'beta4']

    for block_idx, t0 in enumerate(range(0, T_HORIZON, DECISION_PERIOD)):
        # ① Posterior update (from previous 10 days)
        if t0 > 0:
            data_seg   = np.asarray(thr_icu_path[-DECISION_PERIOD:]).reshape(-1, 1)
            n_particles = N_PARTICLES_FULL if t0 < 100 else N_PARTICLES_HALF
            _, bp_new, p_icu_prior, post_dict = smc2_draw_beta(
                data_seg, beta_priors, p_icu_prior, n_particles
            )
            for a in used_actions_prev:
                pname = pnames[a]
                posterior_storage[pname].append(post_dict[pname])
                beta_priors[pname] = bp_new[pname]

        # ② Draw posterior β samples (θ^(k))
        beta_draws = np.vstack([
            sample_beta_surrogate(t0)
            for _ in range(POST_SAMPLES_BASE)
        ])
        for j, pname in enumerate(pnames):
            if posterior_storage[pname]:
                θs, W = posterior_storage[pname][-1]
                beta_draws[:, j] = np.random.choice(
                    θs, size=POST_SAMPLES_BASE, p=W / W.sum()
                )
        beta_draws = np.sort(beta_draws, axis=1)[:, ::-1]

        # ③ Determine candidate threshold triples
        if block_idx % 7 == 0 or not epoch_thrs:
            candidate_triples = triples
        else:
            prev   = np.array(epoch_thrs[-1])
            margin = 0.50 * prev
            step   = 0.10 * prev
            lo, hi = np.maximum(0, prev - margin), prev + margin
            G1 = np.arange(lo[0], hi[0] + step[0]/2, step[0])
            G2 = np.arange(lo[1], hi[1] + step[1]/2, step[1])
            G3 = np.arange(lo[2], hi[2] + step[2]/2, step[2])
            candidate_triples = [
                (a, b, c) for a in G1 for b in G2 for c in G3
                if a <= b <= c and b >= MIN_T2 and c >= MIN_T3
            ]
        if len(candidate_triples) < CANDIDATE_MIN:
            candidate_triples = triples

        # ④ Evaluate and pick best triple φ*
        best_val = -np.inf
        best_thr = epoch_thrs[-1] if epoch_thrs else triples[0]
        for i in range(0, len(candidate_triples), BATCHSIZE_TRIPLES):
            batch  = candidate_triples[i:i + BATCHSIZE_TRIPLES]
            scores = evaluate_thresholds_batch(batch, t0, x_real.copy(), beta_draws)
            scores += 1e-6 * np.random.randn(len(scores))
            k = np.argmax(scores)
            if scores[k] > best_val:
                best_val = scores[k]
                best_thr = batch[k]

        epoch_thrs.append(best_thr)
        print(f"[Block {block_idx:2d}] selected thresholds = "
              f"({best_thr[0]:.0f}, {best_thr[1]:.0f}, {best_thr[2]:.0f})")

        # ⑤ Deploy best_thr over next Δ = 10 days in real env
        used_actions_curr = set()
        curr_a = None
        for day in range(t0, t0 + DECISION_PERIOD):
            icu_now = x_real[4] + x_real[13]
            if curr_a is None or (day - t0) % DECISION_PERIOD == 0:
                curr_a = choose_action(icu_now, best_thr)

            used_actions_curr.add(curr_a)
            env_true.set_intervention(curr_a)
            x_real, icu_real = env_true.step(x_real, day)

            thr_icu_path.append(icu_real)
            weekly_actions.append(curr_a)

            real_streak = update_streak(prev_action, real_streak, curr_a)
            prev_action = curr_a

            if icu_real > CRASH_THRESHOLD:
                r_t = CRASH_PENALTY
            else:
                r_t = -(ICU_SCALE * icu_real +
                        COST_SCALE * intervention_cost(curr_a, real_streak))
            total_reward += r_t

        used_actions_prev = used_actions_curr
        gc.collect()

    return thr_icu_path, epoch_thrs, weekly_actions, total_reward

# ------------------------------------------------------------------
# 9.  Parallel driver & plotting
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    import numpy as np
    import matplotlib.pyplot as plt

    # Match Section 6.2: κ_so-ec ∈ {0.2, 0.5, 0.8}
    cost_scales  = [0.2, 0.5, 0.8]
    n_replicates = 10

    for cost in cost_scales:
        global COST_SCALE
        COST_SCALE = cost

        print(f"\n=== Running threshold policy with κ_so-ec = {cost:.2f} ===")
        print(f"Running {n_replicates} replicates on {os.cpu_count()} CPU cores…")
        sys.stdout.flush()

        root_ss    = np.random.SeedSequence()
        child_seqs = root_ss.spawn(n_replicates)

        def safe_run(rep, seed):
            try:
                print(f"[INFO] Replicate {rep:2d} starting…")
                sys.stdout.flush()
                res = run_one_replicate(rep, seed)
                print(f"[INFO] Replicate {rep:2d} finished.")
                sys.stdout.flush()
                return res
            except Exception as e:
                print(f"[ERROR] Replicate {rep:2d} failed: {e}")
                import traceback; traceback.print_exc(); sys.stdout.flush()
                return None

        delayed_calls = [
            delayed(safe_run)(rep, int(cs.generate_state(1)[0]))
            for rep, cs in enumerate(child_seqs)
        ]

        with Parallel(min(n_replicates, os.cpu_count()), backend="threading") as parallel:
            results = parallel(
                tqdm(delayed_calls, total=n_replicates,
                     desc=f"κ_so-ec = {cost:.2f}")
            )

        results = [r for r in results if r is not None]
        if not results:
            sys.exit("All replicates failed – terminating.")

        all_rl_icus, all_thrs, all_actions, total_rewards = map(list, zip(*results))

        # print summary statistics
        for i, acts in enumerate(all_actions, start=1):
            print(f"Replicate {i:2d} weekly actions:", acts)
        for i, ths in enumerate(all_thrs, start=1):
            print(f"Replicate {i:2d} epoch thresholds:",
                  [tuple(map(int, t)) for t in ths])
        for i, R in enumerate(total_rewards, start=1):
            print(f"Replicate {i:2d} total reward : {R:,.0f}")

        mean_reward = np.mean(total_rewards)
        print(f"\nMean total reward across replicates: {mean_reward:,.0f}")

        # ---------- plotting ensemble vs observed --------------------
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.scatter(range(T_HORIZON), obs_icu, s=15, color='k',
                   alpha=0.6, label='Observed ICU')

        for traj in all_rl_icus:
            ax.plot(range(T_HORIZON), traj, color='C1', alpha=0.20)

        mean_rl = np.mean(all_rl_icus, axis=0)
        ax.plot(range(T_HORIZON), mean_rl, color='C1', lw=2,
                label='Mean threshold ICU')

        std_rl   = np.std(all_rl_icus, axis=0, ddof=1)
        ci95_rl  = 1.96 * std_rl / np.sqrt(len(all_rl_icus))
        lower_rl = np.clip(mean_rl - ci95_rl, 0, None)
        upper_rl = mean_rl + ci95_rl
        ax.fill_between(range(T_HORIZON), lower_rl, upper_rl,
                        color='C1', alpha=0.30, label='95 % CI')

        sim_totals = np.sum(all_rl_icus, axis=1)
        mean_tot   = np.mean(sim_totals)
        std_tot    = np.std(sim_totals, ddof=1)
        ci95_tot   = 1.96 * std_tot / np.sqrt(len(sim_totals))
        real_total = obs_icu.sum()
        mean_reward = np.mean(total_rewards)
        std_reward  = np.std(total_rewards, ddof=1)
        ci95_reward = 1.96 * std_reward / np.sqrt(len(total_rewards))

        ax.set_xlabel('Day')
        ax.set_ylabel('ICU occupancy')
        ax.set_title(f'SMC² + Threshold Control (κ_so-ec = {cost:.2f})')

        txt = (f"Mean total reward : {mean_reward:,.0f} ± {ci95_reward:,.0f}\n"
               f"Mean total ICU    : {mean_tot:,.0f} ± {ci95_tot:,.0f}\n"
               f"Observed ICU load : {real_total:,.0f}")
        ax.text(0.02, 0.65, txt, transform=ax.transAxes,
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.75, lw=0))
        ax.legend(loc='upper right')
        fig.tight_layout()
        fname = f'True_obs_vs_obs_THRESHOLD_cost_{cost:.2f}.png'
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"Plot saved: {fname}")
