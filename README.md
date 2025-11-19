# SEIR–VU model and RL controllers for epidemic control

This repository contains the implementation of **Algorithm 1 (threshold policy πᴵ)** and  
**Algorithm 2 (posterior-averaged Q-learning policy π²)** from the paper:

> *On a Reinforcement Learning Methodology for Epidemic Control, with application to COVID-19*  
> G. Iannucci et al., 2025.

The code implements:

- A national-scale SEIR–VU model with ICU and vaccination.
- Sequential Monte Carlo squared (SMC²) with 500 parameter particles.
- Two control strategies under partial observability:
  - **Algorithm 1:** ICU-thresholded policy πᴵ with finite-horizon planning.
  - **Algorithm 2:** Posterior-averaged tabular Q-learning (ε-greedy) π².

---

## Repository structure

- `algorithm1_threshold_policy.py`  
  Implementation of **Algorithm 1** (finite-horizon ICU-thresholded policy πᴵ), including:
  - SEIR–VU state-space model (`SEIR_ICU`) and RL environment (`SEIR_ICU_RL`)
  - SMC² posterior updates for the transmission parameters β₁,…,β₄
  - Coarse-to-fine grid search over ICU thresholds (τ₁, τ₂, τ₃)
  - Planning horizon **H = 100** days, decision period **Δ = 10**
  - Experiments for socio-economic weight κ_so-ec ∈ {0.2, 0.5, 0.8}

- `algorithm2_posterior_q.py`  
  Implementation of **Algorithm 2** (posterior-averaged Q-learning policy π²), including:
  - The same SEIR–VU model (`SEIR_ICU`) and RL environment (`SEIR_ICU_RL`)
  - SMC² posterior updates
  - Domain-randomised “true” β trajectories
  - Slice-based tabular Q-learning with discount factor **γ = 0.95**
  - Experiments for κ_so-ec ∈ {0.2, 0.5, 0.8}

- `requirements.txt`  
  Python dependencies required to run the experiments (SMC² + RL).

---

## Installation

Create and activate a virtual environment (recommended), then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Run
```bash
python algorithm1_threshold_policy.py
python algorithm2_posterior_q.py
```
