# SEIR–VU model and posterior-averaged Q-learning for epidemic control

This repository contains the implementation of **Policy π² (Algorithm 2)** from the paper:

> On a Reinforcement Learning Methodology for Epidemic Control, with application to COVID-19  
> G. Iannucci et al., 2025.

The code implements:
- A national-scale SEIR–VU model with ICU and vaccination.
- Sequential Monte Carlo squared (SMC²) with 500 parameter particles.
- Posterior-averaged tabular Q-learning (ε-greedy) for epidemic control under partial observability.

## Structure

- `algorithm2_posterior_q.py`  
  Complete implementation of Algorithm 2 (posterior-averaged Q-learning, policy π²), including:
    - SEIR–VU state-space model (`SEIR_ICU`) and RL environment (`SEIR_ICU_RL`)
    - SMC² posterior updates
    - Domain-randomised “true” β trajectories
    - Slice-based tabular Q-learning with discount factor γ = 0.95
    - Experiments for κ_so-ec ∈ {0.2, 0.5, 0.8}

- `requirements.txt`  
  Python dependencies needed to run the experiments.

## Installation

Create and activate a virtual environment (recommended), then run:



```bash
pip install -r requirements.txt

## Run
python algorithm2_posterior_q.py

