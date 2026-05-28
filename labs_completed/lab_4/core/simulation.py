# -*- coding: utf-8 -*-
"""
SimPy discrete-event simulation of M/M/1 queues for empirical validation.
Runs each server in isolation, returns mean sojourn time.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random

import simpy
import numpy as np


@dataclass
class SimResult:
    server_id: int
    lambda_i: float
    mu_i: float
    n_served: int
    w_sim: float       # simulated mean sojourn time
    w_analytical: float  # analytical W = 1/(μ-λ)


def _run_mm1(
    lambda_i: float,
    mu_i: float,
    duration: float,
    seed: int,
) -> tuple[float, int]:
    """Single M/M/1 simulation run. Returns (mean_sojourn, n_served)."""
    rng = random.Random(seed)
    sojourn: list[float] = []

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)

    def source():
        while True:
            yield env.timeout(rng.expovariate(lambda_i))
            env.process(customer())

    def customer():
        arrive = env.now
        with server.request() as req:
            yield req
            yield env.timeout(rng.expovariate(mu_i))
        sojourn.append(env.now - arrive)

    env.process(source())
    env.run(until=duration)

    mean_w = float(np.mean(sojourn)) if sojourn else float('inf')
    return mean_w, len(sojourn)


def simulate_system(
    lambda_total: float,
    mu_list: List[float],
    shares: List[float],
    duration: float = 5000.0,
    seed: int = 42,
) -> List[SimResult]:
    """
    Simulate each server independently for the given duration.
    Each server i receives λ_i = λ_total * shares[i] arrivals/s.
    """
    results = []
    for i, (x, mu) in enumerate(zip(shares, mu_list)):
        lam_i = lambda_total * x
        if lam_i <= 0:
            results.append(SimResult(i, lam_i, mu, 0, 0.0, 0.0))
            continue
        rho = lam_i / mu
        w_anal = (1.0 / (mu - lam_i)) if rho < 1.0 else float('inf')
        if rho >= 1.0:
            results.append(SimResult(i, lam_i, mu, 0, float('inf'), float('inf')))
            continue
        w_sim, n = _run_mm1(lam_i, mu, duration, seed + i)
        results.append(SimResult(i, lam_i, mu, n, w_sim, w_anal))
    return results
