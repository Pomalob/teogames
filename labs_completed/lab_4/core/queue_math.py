# -*- coding: utf-8 -*-
"""Analytical M/M/1 metrics per server (Erlang / birth-death derivation)."""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class ServerMetrics:
    server_id: int
    lambda_i: float    # effective arrival rate to this server (req/s)
    mu_i: float        # service rate of this server (req/s)
    rho: float         # utilization ρ = λ_i / μ_i
    w: float           # mean sojourn time W = 1/(μ - λ) [seconds]
    l_q: float         # mean queue length L_q = ρ²/(1-ρ)
    l: float           # mean number in system L = ρ/(1-ρ)
    stable: bool       # rho < 1


def mm1_metrics(lambda_i: float, mu_i: float, server_id: int = 0) -> ServerMetrics:
    """
    Compute M/M/1 steady-state characteristics.
    M/M/1: Poisson arrivals λ_i, exponential service μ_i, single server.
    Formulas: W = 1/(μ-λ), L_q = ρ²/(1-ρ), L = ρ/(1-ρ).
    """
    if mu_i <= 0:
        raise ValueError("mu_i must be > 0")
    if lambda_i < 0:
        raise ValueError("lambda_i must be >= 0")

    rho = lambda_i / mu_i
    if rho >= 1.0:
        return ServerMetrics(server_id, lambda_i, mu_i, rho,
                             float('inf'), float('inf'), float('inf'), False)

    w = 1.0 / (mu_i - lambda_i)
    l_q = rho ** 2 / (1.0 - rho)
    l = rho / (1.0 - rho)
    return ServerMetrics(server_id, lambda_i, mu_i, rho, w, l_q, l, True)


def system_metrics(
    lambda_total: float,
    mu_list: list[float],
    shares: list[float],
) -> list[ServerMetrics]:
    """Compute M/M/1 metrics for each server given fractional traffic shares."""
    if len(mu_list) != len(shares):
        raise ValueError("mu_list and shares must have the same length")
    if abs(sum(shares) - 1.0) > 1e-4:
        raise ValueError(f"shares must sum to 1, got {sum(shares):.6f}")
    return [
        mm1_metrics(lambda_total * x, mu, i)
        for i, (x, mu) in enumerate(zip(shares, mu_list))
    ]


def mean_response_time(metrics: list[ServerMetrics], shares: list[float]) -> float:
    """
    Weighted mean sojourn time W_avg = Σ x_i * W_i.
    Returns inf if any server is unstable.
    """
    if any(not m.stable for m in metrics):
        return float('inf')
    return sum(x * m.w for x, m in zip(shares, metrics))


def equal_shares(n: int) -> list[float]:
    """Uniform traffic distribution among n servers."""
    return [1.0 / n] * n
