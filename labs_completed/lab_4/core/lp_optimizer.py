# -*- coding: utf-8 -*-
"""
LP optimization of traffic distribution using PuLP.

Two formulations:
  1. Balanced LP  – minimise max utilisation z (fair allocation)
  2. Response LP  – LP approximation: minimise Σ (x_i / μ_i) (linear proxy)

True nonlinear optimum (minimise true M/M/1 W) is computed via SciPy SLSQP
for numerical comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import warnings

import pulp

# PuLP 3.x выдаёт DeprecationWarning на LpVariable и PULP_CBC_CMD (будут удалены в 4.0).
# Функциональность сохранена; предупреждения подавлены для чистого вывода.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")
import numpy as np
from scipy.optimize import minimize as scipy_minimize


@dataclass
class LPResult:
    shares: List[float]       # optimal x_i values
    objective: float          # LP objective value
    solver_status: str        # "Optimal", etc.
    formulation: str          # "balanced" or "response_proxy"


def lp_balance(
    lambda_total: float,
    mu_list: List[float],
    rho_max: float = 0.90,
) -> LPResult:
    """
    LP 1 – minimise maximum server utilisation (load-balancing fairness).

    Variables: x_i ≥ 0 (traffic share), z ≥ 0 (bottleneck utilisation)
    min  z
    s.t. Σ x_i = 1
         λ * x_i / μ_i ≤ z      ∀ i
         z ≤ rho_max
         x_i ≥ 0
    """
    n = len(mu_list)
    prob = pulp.LpProblem("BalancedLoad", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=1.0) for i in range(n)]
    z = pulp.LpVariable("z", lowBound=0.0, upBound=rho_max)

    prob += z
    prob += pulp.lpSum(x) == 1.0
    for i in range(n):
        prob += lambda_total * x[i] / mu_list[i] <= z

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    shares = [max(0.0, pulp.value(x[i]) or 0.0) for i in range(n)]
    # normalise rounding errors
    total = sum(shares)
    if total > 0:
        shares = [s / total for s in shares]

    return LPResult(
        shares=shares,
        objective=float(pulp.value(z) or 0.0),
        solver_status=pulp.LpStatus[status],
        formulation="balanced",
    )


def lp_response_proxy(
    lambda_total: float,
    mu_list: List[float],
    rho_max: float = 0.90,
) -> LPResult:
    """
    LP 2 – minimise Σ (λ x_i / μ_i²) as a linear proxy for M/M/1 response time.

    For small ρ: W_i ≈ 1/μ_i + ρ_i/μ_i = 1/μ_i + λ x_i/μ_i²
    Dropping the constant 1/μ_i gives this linear proxy.

    min  Σ λ x_i / μ_i²
    s.t. Σ x_i = 1
         λ x_i / μ_i ≤ rho_max    ∀ i
         x_i ≥ 0
    """
    n = len(mu_list)
    prob = pulp.LpProblem("ResponseProxy", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=1.0) for i in range(n)]
    prob += pulp.lpSum(lambda_total * x[i] / mu_list[i] ** 2 for i in range(n))
    prob += pulp.lpSum(x) == 1.0
    for i in range(n):
        prob += lambda_total * x[i] / mu_list[i] <= rho_max

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    shares = [max(0.0, pulp.value(x[i]) or 0.0) for i in range(n)]
    total = sum(shares)
    if total > 0:
        shares = [s / total for s in shares]

    return LPResult(
        shares=shares,
        objective=float(pulp.value(prob.objective) or 0.0),
        solver_status=pulp.LpStatus[status],
        formulation="response_proxy",
    )


def nlp_true_response(
    lambda_total: float,
    mu_list: List[float],
    rho_max: float = 0.90,
) -> LPResult:
    """
    NLP – minimise true M/M/1 weighted response time via SciPy SLSQP.

    Objective: W_avg = Σ x_i / (μ_i - λ x_i)    (convex in x_i)
    """
    n = len(mu_list)
    mu = np.array(mu_list)

    def objective(x):
        return float(np.sum(x / (mu - lambda_total * x)))

    def grad(x):
        return (mu / (mu - lambda_total * x) ** 2).tolist()

    # upper bound per server: x_i ≤ rho_max * μ_i / λ
    ub = [rho_max * mu_list[i] / lambda_total for i in range(n)]
    bounds = [(0.0, min(1.0, u)) for u in ub]

    constraints = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}]

    x0 = [1.0 / n] * n
    res = scipy_minimize(
        objective, x0, jac=grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    shares = [max(0.0, float(v)) for v in res.x]
    total = sum(shares)
    if total > 0:
        shares = [s / total for s in shares]

    return LPResult(
        shares=shares,
        objective=float(res.fun),
        solver_status="Optimal" if res.success else res.message,
        formulation="nlp_true",
    )
