# -*- coding: utf-8 -*-
"""Unit tests for LP and NLP optimizers."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.lp_optimizer import lp_balance, lp_response_proxy, nlp_true_response


MU = [100.0, 80.0, 60.0]
LAMBDA = 150.0
RHO_MAX = 0.90


# ---- lp_balance ----

def test_lp_balance_returns_optimal():
    res = lp_balance(LAMBDA, MU, RHO_MAX)
    assert res.solver_status == "Optimal"
    assert res.formulation == "balanced"


def test_lp_balance_shares_sum_to_one():
    res = lp_balance(LAMBDA, MU, RHO_MAX)
    assert abs(sum(res.shares) - 1.0) < 1e-4


def test_lp_balance_utilizations_equal():
    """Optimal balanced allocation: all ρ_i = z (min-max property)."""
    res = lp_balance(LAMBDA, MU, RHO_MAX)
    rhos = [LAMBDA * x / mu for x, mu in zip(res.shares, MU)]
    # all non-zero servers should have equal utilisation ≈ z
    nonzero_rhos = [r for r, x in zip(rhos, res.shares) if x > 1e-6]
    if len(nonzero_rhos) > 1:
        assert max(nonzero_rhos) - min(nonzero_rhos) < 1e-4


def test_lp_balance_respects_rho_max():
    res = lp_balance(LAMBDA, MU, RHO_MAX)
    rhos = [LAMBDA * x / mu for x, mu in zip(res.shares, MU)]
    assert all(r <= RHO_MAX + 1e-5 for r in rhos)


def test_lp_balance_symmetric_servers():
    """Equal servers → equal shares."""
    res = lp_balance(60.0, [100.0, 100.0, 100.0], 0.9)
    for sh in res.shares:
        assert abs(sh - 1.0 / 3.0) < 1e-4


def test_lp_balance_two_servers():
    res = lp_balance(80.0, [100.0, 100.0], 0.9)
    assert abs(res.shares[0] - 0.5) < 1e-4
    assert abs(res.shares[1] - 0.5) < 1e-4


# ---- lp_response_proxy ----

def test_lp_response_proxy_optimal():
    res = lp_response_proxy(LAMBDA, MU, RHO_MAX)
    assert res.solver_status == "Optimal"
    assert res.formulation == "response_proxy"


def test_lp_response_proxy_shares_sum():
    res = lp_response_proxy(LAMBDA, MU, RHO_MAX)
    assert abs(sum(res.shares) - 1.0) < 1e-4


def test_lp_response_proxy_prefers_fast_servers():
    """Faster servers should receive more or equal traffic."""
    res = lp_response_proxy(LAMBDA, MU, RHO_MAX)
    # server 0 (μ=100) should get >= server 2 (μ=60)
    assert res.shares[0] >= res.shares[2] - 1e-4


# ---- nlp_true_response ----

def test_nlp_true_response_feasible():
    res = nlp_true_response(LAMBDA, MU, RHO_MAX)
    assert "Optimal" in res.solver_status
    assert res.formulation == "nlp_true"


def test_nlp_true_response_shares_sum():
    res = nlp_true_response(LAMBDA, MU, RHO_MAX)
    assert abs(sum(res.shares) - 1.0) < 1e-4


def test_nlp_true_response_shares_nonneg():
    res = nlp_true_response(LAMBDA, MU, RHO_MAX)
    assert all(x >= -1e-6 for x in res.shares)


def test_nlp_true_better_than_equal():
    """Optimised allocation should have lower W than equal shares."""
    from core.queue_math import system_metrics, mean_response_time, equal_shares
    res = nlp_true_response(LAMBDA, MU, RHO_MAX)
    opt_mets = system_metrics(LAMBDA, MU, res.shares)
    w_opt = mean_response_time(opt_mets, res.shares)

    eq = equal_shares(len(MU))
    eq_mets = system_metrics(LAMBDA, MU, eq)
    w_eq = mean_response_time(eq_mets, eq)

    assert w_opt <= w_eq + 1e-6


# ---- parametrized: lp_balance on 5 scenarios ----

@pytest.mark.parametrize("lam,mu_list,rho_max", [
    (50,  [100.0, 80.0],              0.90),
    (150, [100.0, 80.0, 60.0],        0.90),
    (200, [200.0, 150.0, 100.0],      0.85),
    (30,  [50.0, 50.0, 50.0, 50.0],   0.90),
    (80,  [100.0, 100.0],             0.95),
])
def test_lp_balance_parametrized(lam, mu_list, rho_max):
    res = lp_balance(lam, mu_list, rho_max)
    assert res.solver_status == "Optimal"
    assert abs(sum(res.shares) - 1.0) < 1e-3
    rhos = [lam * x / mu for x, mu in zip(res.shares, mu_list)]
    assert all(r <= rho_max + 1e-4 for r in rhos)


# ---- parametrized: nlp_true_response on 5 scenarios ----

@pytest.mark.parametrize("lam,mu_list,rho_max", [
    (50,  [100.0, 80.0],              0.90),
    (150, [100.0, 80.0, 60.0],        0.90),
    (200, [200.0, 150.0, 100.0],      0.85),
    (30,  [50.0, 50.0, 50.0, 50.0],   0.90),
    (80,  [100.0, 100.0],             0.95),
])
def test_nlp_true_response_parametrized(lam, mu_list, rho_max):
    from core.queue_math import system_metrics, mean_response_time, equal_shares
    res = nlp_true_response(lam, mu_list, rho_max)
    assert abs(sum(res.shares) - 1.0) < 1e-3

    opt_mets = system_metrics(lam, mu_list, res.shares)
    w_nlp = mean_response_time(opt_mets, res.shares)

    eq = equal_shares(len(mu_list))
    eq_mets = system_metrics(lam, mu_list, eq)
    w_eq = mean_response_time(eq_mets, eq)

    assert w_nlp <= w_eq + 1e-6


# ---- fixture-based: all_formulations_feasible ----

def test_all_formulations_feasible(scenario):
    # все три формулировки находят решение на любом сценарии
    for fn in [lp_balance, lp_response_proxy, nlp_true_response]:
        res = fn(scenario["lambda_total"], scenario["mu_list"], scenario["rho_max"])
        assert abs(sum(res.shares) - 1.0) < 1e-3
