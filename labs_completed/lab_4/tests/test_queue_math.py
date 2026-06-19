# -*- coding: utf-8 -*-
"""Unit tests for M/M/1 analytical formulas."""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.queue_math import mm1_metrics, system_metrics, mean_response_time, equal_shares


# ---- mm1_metrics ----

def test_mm1_basic():
    """λ=5, μ=10 → ρ=0.5, W=1/(10-5)=0.2, Lq=0.5"""
    m = mm1_metrics(5.0, 10.0)
    assert m.stable is True
    assert abs(m.rho - 0.5) < 1e-9
    assert abs(m.w - 0.2) < 1e-9
    assert abs(m.l_q - 0.5) < 1e-9


def test_mm1_high_utilization():
    """λ=9, μ=10 → ρ=0.9, W=1/(10-9)=1.0 s, Lq=ρ²/(1-ρ)=8.1"""
    m = mm1_metrics(9.0, 10.0)
    assert m.stable is True
    assert abs(m.rho - 0.9) < 1e-9
    assert abs(m.w - 1.0) < 1e-9
    assert abs(m.l_q - 8.1) < 1e-9


def test_mm1_unstable():
    """ρ >= 1 → system unstable, W and Lq are inf"""
    m = mm1_metrics(10.0, 10.0)
    assert m.stable is False
    assert math.isinf(m.w)
    assert math.isinf(m.l_q)

    m2 = mm1_metrics(15.0, 10.0)
    assert m2.stable is False


def test_mm1_zero_load():
    """λ=0 → ρ=0, W=1/μ, Lq=0"""
    m = mm1_metrics(0.0, 5.0)
    assert m.stable is True
    assert m.rho == 0.0
    assert abs(m.w - 0.2) < 1e-9
    assert m.l_q == 0.0


def test_mm1_invalid_mu():
    with pytest.raises(ValueError):
        mm1_metrics(1.0, 0.0)
    with pytest.raises(ValueError):
        mm1_metrics(1.0, -1.0)


def test_mm1_invalid_lambda():
    with pytest.raises(ValueError):
        mm1_metrics(-1.0, 5.0)


def test_mm1_server_id():
    m = mm1_metrics(2.0, 10.0, server_id=3)
    assert m.server_id == 3


# ---- system_metrics ----

def test_system_metrics_basic():
    mets = system_metrics(100.0, [100.0, 50.0], [0.5, 0.5])
    assert len(mets) == 2
    assert mets[0].server_id == 0
    assert mets[1].server_id == 1
    # server 0: λ=50, μ=100 → ρ=0.5
    assert abs(mets[0].rho - 0.5) < 1e-9
    # server 1: λ=50, μ=50 → ρ=1.0 → unstable
    assert mets[1].stable is False


def test_system_metrics_shares_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        system_metrics(100.0, [100.0, 100.0], [0.3, 0.3])


def test_system_metrics_length_mismatch():
    with pytest.raises(ValueError):
        system_metrics(100.0, [100.0, 100.0, 100.0], [0.5, 0.5])


# ---- mean_response_time ----

def test_mean_response_time():
    """λ=100, μ=[200,200], shares=[0.5,0.5] → each ρ=0.25 → W=1/150"""
    mets = system_metrics(100.0, [200.0, 200.0], [0.5, 0.5])
    w_avg = mean_response_time(mets, [0.5, 0.5])
    # W_i = 1/(200-50) = 1/150
    assert abs(w_avg - 1.0 / 150.0) < 1e-9


def test_mean_response_time_unstable():
    mets = system_metrics(100.0, [50.0, 200.0], [1.0, 0.0])
    w_avg = mean_response_time(mets, [1.0, 0.0])
    assert math.isinf(w_avg)


# ---- equal_shares ----

def test_equal_shares():
    sh = equal_shares(4)
    assert len(sh) == 4
    assert abs(sum(sh) - 1.0) < 1e-9
    assert all(abs(s - 0.25) < 1e-9 for s in sh)


# ---- parametrized: mm1_metrics ----

@pytest.mark.parametrize("lam,mu,expected_rho,expected_w", [
    (5.0,  10.0, 0.5,   0.2),
    (9.0,  10.0, 0.9,   1.0),
    (1.0, 100.0, 0.01,  1/99),
    (50.0, 80.0, 0.625, 1/30),
])
def test_mm1_parametrized(lam, mu, expected_rho, expected_w):
    m = mm1_metrics(lam, mu)
    assert m.stable is True
    assert abs(m.rho - expected_rho) < 1e-6
    assert abs(m.w - expected_w) < 1e-6


# ---- parametrized: mean_response_time ----

@pytest.mark.parametrize("lam,mu_list,shares,expected_w", [
    (100.0, [200.0, 200.0], [0.5, 0.5], 1.0 / 150.0),
    (60.0,  [100.0, 100.0], [0.5, 0.5], 1.0 / 70.0),
    (40.0,  [100.0, 60.0],  [0.6, 0.4], 0.6 / 76.0 + 0.4 / 44.0),
])
def test_mean_response_time_parametrized(lam, mu_list, shares, expected_w):
    mets = system_metrics(lam, mu_list, shares)
    w_avg = mean_response_time(mets, shares)
    assert abs(w_avg - expected_w) < 1e-9
