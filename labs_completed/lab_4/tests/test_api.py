# -*- coding: utf-8 -*-
"""Integration tests for FastAPI endpoints."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

SERVERS_3 = [
    {"id": 0, "mu": 100.0},
    {"id": 1, "mu": 80.0},
    {"id": 2, "mu": 60.0},
]


# ---- /health ----

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---- /balance ----

def test_balance_balanced():
    payload = {
        "lambda_total": 150.0,
        "servers": SERVERS_3,
        "rho_max": 0.90,
        "formulation": "balanced",
    }
    r = client.post("/balance", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["formulation"] == "balanced"
    assert data["mean_response_ms"] > 0
    shares_sum = sum(s["share"] for s in data["servers"])
    assert abs(shares_sum - 1.0) < 1e-3


def test_balance_response_proxy():
    payload = {
        "lambda_total": 150.0,
        "servers": SERVERS_3,
        "formulation": "response_proxy",
    }
    r = client.post("/balance", json=payload)
    assert r.status_code == 200
    assert r.json()["formulation"] == "response_proxy"


def test_balance_nlp_true():
    payload = {
        "lambda_total": 150.0,
        "servers": SERVERS_3,
        "formulation": "nlp_true",
    }
    r = client.post("/balance", json=payload)
    assert r.status_code == 200
    assert r.json()["formulation"] == "nlp_true"


def test_balance_unknown_formulation():
    payload = {
        "lambda_total": 150.0,
        "servers": SERVERS_3,
        "formulation": "garbage",
    }
    r = client.post("/balance", json=payload)
    assert r.status_code == 400


def test_balance_infeasible_load():
    """Total load exceeds capacity → 422 from validator."""
    payload = {
        "lambda_total": 500.0,
        "servers": [{"id": 0, "mu": 10.0}, {"id": 1, "mu": 10.0}],
        "rho_max": 0.9,
        "formulation": "balanced",
    }
    r = client.post("/balance", json=payload)
    assert r.status_code == 422


def test_balance_all_servers_stable():
    payload = {
        "lambda_total": 100.0,
        "servers": SERVERS_3,
        "formulation": "balanced",
    }
    r = client.post("/balance", json=payload)
    assert r.status_code == 200
    for srv in r.json()["servers"]:
        assert srv["stable"] is True


# ---- /metrics ----

def test_metrics_basic():
    payload = {
        "lambda_total": 100.0,
        "servers": [{"id": 0, "mu": 200.0}, {"id": 1, "mu": 200.0}],
        "shares": [0.5, 0.5],
    }
    r = client.post("/metrics", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    # each server: λ=50, μ=200 → ρ=0.25 → W=1/150 s ≈ 6.667 ms
    for srv in data:
        assert abs(srv["w_ms"] - 1000.0 / 150.0) < 0.01


def test_metrics_shares_not_summing():
    payload = {
        "lambda_total": 100.0,
        "servers": [{"id": 0, "mu": 200.0}, {"id": 1, "mu": 200.0}],
        "shares": [0.3, 0.3],
    }
    r = client.post("/metrics", json=payload)
    assert r.status_code == 422


# ---- /compare ----

def test_compare_returns_all_formulations():
    r = client.get("/compare?lambda_total=150&mu1=100&mu2=80&mu3=60")
    assert r.status_code == 200
    data = r.json()
    assert "balanced" in data
    assert "response_proxy" in data
    assert "nlp_true" in data
    assert "equal" in data


def test_compare_nlp_better_than_equal():
    r = client.get("/compare?lambda_total=150&mu1=100&mu2=80&mu3=60")
    data = r.json()
    assert data["nlp_true"]["mean_response_ms"] <= data["equal"]["mean_response_ms"] + 0.01
