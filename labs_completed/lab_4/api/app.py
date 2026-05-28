# -*- coding: utf-8 -*-
"""FastAPI application for ServerBalancer."""
from __future__ import annotations
from typing import List, Optional
import sys
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

# allow import when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.queue_math import system_metrics, mean_response_time, equal_shares
from core.lp_optimizer import lp_balance, lp_response_proxy, nlp_true_response
from core.simulation import simulate_system

app = FastAPI(
    title="ServerBalancer",
    description="Load balancer prototype: M/M/1 queuing + LP traffic optimisation",
    version="1.0.0",
)


# ---------- Request / Response schemas ----------

class ServerConfig(BaseModel):
    id: int = Field(ge=0)
    mu: float = Field(gt=0, description="Service rate (req/s)")


class BalanceRequest(BaseModel):
    lambda_total: float = Field(gt=0, description="Total arrival rate (req/s)")
    servers: List[ServerConfig] = Field(min_length=2)
    rho_max: float = Field(default=0.90, gt=0, lt=1.0,
                           description="Max allowed utilisation per server")
    formulation: str = Field(default="balanced",
                             description="'balanced' | 'response_proxy' | 'nlp_true'")

    @model_validator(mode="after")
    def check_feasibility(self) -> "BalanceRequest":
        mu_sum = sum(s.mu for s in self.servers)
        if self.lambda_total >= self.rho_max * mu_sum:
            raise ValueError(
                f"System infeasible: λ={self.lambda_total} ≥ ρ_max*Σμ="
                f"{self.rho_max * mu_sum:.2f}. Add servers or reduce load."
            )
        return self


class ServerResult(BaseModel):
    server_id: int
    share: float
    lambda_i: float
    rho: float
    w_ms: float         # mean sojourn time in milliseconds
    l_q: float          # mean queue length
    stable: bool


class BalanceResponse(BaseModel):
    formulation: str
    lp_objective: float
    mean_response_ms: float
    servers: List[ServerResult]


class MetricsRequest(BaseModel):
    lambda_total: float = Field(gt=0)
    servers: List[ServerConfig] = Field(min_length=1)
    shares: List[float]

    @model_validator(mode="after")
    def check_shares(self) -> "MetricsRequest":
        if len(self.shares) != len(self.servers):
            raise ValueError("shares length must match servers length")
        if abs(sum(self.shares) - 1.0) > 1e-3:
            raise ValueError("shares must sum to 1")
        return self


class SimRequest(BaseModel):
    lambda_total: float = Field(gt=0)
    servers: List[ServerConfig] = Field(min_length=1)
    shares: List[float]
    duration: float = Field(default=3000.0, gt=0)
    seed: int = Field(default=42)


# ---------- Routes ----------

@app.get("/health", tags=["system"])
async def health():
    """Liveness health-check."""
    return {"status": "ok", "service": "ServerBalancer"}


@app.post("/balance", response_model=BalanceResponse, tags=["optimisation"])
async def balance(req: BalanceRequest):
    """
    Run LP/NLP optimisation and return optimal traffic distribution.

    Formulations:
    - **balanced**: LP min-max utilisation (fair load balancing)
    - **response_proxy**: LP minimise linear proxy for response time
    - **nlp_true**: SciPy SLSQP minimise true M/M/1 mean response time
    """
    mu_list = [s.mu for s in req.servers]

    solver = {
        "balanced": lp_balance,
        "response_proxy": lp_response_proxy,
        "nlp_true": nlp_true_response,
    }.get(req.formulation)

    if solver is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown formulation '{req.formulation}'. "
                   "Use 'balanced', 'response_proxy', or 'nlp_true'.",
        )

    result = solver(req.lambda_total, mu_list, req.rho_max)

    if result.solver_status not in ("Optimal", "Optimal*"):
        raise HTTPException(
            status_code=422,
            detail=f"Solver returned non-optimal status: {result.solver_status}",
        )

    metrics = system_metrics(req.lambda_total, mu_list, result.shares)
    avg_w_ms = mean_response_time(metrics, result.shares) * 1000.0

    server_results = [
        ServerResult(
            server_id=m.server_id,
            share=round(result.shares[i], 6),
            lambda_i=round(m.lambda_i, 4),
            rho=round(m.rho, 4),
            w_ms=round(m.w * 1000.0, 3) if m.stable else float('inf'),
            l_q=round(m.l_q, 4) if m.stable else float('inf'),
            stable=m.stable,
        )
        for i, m in enumerate(metrics)
    ]

    return BalanceResponse(
        formulation=result.formulation,
        lp_objective=round(result.objective, 6),
        mean_response_ms=round(avg_w_ms, 3),
        servers=server_results,
    )


@app.post("/metrics", response_model=List[ServerResult], tags=["analysis"])
async def metrics(req: MetricsRequest):
    """Compute M/M/1 analytical metrics for a given traffic distribution."""
    mu_list = [s.mu for s in req.servers]
    mets = system_metrics(req.lambda_total, mu_list, req.shares)
    return [
        ServerResult(
            server_id=m.server_id,
            share=round(req.shares[i], 6),
            lambda_i=round(m.lambda_i, 4),
            rho=round(m.rho, 4),
            w_ms=round(m.w * 1000.0, 3) if m.stable else float('inf'),
            l_q=round(m.l_q, 4) if m.stable else float('inf'),
            stable=m.stable,
        )
        for i, m in enumerate(mets)
    ]


@app.post("/simulate", tags=["simulation"])
async def simulate(req: SimRequest):
    """
    Run SimPy discrete-event simulation of M/M/1 queues.
    Returns simulated vs analytical mean sojourn time per server.
    """
    mu_list = [s.mu for s in req.servers]
    sim_results = simulate_system(
        req.lambda_total, mu_list, req.shares, req.duration, req.seed
    )
    return [
        {
            "server_id": r.server_id,
            "lambda_i": round(r.lambda_i, 4),
            "rho": round(r.lambda_i / r.mu_i, 4),
            "w_sim_ms": round(r.w_sim * 1000.0, 3) if r.w_sim < 1e9 else None,
            "w_analytical_ms": round(r.w_analytical * 1000.0, 3) if r.w_analytical < 1e9 else None,
            "n_served": r.n_served,
        }
        for r in sim_results
    ]


@app.get("/compare", tags=["analysis"])
async def compare(
    lambda_total: float = 150.0,
    mu1: float = 100.0,
    mu2: float = 80.0,
    mu3: float = 60.0,
    rho_max: float = 0.90,
):
    """
    Quick comparison of all three formulations for a 3-server scenario.
    GET /compare?lambda_total=150&mu1=100&mu2=80&mu3=60
    """
    mu_list = [mu1, mu2, mu3]
    servers = [ServerConfig(id=i, mu=mu) for i, mu in enumerate(mu_list)]
    results = {}
    for form in ("balanced", "response_proxy", "nlp_true"):
        req = BalanceRequest(
            lambda_total=lambda_total,
            servers=servers,
            rho_max=rho_max,
            formulation=form,
        )
        resp = await balance(req)
        results[form] = {
            "mean_response_ms": resp.mean_response_ms,
            "shares": [s.share for s in resp.servers],
            "utilizations": [s.rho for s in resp.servers],
        }
    # baseline: equal shares
    eq = equal_shares(len(mu_list))
    eq_mets = system_metrics(lambda_total, mu_list, eq)
    results["equal"] = {
        "mean_response_ms": round(mean_response_time(eq_mets, eq) * 1000.0, 3),
        "shares": [round(x, 6) for x in eq],
        "utilizations": [round(m.rho, 4) for m in eq_mets],
    }
    return results
