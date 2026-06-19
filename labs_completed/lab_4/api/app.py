# -*- coding: utf-8 -*-
"""FastAPI application for ServerBalancer."""
from __future__ import annotations
from typing import List, Optional
import sys
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

# allow import when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.queue_math import system_metrics, mean_response_time, equal_shares
from core.lp_optimizer import lp_balance, lp_response_proxy, nlp_true_response
from core.simulation import simulate_system
from core.plots import generate_charts_b64

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


# ─────────────────────────── Dashboard ───────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ServerBalancer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f4f8;color:#1a202c}
header{background:#1a1a2e;color:#fff;padding:16px 40px;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:1.3rem;font-weight:700;letter-spacing:.02em}
header span{opacity:.5;font-size:.85rem;display:block;margin-top:2px}
header a{color:#90cdf4;font-size:.82rem;text-decoration:none;border:1px solid #4a5568;
  padding:5px 12px;border-radius:5px;transition:all .15s}
header a:hover{background:#2d3748;border-color:#90cdf4}
.container{max-width:1100px;margin:28px auto;padding:0 20px}
.card{background:#fff;border-radius:10px;padding:22px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.section-title{font-size:.75rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
.preset-btn{background:#edf2f7;color:#2d3748;border:2px solid transparent;padding:6px 14px;
  border-radius:20px;cursor:pointer;font-size:.83rem;font-weight:600;transition:all .15s}
.preset-btn:hover{background:#e2e8f0;border-color:#cbd5e0}
.preset-btn.active{background:#ebf8ff;border-color:#4299e1;color:#2b6cb0}
.form-row{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap}
.fg{display:flex;flex-direction:column;gap:5px}
label{font-size:.72rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em}
input[type=number]{width:90px;padding:8px 10px;border:1px solid #e2e8f0;border-radius:6px;
  font-size:.95rem;color:#2d3748;transition:border .2s}
input[type=number]:focus{outline:none;border-color:#4299e1}
.btn{background:#3182ce;color:#fff;border:none;padding:9px 22px;border-radius:6px;cursor:pointer;
  font-size:.95rem;font-weight:600;transition:background .15s}
.btn:hover{background:#2b6cb0}
.hint{text-align:center;padding:40px 20px;color:#a0aec0}
.hint-icon{font-size:2rem;margin-bottom:10px}
.hint p{font-size:.9rem}
#err-banner{display:none;background:#fff5f5;border:1px solid #fc8181;color:#c53030;
  border-radius:8px;padding:12px 18px;margin-bottom:16px;font-size:.9rem;font-weight:600}
.summary{font-size:1rem;font-weight:700;color:#276749;background:#f0fff4;border:1px solid #9ae6b4;
  border-radius:6px;padding:10px 16px;margin-bottom:18px}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{padding:9px 12px;text-align:left;font-size:.72rem;font-weight:700;color:#718096;
  text-transform:uppercase;letter-spacing:.05em;border-bottom:2px solid #e2e8f0;background:#f7fafc}
td{padding:11px 12px;border-bottom:1px solid #edf2f7;vertical-align:middle}
tr.best td{background:#f0fff4;font-weight:600}
tr.baseline td{color:#a0aec0}
tr.worse td{color:#718096}
.rho-g{color:#276749;font-weight:600}
.rho-y{color:#975a16;font-weight:600}
.rho-r{color:#c53030;font-weight:600}
.delta-pos{color:#276749;font-size:.8rem;font-weight:700}
.delta-neg{color:#c53030;font-size:.8rem;font-weight:700}
.delta-neu{color:#a0aec0;font-size:.8rem}
.badge-warn{background:#fffbeb;color:#92400e;border:1px solid #f6d860;border-radius:4px;
  font-size:.72rem;padding:1px 6px;font-weight:700;margin-left:6px}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
.chart-wide{grid-column:1/-1}
.chart-card{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.chart-card p{font-size:.75rem;font-weight:700;color:#718096;text-transform:uppercase;
  letter-spacing:.05em;margin-bottom:10px}
.chart-card img{width:100%;border-radius:4px}
.legend{display:flex;gap:16px;font-size:.78rem;color:#718096;margin-top:8px}
.legend span{display:flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
#loading{text-align:center;padding:30px;color:#718096;display:none}
#results{display:none}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #e2e8f0;
  border-top-color:#3182ce;border-radius:50%;animation:spin .7s linear infinite;
  vertical-align:middle;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <div>
    <h1>ServerBalancer</h1>
    <span>M/M/1 queueing &middot; LP/NLP optimisation</span>
  </div>
  <a href="/docs" target="_blank">API Docs &rarr;</a>
</header>
<div class="container">

  <div class="card">
    <div class="section-title">Пресеты</div>
    <div class="presets" id="presets"></div>
    <div class="section-title">Параметры системы</div>
    <div class="form-row">
      <div class="fg"><label>λ total (req/s)</label><input type="number" id="lam" value="" min="1" step="10"></div>
      <div class="fg"><label>μ₁ (req/s)</label><input type="number" id="mu1" value="" min="1" step="10"></div>
      <div class="fg"><label>μ₂ (req/s)</label><input type="number" id="mu2" value="" min="1" step="10"></div>
      <div class="fg"><label>μ₃ (req/s)</label><input type="number" id="mu3" value="" min="1" step="10"></div>
      <div class="fg"><label>ρ_max</label><input type="number" id="rho" value="" min="0.1" max="0.99" step="0.05"></div>
      <button class="btn" onclick="analyze()">Оптимизировать</button>
    </div>
  </div>

  <div id="err-banner"></div>

  <div id="hint" class="card">
    <div class="hint">
      <div class="hint-icon">&#8593;</div>
      <p>Выберите пресет — анализ запустится автоматически.<br>Или введите параметры вручную и нажмите <b>Оптимизировать</b>.</p>
    </div>
  </div>

  <div id="loading"><span class="spinner"></span>Вычисляю...</div>

  <div id="results">
    <div class="card">
      <div class="section-title">Сравнение методов</div>
      <div id="summary" class="summary"></div>
      <table>
        <thead>
          <tr>
            <th>Метод</th><th id="h0">x₀</th><th id="h1">x₁</th><th id="h2">x₂</th>
            <th>W_avg, мс</th><th title="Разница с равными долями">vs baseline</th>
          </tr>
        </thead>
        <tbody id="tBody"></tbody>
      </table>
    </div>
    <div class="card">
      <div class="section-title">Характеристики серверов (NLP оптимум)</div>
      <table>
        <thead><tr><th>Сервер</th><th>λᵢ req/s</th><th>ρᵢ</th><th>Wᵢ мс</th><th>Lq</th></tr></thead>
        <tbody id="sBody"></tbody>
      </table>
      <div class="legend">
        <span><span class="dot" style="background:#276749"></span>ρ &lt; 0.6 — низкая нагрузка</span>
        <span><span class="dot" style="background:#975a16"></span>ρ 0.6–0.8 — средняя</span>
        <span><span class="dot" style="background:#c53030"></span>ρ &gt; 0.8 — высокая</span>
      </div>
    </div>
    <div class="charts-grid" id="chartsDiv"></div>
  </div>

</div>
<script>
const PRESETS = [
  {label:"Вариант 7 (базовый)", lam:150, mu:[100,80,60],    rho:0.90},
  {label:"Лёгкая нагрузка",    lam:50,  mu:[100,80,60],    rho:0.90},
  {label:"Высокая нагрузка",   lam:200, mu:[200,150,100],  rho:0.85},
  {label:"Равные серверы",     lam:90,  mu:[100,100,100],  rho:0.90},
  {label:"Почти на пределе",   lam:220, mu:[200,150,100],  rho:0.95},
];

function buildPresets(){
  const div=document.getElementById('presets');
  PRESETS.forEach((p,i)=>{
    const btn=document.createElement('button');
    btn.className='preset-btn';
    btn.textContent=p.label;
    btn.onclick=()=>applyPreset(i,true);
    div.appendChild(btn);
  });
}

function applyPreset(i, autoRun=false){
  const p=PRESETS[i];
  document.getElementById('lam').value=p.lam;
  document.getElementById('mu1').value=p.mu[0];
  document.getElementById('mu2').value=p.mu[1];
  document.getElementById('mu3').value=p.mu[2];
  document.getElementById('rho').value=p.rho;
  document.querySelectorAll('.preset-btn').forEach((b,j)=>b.classList.toggle('active',j===i));
  if(autoRun) analyze();
}

function showError(msg){
  const b=document.getElementById('err-banner');
  b.textContent='Ошибка: '+msg;
  b.style.display='block';
  setTimeout(()=>b.style.display='none', 5000);
}

async function analyze(){
  const lam=+document.getElementById('lam').value;
  const mu_list=[+document.getElementById('mu1').value,+document.getElementById('mu2').value,
                 +document.getElementById('mu3').value];
  const rho_max=+document.getElementById('rho').value;
  if(!lam||mu_list.some(m=>!m)||!rho_max){showError('Заполните все поля');return;}
  document.getElementById('err-banner').style.display='none';
  document.getElementById('hint').style.display='none';
  document.getElementById('loading').style.display='block';
  document.getElementById('results').style.display='none';
  try{
    const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lambda_total:lam,mu_list,rho_max})});
    if(!r.ok){const e=await r.json();showError(e.detail||'Ошибка сервера');return;}
    render(await r.json());
  }catch(e){showError(e.message);}
  finally{document.getElementById('loading').style.display='none';}
}

function render(d){
  document.getElementById('summary').textContent=
    `★  NLP оптимум: W_avg = ${d.best_w_ms.toFixed(3)} мс — улучшение ${d.improvement.toFixed(1)}% vs равномерного`;

  const n=d.servers.length;
  ['h0','h1','h2'].forEach((id,i)=>{
    const el=document.getElementById(id);
    if(el) el.textContent=i<n?`x${i} (μ=${d.servers[i].mu})`:`x${i}`;
  });

  // Find baseline W for delta column
  const baselineRow=d.methods.find(m=>m.baseline);
  const wBase=baselineRow?baselineRow.w_ms:null;

  const tb=document.getElementById('tBody'); tb.innerHTML='';
  d.methods.forEach(m=>{
    const tr=document.createElement('tr');
    const worse=!m.baseline&&wBase&&m.w_ms>wBase;
    tr.className=m.best?'best':m.baseline?'baseline':worse?'worse':'';
    const sh=m.shares.map(x=>x.toFixed(4)).join('</td><td>');
    const wLabel=`${m.best?'★ ':''}<b>${m.w_ms.toFixed(3)}</b>${worse?'<span class="badge-warn">хуже baseline</span>':''}`;
    let delta='<span class="delta-neu">—</span>';
    if(!m.baseline&&wBase){
      const pct=((wBase-m.w_ms)/wBase*100);
      delta=pct>=0
        ?`<span class="delta-pos">▼ ${pct.toFixed(1)}%</span>`
        :`<span class="delta-neg">▲ ${Math.abs(pct).toFixed(1)}%</span>`;
    }
    tr.innerHTML=`<td>${m.name}</td><td>${sh}</td><td>${wLabel}</td><td>${delta}</td>`;
    tb.appendChild(tr);
  });

  const sb=document.getElementById('sBody'); sb.innerHTML='';
  d.servers.forEach(s=>{
    const cls=s.rho>0.8?'rho-r':s.rho>0.6?'rho-y':'rho-g';
    sb.innerHTML+=`<tr>
      <td>Server ${s.id} <span style="color:#a0aec0;font-size:.82rem">(μ=${s.mu})</span></td>
      <td>${s.lambda_i.toFixed(2)}</td>
      <td class="${cls}">${s.rho.toFixed(4)}</td>
      <td>${s.w_ms.toFixed(3)}</td>
      <td>${s.l_q.toFixed(4)}</td></tr>`;
  });

  const cd=document.getElementById('chartsDiv'); cd.innerHTML='';
  d.charts.forEach((c,i)=>{
    const wide=i===d.charts.length-1?'chart-wide':'';
    cd.innerHTML+=`<div class="chart-card ${wide}"><p>${c.name}</p>
      <img src="data:image/png;base64,${c.data}" loading="lazy"></div>`;
  });
  document.getElementById('results').style.display='block';
}

buildPresets();
</script>
</body>
</html>"""


class AnalyzeRequest(BaseModel):
    lambda_total: float = Field(gt=0)
    mu_list: List[float] = Field(min_length=2)
    rho_max: float = Field(default=0.90, gt=0, lt=1.0)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Interactive web dashboard."""
    return _DASHBOARD_HTML


@app.post("/analyze", tags=["dashboard"])
async def analyze(req: AnalyzeRequest):
    """
    Run all three optimisation formulations + generate charts.
    Used by the web dashboard at GET /.
    """
    mu_list = req.mu_list
    mu_sum = sum(mu_list)
    if req.lambda_total >= req.rho_max * mu_sum:
        raise HTTPException(
            status_code=400,
            detail=f"Система нефизична: λ={req.lambda_total} ≥ ρ_max·Σμ={req.rho_max*mu_sum:.1f}. "
                   "Уменьшите λ или добавьте серверов.",
        )

    solvers = {
        "Balanced LP (min-max ρ)": lp_balance,
        "Response LP (lin. proxy)": lp_response_proxy,
        "NLP true W (SLSQP)":       nlp_true_response,
    }
    results = {name: fn(req.lambda_total, mu_list, req.rho_max) for name, fn in solvers.items()}

    eq = equal_shares(len(mu_list))
    eq_mets = system_metrics(req.lambda_total, mu_list, eq)
    w_eq_ms = mean_response_time(eq_mets, eq) * 1000.0

    best_name = "NLP true W (SLSQP)"
    best_res = results[best_name]
    best_mets = system_metrics(req.lambda_total, mu_list, best_res.shares)
    best_w_ms = mean_response_time(best_mets, best_res.shares) * 1000.0
    improvement = (w_eq_ms - best_w_ms) / w_eq_ms * 100.0 if w_eq_ms > 0 else 0.0

    # Build methods list for table
    methods = []
    w_list = []
    for name, res in results.items():
        mets = system_metrics(req.lambda_total, mu_list, res.shares)
        w_ms = mean_response_time(mets, res.shares) * 1000.0
        w_list.append(w_ms)
        methods.append({"name": name, "shares": [round(x, 4) for x in res.shares],
                        "w_ms": round(w_ms, 3), "best": False, "baseline": False})
    # Mark best
    best_idx = w_list.index(min(w_list))
    methods[best_idx]["best"] = True
    # Add baseline
    methods.append({"name": "Равные доли (baseline)",
                    "shares": [round(x, 4) for x in eq],
                    "w_ms": round(w_eq_ms, 3), "best": False, "baseline": True})

    # Server metrics for NLP optimum
    servers = [
        {"id": m.server_id, "mu": round(mu_list[m.server_id], 1),
         "lambda_i": round(m.lambda_i, 2), "rho": round(m.rho, 4),
         "w_ms": round(m.w * 1000.0, 3) if m.stable else 9999.0,
         "l_q": round(m.l_q, 4) if m.stable else 9999.0}
        for m in best_mets
    ]

    charts = generate_charts_b64(req.lambda_total, mu_list, results, eq)

    return {
        "best_w_ms": round(best_w_ms, 3),
        "improvement": round(improvement, 2),
        "methods": methods,
        "servers": servers,
        "charts": charts,
    }
