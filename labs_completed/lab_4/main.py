# -*- coding: utf-8 -*-
"""
ServerBalancer – command-line entry point.

Usage:
  python main.py                      # run analysis with default config
  python main.py --serve              # start FastAPI server
  python main.py --lambda 150 --sim   # include SimPy validation
"""
from __future__ import annotations
import sys
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import yaml
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

from core.queue_math import system_metrics, mean_response_time, equal_shares
from core.lp_optimizer import lp_balance, lp_response_proxy, nlp_true_response
from core.simulation import simulate_system

console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_analysis(cfg: dict, run_sim: bool = False) -> None:
    lambda_total = cfg["lambda_total"]
    mu_list = [s["mu"] for s in cfg["servers"]]
    rho_max = cfg.get("rho_max", 0.90)
    sim_duration = cfg.get("simulation_duration", 3000.0)

    console.print(f"\n[bold cyan]ServerBalancer[/bold cyan] — λ={lambda_total} req/s, "
                  f"серверы μ={mu_list}, ρ_max={rho_max}\n")

    # --- Run all three formulations ---
    results = {
        "Balanced LP (min-max ρ)": lp_balance(lambda_total, mu_list, rho_max),
        "Response LP (lin. proxy)": lp_response_proxy(lambda_total, mu_list, rho_max),
        "NLP true W (SLSQP)":       nlp_true_response(lambda_total, mu_list, rho_max),
    }

    # Baseline: equal shares
    eq_shares = equal_shares(len(mu_list))
    eq_mets = system_metrics(lambda_total, mu_list, eq_shares)
    eq_w = mean_response_time(eq_mets, eq_shares)

    # --- Table 1: shares and W ---
    tbl = Table(title="Распределение трафика и среднее время отклика")
    tbl.add_column("Метод", style="bold")
    for i, mu in enumerate(mu_list):
        tbl.add_column(f"x_{i} (μ={mu:.0f})", justify="right")
    tbl.add_column("W_avg, мс", justify="right", style="green")
    tbl.add_column("Статус", justify="center")

    for name, res in results.items():
        mets = system_metrics(lambda_total, mu_list, res.shares)
        w_ms = mean_response_time(mets, res.shares) * 1000.0
        cells = [f"{x:.4f}" for x in res.shares]
        cells.append(f"{w_ms:.3f}")
        cells.append(res.solver_status)
        tbl.add_row(name, *cells)

    # equal baseline
    eq_w_ms = eq_w * 1000.0
    tbl.add_row(
        "[dim]Равные доли (baseline)[/dim]",
        *[f"{x:.4f}" for x in eq_shares],
        f"[dim]{eq_w_ms:.3f}[/dim]",
        "[dim]—[/dim]",
    )
    console.print(tbl)

    # --- Table 2: M/M/1 metrics for NLP optimum ---
    best_res = results["NLP true W (SLSQP)"]
    best_mets = system_metrics(lambda_total, mu_list, best_res.shares)

    tbl2 = Table(title="Характеристики M/M/1 при оптимальном (NLP) распределении")
    tbl2.add_column("Сервер", justify="center")
    tbl2.add_column("λ_i, req/s", justify="right")
    tbl2.add_column("ρ_i", justify="right")
    tbl2.add_column("W_i, мс", justify="right")
    tbl2.add_column("L_q", justify="right")
    tbl2.add_column("Устойчива", justify="center")

    for m in best_mets:
        tbl2.add_row(
            str(m.server_id),
            f"{m.lambda_i:.2f}",
            f"{m.rho:.4f}",
            f"{m.w * 1000:.3f}" if m.stable else "∞",
            f"{m.l_q:.4f}" if m.stable else "∞",
            "✓" if m.stable else "✗",
        )
    console.print(tbl2)

    # --- SimPy validation (optional) ---
    if run_sim:
        console.print("\n[yellow]Запуск SimPy симуляции (M/M/1)...[/yellow]")
        sim_res = simulate_system(
            lambda_total, mu_list, best_res.shares, sim_duration
        )
        tbl3 = Table(title=f"Валидация SimPy (T={sim_duration:.0f}s)")
        tbl3.add_column("Сервер", justify="center")
        tbl3.add_column("W_аналит., мс", justify="right")
        tbl3.add_column("W_сим., мс", justify="right")
        tbl3.add_column("Обслужено", justify="right")

        for r in sim_res:
            tbl3.add_row(
                str(r.server_id),
                f"{r.w_analytical * 1000:.3f}" if r.w_analytical < 1e9 else "∞",
                f"{r.w_sim * 1000:.3f}" if r.w_sim < 1e9 else "∞",
                str(r.n_served),
            )
        console.print(tbl3)

    # --- Improvement vs baseline ---
    best_w_ms = best_res.objective * 1000.0
    # Recalculate properly
    best_mets_check = system_metrics(lambda_total, mu_list, best_res.shares)
    best_w_ms = mean_response_time(best_mets_check, best_res.shares) * 1000.0
    improvement = (eq_w_ms - best_w_ms) / eq_w_ms * 100.0
    console.print(
        f"\n[bold green]Оптимальное W = {best_w_ms:.3f} мс[/bold green] "
        f"(улучшение vs равномерного: {improvement:.1f}%)\n"
    )


def main():
    parser = argparse.ArgumentParser(description="ServerBalancer CLI")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--sim", action="store_true", help="Run SimPy validation")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        console.print(f"[red]Конфиг не найден: {config_path}[/red]")
        raise SystemExit(1)

    cfg = load_config(str(config_path))

    if args.serve:
        import uvicorn
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        run_analysis(cfg, run_sim=args.sim)


if __name__ == "__main__":
    main()
