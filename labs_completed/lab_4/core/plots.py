# -*- coding: utf-8 -*-
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

plt.rcParams['font.family'] = 'DejaVu Sans'

SAVE_DIR = Path(__file__).parent.parent / 'plots'


def _w_avg_ms(lambda_total: float, mu_list: List[float], shares: List[float]) -> float:
    mu = np.array(mu_list)
    x = np.array(shares)
    lam_i = lambda_total * x
    if np.any(lam_i >= mu):
        return float('inf')
    w = np.sum(x / (mu - lam_i))
    return w * 1000.0


def save_all_plots(
    lambda_total: float,
    mu_list: List[float],
    results: Dict[str, Any],
    eq_shares: List[float],
    eq_mets: List[Any],
    save_dir: Path,
    rho_max: float = 0.90,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    method_names = list(results.keys())
    all_names = method_names + ["Равные доли"]
    all_shares = [results[n].shares for n in method_names] + [eq_shares]

    w_values = [_w_avg_ms(lambda_total, mu_list, s) for s in all_shares]

    # Baseline W is the last entry (Равные доли)
    w_baseline = w_values[-1]
    # Best W is among the method entries (not baseline)
    best_idx = int(np.argmin(w_values[:len(method_names)]))
    best_w = w_values[best_idx]
    improvement_pct = (w_baseline - best_w) / w_baseline * 100.0 if (w_baseline > 0 and not math.isinf(w_baseline)) else 0.0

    # Colour map: per-method colours
    _name_colors = {
        "NLP": "mediumseagreen",
        "Response LP": "cornflowerblue",
        "Balanced LP": "steelblue",
        "Равные доли": "lightgray",
    }

    def _bar_color(name: str) -> str:
        for key, col in _name_colors.items():
            if key in name:
                return col
        return "steelblue"

    colors = [_bar_color(n) for n in all_names]

    # --- Plot 1: W_avg comparison ---
    finite_w = [w for w in w_values if not math.isinf(w)]
    x_scale = max(finite_w) if finite_w else 1.0
    # Replace inf with capped value so matplotlib doesn't receive inf in bar data
    bar_values = [min(w, x_scale * 1.1) for w in w_values]

    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = np.arange(len(all_names))
    bars = ax.barh(y_pos, bar_values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_names, fontsize=10)
    ax.set_xlabel("W_avg (мс)", fontsize=11)
    ax.set_title("Сравнение среднего времени отклика W_avg", fontsize=12, pad=10)

    if not math.isinf(w_baseline):
        ax.axvline(x=w_baseline, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"baseline = {w_baseline:.3f} мс")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc='lower right')

    for i, (bar, val) in enumerate(zip(bars, w_values)):
        label = "∞" if math.isinf(val) else f"{val:.3f}"
        color = 'gray' if math.isinf(val) else None
        kw = dict(va='center', ha='left', fontsize=9)
        if color:
            kw['color'] = color
        ax.text(bar.get_width() + x_scale * 0.01, bar.get_y() + bar.get_height() / 2,
                label, **kw)
        if i == best_idx and improvement_pct > 0:
            ax.annotate(
                f"−{improvement_pct:.1f}% vs baseline",
                xy=(val, bar.get_y() + bar.get_height() / 2),
                xytext=(val + x_scale * 0.12, bar.get_y() + bar.get_height() / 2 + 0.4),
                fontsize=9, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
                va='center',
            )
    ax.set_xlim(0, x_scale * 1.3)
    fig.tight_layout()
    fig.savefig(save_dir / "plot_w_comparison.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: grouped bar chart of shares ---
    n_servers = len(mu_list)
    n_methods = len(all_names)
    x = np.arange(n_servers)
    width = 0.8 / n_methods
    equal_share = 1.0 / n_servers

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (name, shares) in enumerate(zip(all_names, all_shares)):
        offset = (idx - n_methods / 2 + 0.5) * width
        color = _bar_color(name)
        ax.bar(x + offset, shares, width=width * 0.9, label=name, color=color, edgecolor='white')

    # Horizontal dashed line at equal share level
    ax.axhline(y=equal_share, color="gray", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"equal baseline (1/N = {equal_share:.3f})")

    server_labels = [f"Server {i}\n(μ={mu_list[i]:.0f})" for i in range(n_servers)]
    ax.set_xticks(x)
    ax.set_xticklabels(server_labels, fontsize=10)
    ax.set_ylabel("Доля трафика x_i", fontsize=11)
    ax.set_title("Распределение трафика по серверам", fontsize=12, pad=10)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(save_dir / "plot_shares.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: W_avg (NLP) vs lambda ---
    mu = np.array(mu_list)
    lambda_cap = float(np.sum(mu) * rho_max)
    lambdas = np.linspace(lambda_cap * 0.10, lambda_cap * 0.90, 120)

    w_curve = []
    w_baseline_curve = []
    from scipy.optimize import minimize as scipy_minimize

    for lam in lambdas:
        n = len(mu_list)
        eq_x = np.array([1.0 / n] * n)
        lam_i_eq = lam * eq_x
        if np.any(lam_i_eq >= mu):
            w_baseline_curve.append(float('nan'))
        else:
            w_baseline_curve.append(float(np.sum(eq_x / (mu - lam_i_eq)) * 1000.0))

        ub = [rho_max * mu_list[i] / lam for i in range(n)]
        bounds = [(0.0, min(1.0, u)) for u in ub]
        constraints = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}]
        x0 = [1.0 / n] * n

        def obj(x, _lam=lam):
            return float(np.sum(x / (mu - _lam * x)))

        res = scipy_minimize(obj, x0, method="SLSQP", bounds=bounds,
                             constraints=constraints,
                             options={"ftol": 1e-10, "maxiter": 500})
        if res.success:
            w_curve.append(res.fun * 1000.0)
        else:
            w_curve.append(float('nan'))

    w_curve_arr = np.array(w_curve, dtype=float)
    w_baseline_arr = np.array(w_baseline_curve, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambdas, w_curve_arr, color="mediumseagreen", linewidth=2, label="NLP оптимум")
    ax.plot(lambdas, w_baseline_arr, color="gray", linewidth=1.2,
            linestyle="-.", alpha=0.7, label="Равные доли (baseline)")

    # Fill improvement zone (NLP below baseline)
    ax.fill_between(lambdas, w_curve_arr, w_baseline_arr,
                    where=(w_curve_arr < w_baseline_arr),
                    color="mediumseagreen", alpha=0.08, label="зона улучшения")

    # Annotate improvement zone label in the middle of the range
    mid_idx = len(lambdas) // 2
    zone_mid_y = (w_curve_arr[mid_idx] + w_baseline_arr[mid_idx]) / 2
    ax.text(lambdas[mid_idx], zone_mid_y, "зона улучшения",
            fontsize=9, color="green", ha='center', va='center', alpha=0.7)

    # Vertical line at current lambda_total with annotation
    ax.axvline(x=lambda_total, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Текущий λ = {lambda_total:.1f}")

    # Find y value of NLP curve closest to lambda_total for annotation placement
    closest_idx = int(np.argmin(np.abs(lambdas - lambda_total)))
    y_at_current = w_curve_arr[closest_idx] if not np.isnan(w_curve_arr[closest_idx]) else ax.get_ylim()[1] * 0.5
    ax.annotate(
        "(текущий λ)",
        xy=(lambda_total, y_at_current),
        xytext=(lambda_total + lambda_cap * 0.03, y_at_current * 1.05),
        fontsize=9, color="tomato",
        arrowprops=dict(arrowstyle="->", color="tomato", lw=1.0),
    )

    ax.set_xlabel("Входящий поток λ (req/s)", fontsize=11)
    ax.set_ylabel("W_avg (мс)", fontsize=11)
    ax.set_title("W_avg (NLP) в зависимости от λ", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_dir / "plot_w_vs_lambda.png", dpi=150)
    plt.close(fig)


def generate_charts_b64(
    lambda_total: float,
    mu_list: List[float],
    results: Dict[str, Any],
    eq_shares: List[float],
) -> List[Dict[str, str]]:
    """Generate all charts and return as list of {name, data} base64 PNG dicts."""
    import io, base64

    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    method_names = list(results.keys())
    all_names = method_names + ["Равные доли"]
    all_shares = [results[n].shares for n in method_names] + [eq_shares]
    w_values = [_w_avg_ms(lambda_total, mu_list, s) for s in all_shares]
    w_baseline = w_values[-1]
    best_idx = int(np.argmin(w_values[:len(method_names)]))
    best_w = w_values[best_idx]
    improvement_pct = (w_baseline - best_w) / w_baseline * 100.0 if (w_baseline > 0 and not math.isinf(w_baseline)) else 0.0

    def _bar_color(name: str) -> str:
        for key, col in {"NLP": "mediumseagreen", "Response LP": "cornflowerblue",
                         "Balanced LP": "steelblue", "Равные доли": "lightgray"}.items():
            if key in name:
                return col
        return "steelblue"

    colors = [_bar_color(n) for n in all_names]
    out = []

    # Chart 1: W_avg comparison
    finite_w = [w for w in w_values if not math.isinf(w)]
    x_scale = max(finite_w) if finite_w else 1.0
    bar_values = [min(w, x_scale * 1.1) for w in w_values]

    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = np.arange(len(all_names))
    bars = ax.barh(y_pos, bar_values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos); ax.set_yticklabels(all_names, fontsize=10)
    ax.set_xlabel("W_avg (мс)", fontsize=11)
    ax.set_title("Сравнение среднего времени отклика W_avg", fontsize=12)
    if not math.isinf(w_baseline) and not math.isnan(w_baseline):
        ax.axvline(x=w_baseline, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"baseline = {w_baseline:.3f} мс")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc='lower right')
    for i, (bar, val) in enumerate(zip(bars, w_values)):
        label = "∞" if math.isinf(val) else f"{val:.3f}"
        kw = dict(va='center', ha='left', fontsize=9)
        if math.isinf(val):
            kw['color'] = 'gray'
        ax.text(bar.get_width() + x_scale * 0.01, bar.get_y() + bar.get_height() / 2,
                label, **kw)
        if i == best_idx and improvement_pct > 0:
            ax.annotate(f"−{improvement_pct:.1f}% vs baseline",
                        xy=(val, bar.get_y() + bar.get_height() / 2),
                        xytext=(val + x_scale * 0.12, bar.get_y() + bar.get_height() / 2 + 0.4),
                        fontsize=9, color="darkgreen",
                        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2), va='center')
    ax.set_xlim(0, x_scale * 1.35)
    fig.tight_layout()
    out.append({"name": "Сравнение W_avg", "data": _fig_to_b64(fig)})

    # Chart 2: Traffic shares
    n_servers = len(mu_list)
    n_methods = len(all_names)
    x = np.arange(n_servers)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (name, shares) in enumerate(zip(all_names, all_shares)):
        offset = (idx - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, shares, width=width * 0.9, label=name, color=_bar_color(name), edgecolor='white')
    ax.axhline(y=1.0 / n_servers, color="gray", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"equal baseline (1/N = {1.0/n_servers:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Server {i}\n(μ={mu_list[i]:.0f})" for i in range(n_servers)], fontsize=10)
    ax.set_ylabel("Доля трафика x_i", fontsize=11)
    ax.set_title("Распределение трафика по серверам", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out.append({"name": "Доли трафика", "data": _fig_to_b64(fig)})

    # Chart 3: W vs lambda
    from scipy.optimize import minimize as scipy_minimize
    mu = np.array(mu_list)
    rho_max = 0.90
    lambda_cap = float(np.sum(mu) * rho_max)
    lambdas = np.linspace(lambda_cap * 0.10, lambda_cap * 0.90, 80)
    w_curve, w_baseline_curve = [], []
    n = len(mu_list)
    for lam in lambdas:
        eq_x = np.array([1.0 / n] * n)
        lam_i_eq = lam * eq_x
        w_baseline_curve.append(float(np.sum(eq_x / (mu - lam_i_eq)) * 1000.0) if not np.any(lam_i_eq >= mu) else float('nan'))
        ub = [rho_max * mu_list[i] / lam for i in range(n)]
        res = scipy_minimize(lambda x, _l=lam: float(np.sum(x / (mu - _l * x))), [1.0/n]*n,
                             method="SLSQP", bounds=[(0.0, min(1.0, u)) for u in ub],
                             constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
                             options={"ftol": 1e-10, "maxiter": 500})
        w_curve.append(res.fun * 1000.0 if res.success else float('nan'))
    w_curve_arr = np.array(w_curve, dtype=float)
    w_base_arr = np.array(w_baseline_curve, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambdas, w_curve_arr, color="mediumseagreen", linewidth=2, label="NLP оптимум")
    ax.plot(lambdas, w_base_arr, color="gray", linewidth=1.2, linestyle="-.", alpha=0.7, label="Равные доли (baseline)")
    ax.fill_between(lambdas, w_curve_arr, w_base_arr, where=(w_curve_arr < w_base_arr),
                    color="mediumseagreen", alpha=0.08, label="зона улучшения")
    ax.axvline(x=lambda_total, color="tomato", linestyle="--", linewidth=1.5, label=f"Текущий λ = {lambda_total:.1f}")
    closest_idx = int(np.argmin(np.abs(lambdas - lambda_total)))
    y_ann = w_curve_arr[closest_idx] if not np.isnan(w_curve_arr[closest_idx]) else 10.0
    ax.annotate("(текущий λ)", xy=(lambda_total, y_ann),
                xytext=(lambda_total + lambda_cap * 0.03, y_ann * 1.2),
                fontsize=9, color="tomato", arrowprops=dict(arrowstyle="->", color="tomato", lw=1.0))
    ax.set_xlabel("Входящий поток λ (req/s)", fontsize=11)
    ax.set_ylabel("W_avg (мс)", fontsize=11)
    ax.set_title("W_avg (NLP) в зависимости от λ", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out.append({"name": "W vs λ", "data": _fig_to_b64(fig)})

    return out
