# -*- coding: utf-8 -*-
"""
Лабораторная работа №2 «Анализ систем массового обслуживания»
Студент: Лобанов Р. Е.

Задача 1. Многоканальная СМО с отказами (M/M/k/0)
  α=1.0 ч, n=36 зявок/сутки, k=4 каналов

Задача 2. Многоканальная СМО с неограниченной очередью (M/M/k/∞)
  λ=48 заявок/сутки, t=10 мин, α=6 (коэф. затрат), n=4 (для P(очередь≤n))

Задача 3. Многоканальная СМО с ограниченной очередью (M/M/k/k+n)
  λ=5 зявок/ч, t=2 мин, k=15 каналов, n=3 места в очереди, T=12 ч, C=85 у.е.

Задача 4. СМО с «нетерпеливыми» заявками (ω-предел)
  λ=0.6 зявок/мин, t=2.5 мин, k=3 канала, ω=12 мин, C=300 у.е., ε=0.01

Задача 5. Замкнутая одноканальная СМО (система Энгсета, M/M/1//N)
  n=18 источников, k=4 заявки/мес на источник, t=2.0 дня, P_min=75%

Задача 6. Замкнутая многоканальная СМО (M/M/k//N)
  k=3 канала, n=12 источников, λ=1.4 зявок/ч от источника, t=0.2 ч
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

_HERE = Path(__file__).parent

plt.rcParams['font.family'] = 'DejaVu Sans'

# ─────────────────── общие вспомогательные функции ───────────────────

def erlang_b(rho0: float, k: int) -> float:
    """Вероятность отказа (формула Эрланга B) для M/M/k/0."""
    numerator = rho0**k / math.factorial(k)
    denominator = sum(rho0**j / math.factorial(j) for j in range(k + 1))
    return numerator / denominator


def erlang_c_p0(rho0: float, k: int) -> Tuple[float, float]:
    """P0 и P_ожид (формула Эрланга C) для M/M/k/∞. ρ=rho0/k < 1."""
    rho = rho0 / k
    sum_terms = sum(rho0**j / math.factorial(j) for j in range(k))
    last = rho0**k / (math.factorial(k) * (1 - rho))
    p0 = 1.0 / (sum_terms + last)
    p_wait = last * p0
    return p0, p_wait


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 1. Многоканальная СМО с отказами (M/M/k/0)
# ═══════════════════════════════════════════════════════════════

def task1():
    section("Задача 1. M/M/k/0 — многоканальная СМО с отказами")

    alpha = 1.0      # среднее время обслуживания, ч
    n_day = 36       # заявок в сутки
    k_given = 4      # заданное число каналов

    lam = n_day / 24         # интенсивность потока, зявок/ч
    mu = 1.0 / alpha         # интенсивность обслуживания, зявок/ч на канал
    rho0 = lam / mu          # предложенная нагрузка (эрланги)

    print(f"\nПараметры: α={alpha} ч, n={n_day} зявок/сутки, k={k_given}")
    print(f"Расчётные: λ={lam:.4f} зявок/ч, μ={mu:.4f} зявок/ч, ρ₀={rho0:.4f} Эрл")

    # 1a. Минимальное k при Q ≥ 95 %
    k_min = 1
    while True:
        p_otk = erlang_b(rho0, k_min)
        if 1 - p_otk >= 0.95:
            break
        k_min += 1

    print(f"\n1a. Минимальное число каналов для Q≥95%: k_min = {k_min}")
    print(f"    P_отк(k={k_min}) = {erlang_b(rho0, k_min):.5f}, Q = {1-erlang_b(rho0,k_min):.5f}")

    # 1b. Решение для k = k_given
    print(f"\n1b. Расчёт для k = {k_given}:")
    p_states = [rho0**j / math.factorial(j) for j in range(k_given + 1)]
    total = sum(p_states)
    p_states = [p / total for p in p_states]

    print("    Предельные вероятности состояний:")
    for j, p in enumerate(p_states):
        print(f"      P({j}) = {p:.5f}")

    p_otk = p_states[k_given]
    Q_rel = 1 - p_otk
    A_abs = lam * Q_rel
    avg_busy = sum(j * p_states[j] for j in range(1, k_given + 1))
    eta = avg_busy / k_given  # коэффициент загрузки

    print(f"\n    Вероятность отказа:              P_отк = {p_otk:.5f}")
    print(f"    Относительная пропускная способность: Q = {Q_rel:.5f}")
    print(f"    Абсолютная пропускная способность:  A = {A_abs:.5f} зявок/ч")
    print(f"    Среднее число занятых каналов:     L_обсл = {avg_busy:.5f}")
    print(f"    Коэффициент загрузки каналов:       η = {eta:.5f}")

    # График Q(k)
    ks = list(range(1, 11))
    qs = [1 - erlang_b(rho0, k) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, qs, 'o-', color='steelblue')
    ax.axhline(0.95, color='red', linestyle='--', label='Q=0.95')
    ax.axvline(k_min, color='green', linestyle=':', label=f'k_min={k_min}')
    ax.set_xlabel('Число каналов k')
    ax.set_ylabel('Относительная пропускная способность Q')
    ax.set_title('Задача 1. Q(k) для M/M/k/0')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(_HERE / 'task1_plot.png', dpi=120)
    plt.close()
    print("\n    [График сохранён: task1_plot.png]")

    print("\nВыводы: Система с отказами при λ=1.5 зявок/ч и μ=1 зявок/ч."
          f"\n  Для обеспечения Q≥95% нужно k_min={k_min} каналов."
          f"\n  При k={k_given}: загрузка η={eta:.3f}, абс. пропускная A={A_abs:.3f} зявок/ч.")


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 2. Многоканальная СМО с неограниченной очередью (M/M/k/∞)
# ═══════════════════════════════════════════════════════════════

def task2():
    section("Задача 2. M/M/k/∞ — СМО с неограниченной очередью")

    lam_day = 48       # заявок/сутки
    t_min = 10         # среднее время обслуживания, мин
    alpha_c = 6        # коэффициент затрат
    n_queue = 4        # для расчёта P(очередь ≤ n)

    lam = lam_day / 24       # зявок/ч
    mu = 60.0 / t_min        # зявок/ч на канал
    rho0 = lam / mu          # λ/μ

    print(f"\nПараметры: λ={lam_day} зявок/сутки, t={t_min} мин, α_c={alpha_c}, n={n_queue}")
    print(f"Расчётные: λ={lam:.4f} зявок/ч, μ={mu:.4f} зявок/ч, ρ₀={rho0:.4f}")

    # 2a. k_min — минимальное для устойчивости
    k_min = max(1, math.ceil(rho0) if rho0 > 1 else 1)
    while lam / (k_min * mu) >= 1.0:
        k_min += 1
    print(f"\n2a. k_min (ρ<1): k_min = {k_min}, ρ = {lam/(k_min*mu):.4f}")

    def mm_k_metrics(k):
        rho = lam / (k * mu)
        if rho >= 1:
            return None
        p0, p_wait = erlang_c_p0(rho0, k)
        L_q = p_wait * rho / (1 - rho)
        W_q = L_q / lam          # ч
        W_q_min = W_q * 60       # мин
        W_min = W_q_min + t_min  # мин
        L = L_q + rho0
        return {'rho': rho, 'p0': p0, 'p_wait': p_wait,
                'L_q': L_q, 'W_q_min': W_q_min, 'W_min': W_min, 'L': L}

    m = mm_k_metrics(k_min)
    print(f"    P0={m['p0']:.5f}, P_ожид={m['p_wait']:.5f}")
    print(f"    L_оч={m['L_q']:.5f}, L={m['L']:.5f}")
    print(f"    W_ожид={m['W_q_min']:.3f} мин, W={m['W_min']:.3f} мин")

    # 2b. k_opt — минимум затрат C(k) = k/α + W (мин)
    print(f"\n2b. Поиск k_opt (мин C(k)=k/α+W):")
    best_k, best_c = k_min, float('inf')
    costs = {}
    for k in range(k_min, k_min + 15):
        met = mm_k_metrics(k)
        if met is None:
            continue
        c = k / alpha_c + met['W_min']
        costs[k] = c
        print(f"    k={k:2d}: C={c:.4f}, W={met['W_min']:.3f} мин")
        if c < best_c:
            best_c, best_k = c, k

    print(f"\n    k_opt = {best_k}, C_min = {best_c:.4f}")

    # 2c. Сравнение k_min и k_opt
    print(f"\n2c. Сравнение k_min={k_min} и k_opt={best_k}:")
    for label, k in [('k_min', k_min), ('k_opt', best_k)]:
        met = mm_k_metrics(k)
        print(f"    {label}: ρ={met['rho']:.4f}, L_оч={met['L_q']:.5f},"
              f" W_ожид={met['W_q_min']:.3f} мин, W={met['W_min']:.3f} мин")

    # 2d. P(очередь ≤ n) для k_opt
    k_use = best_k
    met = mm_k_metrics(k_use)
    p0, p_wait = erlang_c_p0(rho0, k_use)
    rho = lam / (k_use * mu)

    # P(j) = ρ₀ʲ/j! * P0 для j<=k; ρ₀ᵏ/(k! * k^(j-k)) * P0 для j>k
    def p_j(j):
        if j <= k_use:
            return rho0**j / math.factorial(j) * p0
        return rho0**k_use / math.factorial(k_use) * rho**(j - k_use) * p0

    p_le_n = sum(p_j(j) for j in range(k_use + n_queue + 1))
    print(f"\n2d. P(очередь ≤ {n_queue}) при k={k_use}: {p_le_n:.5f}")

    # График стоимости
    fig, ax = plt.subplots(figsize=(7, 4))
    ks_plot = list(costs.keys())
    ax.plot(ks_plot, list(costs.values()), 'o-', color='steelblue')
    ax.axvline(best_k, color='green', linestyle=':', label=f'k_opt={best_k}')
    ax.set_xlabel('Число каналов k')
    ax.set_ylabel('Затраты C(k)')
    ax.set_title('Задача 2. Функция затрат C(k)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(_HERE / 'task2_plot.png', dpi=120)
    plt.close()
    print("    [График сохранён: task2_plot.png]")

    print(f"\nВыводы: k_min={k_min} обеспечивает устойчивость (ρ={lam/(k_min*mu):.3f}<1)."
          f"\n  k_opt={best_k} минимизирует затраты C={best_c:.3f}."
          f"\n  P(в очереди ≤{n_queue} зявок)={p_le_n:.4f} при k={k_use}.")


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 3. Многоканальная СМО с ограниченной очередью (M/M/k/k+n)
# ═══════════════════════════════════════════════════════════════

def task3():
    section("Задача 3. M/M/k/(k+n) — СМО с ограниченной очередью")

    lam = 5.0      # зявок/ч
    t_min = 2.0    # мин
    k = 15         # число каналов
    n = 3          # мест в очереди
    T_work = 12    # ч работы
    C_cost = 85.0  # у.е. за одну обслуженную заявку

    mu = 60.0 / t_min    # зявок/ч на канал
    rho0 = lam / mu      # λ/μ
    m = k + n            # полная ёмкость системы

    print(f"\nПараметры: λ={lam} зявок/ч, t={t_min} мин, k={k} кан., n={n} мест, T={T_work} ч, C={C_cost}")
    print(f"Расчётные: μ={mu:.4f} зявок/ч, ρ₀={rho0:.4f}, m=k+n={m}")

    # Предельные вероятности
    def unnorm_p(j):
        if j <= k:
            return rho0**j / math.factorial(j)
        return rho0**j / (math.factorial(k) * k**(j - k))

    Z = sum(unnorm_p(j) for j in range(m + 1))
    probs = [unnorm_p(j) / Z for j in range(m + 1)]

    print("\nПредельные вероятности состояний P(j):")
    for j, p in enumerate(probs):
        print(f"  P({j:2d}) = {p:.5f}")

    p_otk = probs[m]
    Q_rel = 1 - p_otk
    lam_abs = lam * Q_rel
    L_q = sum((j - k) * probs[j] for j in range(k + 1, m + 1))
    L_obsl = sum(min(j, k) * probs[j] for j in range(1, m + 1))
    L = L_q + L_obsl
    W_q = (L_q / lam_abs) * 60 if lam_abs > 0 else 0   # мин
    W_obsl = (L_obsl / lam_abs) * 60 if lam_abs > 0 else 0
    W = (L / lam_abs) * 60 if lam_abs > 0 else 0

    S_loss = C_cost * lam * p_otk * T_work

    print(f"\nВероятность отказа:               P_отк = {p_otk:.5f}")
    print(f"Относительная пропускная способность: Q = {Q_rel:.5f}")
    print(f"Абсолютная пропускная способность:   A = {lam_abs:.5f} зявок/ч")
    print(f"Среднее число заявок в очереди:     L_q = {L_q:.5f}")
    print(f"Среднее число обслуживаемых:      L_обсл = {L_obsl:.5f}")
    print(f"Среднее число заявок в системе:      L = {L:.5f}")
    print(f"Среднее время ожидания:            W_q = {W_q:.3f} мин")
    print(f"Среднее время обслуживания:       W_обсл = {W_obsl:.3f} мин")
    print(f"Среднее время пребывания:           W = {W:.3f} мин")
    print(f"\nПотеря выручки за T={T_work} ч: S = {S_loss:.2f} у.е.")

    print(f"\nВыводы: При k={k} каналах и λ={lam} зявок/ч нагрузка ρ={lam/(k*mu):.4f}."
          f"\n  Отказы редки (P_отк={p_otk:.4f}), потери={S_loss:.1f} у.е. за смену.")


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 4. СМО с нетерпеливыми заявками (ω-предел)
# ═══════════════════════════════════════════════════════════════

def task4():
    section("Задача 4. M/M/k + нетерпеливые заявки (ω-предел)")

    lam = 0.6      # зявок/мин
    t = 2.5        # мин
    k = 3          # каналов
    omega = 12.0   # макс. время ожидания, мин
    C_income = 300.0
    eps = 0.01     # точность

    mu = 1.0 / t   # 0.4 зявок/мин
    rho0 = lam / mu

    print(f"\nПараметры: λ={lam} зявок/мин, t={t} мин, k={k}, ω={omega} мин, C={C_income}, ε={eps}")
    print(f"Расчётные: μ={mu:.4f} зявок/мин, ρ₀={rho0:.4f}")

    # Скорость выбытия из состояния j
    def death_rate(j):
        return j * mu if j <= k else k * mu + (j - k) / omega

    # Ненормированные вероятности через произведение λ/μᵢ
    unnorm = [1.0]
    j = 1
    while True:
        unnorm.append(unnorm[-1] * lam / death_rate(j))
        if unnorm[-1] < eps * 1e-3:
            break
        j += 1

    total = sum(unnorm)
    probs = [u / total for u in unnorm]
    J = len(probs)

    print(f"\nТрункирование при j={J-1} (P({J-1}) < {eps*1e-3:.2e})")
    print("Предельные вероятности P(j):")
    for j_i, p in enumerate(probs[:min(J, 15)]):
        marker = " ← в очереди" if j_i > k else ""
        print(f"  P({j_i:2d}) = {p:.5f}{marker}")
    if J > 15:
        print(f"  ... (всего {J} состояний)")

    L_q = sum((j_i - k) * probs[j_i] for j_i in range(k + 1, J))
    L_obsl = sum(min(j_i, k) * probs[j_i] for j_i in range(1, J))
    L = L_q + L_obsl

    nu_leave = L_q / omega           # интенсивность ухода из очереди
    lam_served = lam - nu_leave      # интенсивность обслуживания

    P_served = lam_served / lam
    W_q = (L_q / lam_served) if lam_served > 0 else 0   # мин
    W_obsl = t
    W = W_q + W_obsl

    D_loss = C_income * nu_leave

    print(f"\nСредняя длина очереди:          L_q = {L_q:.5f}")
    print(f"Среднее число обслуживаемых:  L_обсл = {L_obsl:.5f}")
    print(f"Среднее число в системе:          L = {L:.5f}")
    print(f"Интенсивность ухода из очереди: ν_уход = {nu_leave:.5f} зявок/мин")
    print(f"Вероятность обслуживания:    P_обсл = {P_served:.5f}")
    print(f"Среднее время ожидания:         W_q = {W_q:.3f} мин")
    print(f"Среднее время обслуживания:  W_обсл = {W_obsl:.3f} мин")
    print(f"Среднее время пребывания:        W = {W:.3f} мин")
    print(f"\nСредние потери дохода: Д = {D_loss:.2f} у.е./мин")

    print(f"\nВыводы: При ω={omega} мин ρ={lam/(k*mu):.3f}."
          f"\n  Уходит {nu_leave:.4f} зявок/мин ({(1-P_served)*100:.2f}%)."
          f"\n  Потери дохода {D_loss:.2f} у.е./мин.")


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 5. Замкнутая одноканальная СМО (система Энгсета, M/M/1//N)
# ═══════════════════════════════════════════════════════════════

def task5():
    section("Задача 5. M/M/1//N — замкнутая одноканальная СМО (Энгсет)")

    N = 18        # источников заявок
    k_month = 4   # заявок в месяц от одного источника
    t_days = 2.0  # дней на обслуживание
    P_min = 75.0  # минимальный % активных источников

    # Единица времени: месяц (30 дней)
    lam_src = k_month              # заявок/мес от одного активного источника
    mu = 30.0 / t_days             # обслуживаний/мес (=15)
    a = lam_src / mu               # нагрузка от одного источника

    print(f"\nПараметры: N={N} источников, k={k_month} зявок/мес, t={t_days} дней, P_min={P_min}%")
    print(f"Расчётные: λ_ист={lam_src}/мес, μ={mu}/мес, a=λ/μ={a:.4f}")

    # P(j) = C(N,j)*aʲ*P(0)
    unnorm = [math.comb(N, j) * a**j for j in range(N + 1)]
    Z = sum(unnorm)
    probs = [u / Z for u in unnorm]

    print("\nПредельные вероятности P(j) (j = число неисправных источников):")
    for j, p in enumerate(probs):
        print(f"  P({j:2d}) = {p:.5f}")

    # P(активных ≥ P_min%)
    threshold = math.ceil(P_min / 100 * N)   # минимум активных
    max_broken = N - threshold                # максимум неисправных
    p_ok = sum(probs[j] for j in range(max_broken + 1))

    print(f"\nP(активных ≥ {threshold} из {N}): P_ok = {p_ok:.5f}")
    print(f"  (P_min={P_min}% → threshold={threshold}, max_broken={max_broken})")

    L_broken = sum(j * probs[j] for j in range(N + 1))
    L_obsl = 1 - probs[0]                    # среднее число в обслуживании
    L_q = L_broken - L_obsl                  # среднее число в очереди
    A = mu * L_obsl                           # абс. пропускная способность, зявок/мес
    W_obsl = t_days                           # среднее время обслуживания, дни
    W_q = (L_q / A) * 30 if A > 0 else 0    # среднее время ожидания, дни

    print(f"\nСреднее число неисправных:      L = {L_broken:.5f}")
    print(f"Из них в обслуживании:     L_обсл = {L_obsl:.5f}")
    print(f"Из них в очереди:              L_q = {L_q:.5f}")
    print(f"Абс. пропускная способность:    A = {A:.5f} зявок/мес")
    print(f"Среднее время обслуживания:  W_обсл = {W_obsl:.3f} дней")
    print(f"Среднее время ожидания:         W_q = {W_q:.3f} дней")

    print(f"\nВыводы: P(активных ≥{threshold})={p_ok:.4f}."
          f"\n  В очереди в среднем {L_q:.3f} машин, ожидание {W_q:.2f} дней.")


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 6. Замкнутая многоканальная СМО (M/M/k//N)
# ═══════════════════════════════════════════════════════════════

def task6():
    section("Задача 6. M/M/k//N — замкнутая многоканальная СМО")

    k = 3         # каналов
    N = 12        # источников
    lam_src = 1.4 # зявок/ч от одного активного источника
    t = 0.2       # ч (время обслуживания)

    mu = 1.0 / t
    rho0 = lam_src / mu    # нагрузка от одного источника (λ/μ)

    print(f"\nПараметры: k={k} кан., N={N} ист., λ={lam_src} зявок/ч, t={t} ч")
    print(f"Расчётные: μ={mu:.4f} зявок/ч, ρ₀=λ/μ={rho0:.4f}")

    # P(j): ненормированные веса (из уравнений локального баланса)
    # j≤k: P(j)=P(0)×C(N,j)×aʲ; j>k: P(j)=P(0)×C(N,j)×j!×aʲ/(k!×kʲ⁻ᵏ)
    def unnorm_p6(j):
        if j <= k:
            return math.comb(N, j) * rho0**j
        return math.comb(N, j) * math.factorial(j) * rho0**j / (math.factorial(k) * k**(j - k))

    Z = sum(unnorm_p6(j) for j in range(N + 1))
    probs = [unnorm_p6(j) / Z for j in range(N + 1)]

    print("\nПредельные вероятности P(j):")
    for j, p in enumerate(probs):
        print(f"  P({j:2d}) = {p:.5f}")

    p0 = probs[0]
    L_q = sum((j - k) * probs[j] for j in range(k + 1, N + 1))
    L_obsl = sum(min(j, k) * probs[j] for j in range(1, N + 1))
    L = sum(j * probs[j] for j in range(N + 1))
    avg_free_ch = sum((k - j) * probs[j] for j in range(k + 1))
    P_queue = sum(probs[j] for j in range(k + 1, N + 1))

    A = mu * L_obsl                     # абс. пропускная способность, зявок/ч
    Q = A / (N * lam_src) if N > 0 else 0   # относительная пропускная способность

    W_q = (L_q / A) * 60 if A > 0 else 0    # мин
    W_obsl = t * 60                           # мин
    W = (L / A) * 60 if A > 0 else 0         # мин

    print(f"\n1. Вероятность простоя всех каналов: P(0) = {p0:.5f}")
    print(f"2. Среднее число заявок в очереди:   L_q = {L_q:.5f}")
    print(f"3. Среднее число заявок в системе:     L = {L:.5f}")
    print(f"4. Среднее число свободных каналов:   {avg_free_ch:.5f}")
    print(f"5. Среднее число занятых каналов: L_обсл = {L_obsl:.5f}")
    print(f"6. Вероятность наличия очереди:   P_оч = {P_queue:.5f}")
    print(f"7. Абс. пропускная способность:        A = {A:.5f} зявок/ч")
    print(f"8. Отн. пропускная способность:        Q = {Q:.5f}")
    print(f"9. Среднее время ожидания:             W_q = {W_q:.3f} мин")
    print(f"10. Среднее время обслуживания:     W_обсл = {W_obsl:.3f} мин")
    print(f"11. Среднее время пребывания:           W = {W:.3f} мин")

    # График P(j)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(N + 1), probs, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Число занятых мест j (в очереди + обслуживании)')
    ax.set_ylabel('P(j)')
    ax.set_title('Задача 6. Распределение вероятностей состояний M/M/k//N')
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(_HERE / 'task6_plot.png', dpi=120)
    plt.close()
    print("\n    [График сохранён: task6_plot.png]")

    print(f"\nВыводы: Q={Q:.4f} — относительная пропускная способность."
          f"\n  Очередь возникает с вероятностью {P_queue:.4f}."
          f"\n  Среднее ожидание {W_q:.2f} мин, время пребывания {W:.2f} мин.")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Лабораторная работа №2 «Анализ СМО»                    ║")
    print("║  Лобанов Р. Е.                                           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    task1()
    task2()
    task3()
    task4()
    task5()
    task6()

    print("\n" + "=" * 60)
    print("  Все задачи выполнены.")
    print("=" * 60)
