# -*- coding: utf-8 -*-
"""
Лабораторная работа №3 «Методы оптимизации и динамическое программирование»
Студент: Батраков Я. А., Вариант 6

Задача 1. Линейное программирование (scipy.optimize.linprog + двойственная задача)
  F = 12x1+10x2+13x3+9x4+16x5+11x6 → max

Задача 2. Нелинейное ДП: оптимизация распределения нагрузки
  F1(X) = min[2Y²+5(X−Y)²], Fk = min[2Y²+5(X−Y)²+Fk-1(0.6Y+0.4(X−Y))]
  N=3, X=9, h=1

Задача 3. Алгоритм Джонсона (7 задач, CI/CD конвейер)
  A=[7,1,4,6,2,9,5], B=[3,8,5,2,7,4,6]

Задача 4. Замена оборудования (Беллман)
  R(t)=32−4t, C(t)=3+3t, N=4 сезона, Tmax=6, t0=1

Задача 5. Распределение ресурсов (ДП)
  Z=18, N=3, g(Y)=7Y+0.12Y², h(X−Y)=4(X−Y)+0.16(X−Y)²
  α=0.5, β=0.5, сетка {0,3,6,...,18}, ΔY=3
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize_scalar
from scipy.interpolate import interp1d
from collections import deque
from pathlib import Path

SAVE_DIR = Path(__file__).parent


def section(title: str):
    print('\n' + '=' * 62)
    print(f'  {title}')
    print('=' * 62)


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 1. Линейное программирование (Вариант 6)
#  F = 12x1+10x2+13x3+9x4+16x5+11x6 → max
# ═══════════════════════════════════════════════════════════════

def task1():
    section('Задача 1. ЛП: планирование выпуска микросервисов (Вариант 6)')

    c = [-12, -10, -13, -9, -16, -11]

    A_ub = [
        [ 3,  4,  2,  5,  2,  6],   # DevOps-часы ≤ 230
        [ 2,  3,  1,  4,  1,  5],   # бюджет ≤ 170
        [15, 20, 10, 25,  8, 30],   # вычисл. квоты ≤ 950
        [-1, -1, -1,  0,  0,  0],   # x1+x2+x3 ≥ 16  → -(x1+x2+x3) ≤ -16
        [ 0,  0,  0, -1, -1, -1],   # x4+x5+x6 ≥ 13  → -(x4+x5+x6) ≤ -13
        [ 1,  0,  0,  1,  0,  0],   # x1+x4 ≤ 18
    ]
    b_ub = [230, 170, 950, -16, -13, 18]
    bounds = [(0, None)] * 6

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    x = res.x
    F_opt = -res.fun

    print(f'\nПостановка: F = 12x1+10x2+13x3+9x4+16x5+11x6 → max')
    print(f'Оптимальное решение (прямая задача):')
    for i, xi in enumerate(x, 1):
        print(f'  x{i} = {xi:.4f}')
    print(f'  F* = {F_opt:.4f} млн руб./квартал')

    shadow = [-m for m in res.ineqlin.marginals]   # для задачи max: знак dual переворачивается
    print('\nТеневые цены (двойственные переменные):')
    names = ['DevOps-часы', 'Бюджет облако', 'Вычисл. квоты',
             'x1+x2+x3≥16', 'x4+x5+x6≥13', 'x1+x4≤18']
    for name, s in zip(names, shadow):
        print(f'  {name:20s}: {s:.4f}')

    print('\nАнализ чувствительности: изменение ограничений на ±15%:')
    print(f'  {"Ограничение":20s} | {"−15%":>10} | {"Базовое":>10} | {"+15%":>10}')
    print(f'  {"-"*20}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}')
    for idx, (name, b_base) in enumerate(zip(names, b_ub)):
        results = []
        for factor in [0.85, 1.0, 1.15]:
            b_mod = b_ub.copy()
            # Для ≥-ограничений b_base < 0: инвертируем factor, чтобы
            # «−15%» всегда означало ужесточение (tighter), «+15%» — смягчение
            f = (2.0 - factor) if b_base < 0 else factor
            b_mod[idx] = b_base * f
            r = linprog(c, A_ub=A_ub, b_ub=b_mod, bounds=bounds, method='highs')
            results.append(-r.fun if r.status == 0 else float('nan'))
        print(f'  {name:20s} | {results[0]:>10.3f} | {results[1]:>10.3f} | {results[2]:>10.3f}')

    print('\nВыводы:')
    print(f'  Оптимальная маржа = {F_opt:.2f} млн руб./квартал.')
    print('  Наиболее дефицитные ресурсы — те, у которых теневая цена > 0.')
    print('  Увеличение бюджета на 15% позволяет растить маржу.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 2. Нелинейное ДП: оптимизация распределения нагрузки
#  Fk(X) = min_{0≤Y≤X} [2Y² + 5(X−Y)² + F_{k-1}(0.6Y + 0.4(X−Y))]
#  N=3, X=9, h=1
# ═══════════════════════════════════════════════════════════════

def task2():
    section('Задача 2. Нелинейное ДП: распределение нагрузки (N=3, X=9)')

    N = 3
    X_start = 9
    h = 1
    X_grid = np.arange(0, X_start + h, h, dtype=float)

    def stage_cost(Y, X):
        return 2 * Y**2 + 5 * (X - Y)**2

    def next_state(Y, X):
        return 0.6 * Y + 0.4 * (X - Y)   # = 0.4X + 0.2Y

    F_vals = [None] * (N + 1)
    Y_opt  = [None] * (N + 1)

    F1 = np.zeros(len(X_grid))
    Y1 = np.zeros(len(X_grid))
    for i, X in enumerate(X_grid):
        if X == 0:
            F1[i], Y1[i] = 0.0, 0.0
            continue
        res = minimize_scalar(stage_cost, args=(X,), bounds=(0, X), method='bounded')
        Y1[i] = res.x
        F1[i] = res.fun
    F_vals[1] = F1
    Y_opt[1]  = Y1

    print('\nF_1(X) — базовый этап:')
    for X, F, Y in zip(X_grid, F1, Y1):
        print(f'  X={X:.0f}: F1={F:.4f}, Y*={Y:.4f}')

    for k in range(2, N + 1):
        Fk = np.zeros(len(X_grid))
        Yk = np.zeros(len(X_grid))
        F_interp = interp1d(X_grid, F_vals[k - 1],
                            kind='linear', fill_value='extrapolate')

        for i, X in enumerate(X_grid):
            if X == 0:
                continue

            def obj(Y, X=X):
                nxt = next_state(Y, X)
                nxt = max(0.0, nxt)
                return stage_cost(Y, X) + F_interp(nxt)

            res = minimize_scalar(obj, bounds=(0, X), method='bounded')
            Yk[i] = res.x
            Fk[i] = res.fun

        F_vals[k] = Fk
        Y_opt[k]  = Yk
        print(f'\nF_{k}(X):')
        for X, F, Y in zip(X_grid, Fk, Yk):
            print(f'  X={X:.0f}: F{k}={F:.4f}, Y*={Y:.4f}')

    print(f'\n--- Оптимальная траектория, начало X={X_start} ---')
    print(f'  Этап | X_вход  | Y*     | X_выход | Затраты_этапа')
    X_cur = float(X_start)
    total_cost = 0.0
    F_interp_N = interp1d(X_grid, F_vals[N], kind='linear', fill_value='extrapolate')
    print(f'  F_{N}({X_start}) = {F_interp_N(X_start):.4f} (оптимальные суммарные затраты)')

    for k in range(N, 0, -1):
        if X_cur <= 0:
            break
        Ystar_res = minimize_scalar(
            lambda Y, k=k, X_cur=X_cur: stage_cost(Y, X_cur) + (
                interp1d(X_grid, F_vals[k - 1], kind='linear',
                         fill_value='extrapolate')(max(0.0, next_state(Y, X_cur)))
                if k > 1 else 0.0),
            bounds=(0, X_cur), method='bounded')
        Y_star = Ystar_res.x
        cost_k = stage_cost(Y_star, X_cur)
        X_next = max(0.0, next_state(Y_star, X_cur))
        print(f'  {k:5d} | {X_cur:7.4f} | {Y_star:6.4f} | {X_next:7.4f} | {cost_k:.4f}')
        total_cost += cost_k
        X_cur = X_next

    print(f'  Сумма затрат по этапам: {total_cost:.4f}')

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(1, N + 1):
        ax.plot(X_grid, F_vals[k], 'o-', label=f'F_{k}(X)')
    ax.set_xlabel('X (ресурс)'), ax.set_ylabel('Оптимальные затраты F_k(X)')
    ax.set_title('Задача 2. Функции Беллмана F_k(X) для нелинейного ДП')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'task2_dp_plot.png', dpi=120)
    plt.close()
    print('  [График сохранён: task2_dp_plot.png]')

    print('\nВыводы: Нелинейное ДП позволило найти оптимальное распределение'
          '\n  нагрузки по 3 этапам при минимизации суммарных затрат.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 3. Задача Джонсона: CI/CD конвейер (7 задач)
#  A=[7,1,4,6,2,9,5], B=[3,8,5,2,7,4,6]
# ═══════════════════════════════════════════════════════════════

def task3():
    section('Задача 3. Алгоритм Джонсона: CI/CD конвейер (7 задач)')

    A = [7, 1, 4, 6, 2, 9, 5]   # время генерации кода (стадия 1)
    B = [3, 8, 5, 2, 7, 4, 6]   # время статического анализа (стадия 2)
    n = len(A)

    print(f'\nЗадачи: A={A}')
    print(f'        B={B}')

    def johnson(A, B):
        jobs = list(range(len(A)))
        front, back = deque(), deque()
        while jobs:
            min_val = min(min(A[j], B[j]) for j in jobs)
            remaining = []
            for j in jobs:
                if A[j] == min_val:
                    front.append(j)
                elif B[j] == min_val:
                    back.appendleft(j)
                else:
                    remaining.append(j)
            jobs = remaining
        return list(front) + list(back)

    def makespan(order, A, B):
        t_a = 0
        t_b = 0
        schedule = []
        for j in order:
            t_a += A[j]
            t_b = max(t_b, t_a) + B[j]
            schedule.append((j + 1, t_a, t_b))
        return t_b, schedule

    order = johnson(A, B)
    ms_j, sched_j = makespan(order, A, B)

    print(f'\nОптимальный порядок (Джонсон): {[j+1 for j in order]}')
    print(f'Таблица расписания:')
    print(f'  {"Задача":>6} | {"A[j]":>5} | {"B[j]":>5} | {"Конец A":>8} | {"Конец B":>8}')
    print(f'  {"-"*6}-+-{"-"*5}-+-{"-"*5}-+-{"-"*8}-+-{"-"*8}')
    for (j, ea, eb) in sched_j:
        print(f'  {j:6d} | {A[j-1]:5d} | {B[j-1]:5d} | {ea:8d} | {eb:8d}')
    print(f'  Makespan (Джонсон): {ms_j}')

    t_a_end = 0
    idle_B = 0
    t_b_prev = 0
    for j in order:
        t_a_end += A[j]
        idle_B += max(0, t_a_end - t_b_prev)
        t_b_prev = max(t_b_prev, t_a_end) + B[j]
    print(f'  Простой машины B: {idle_B} мин')

    original_order = list(range(n))
    ms_orig, _ = makespan(original_order, A, B)
    print(f'\nМейкспэн в исходном порядке (1..7): {ms_orig}')
    print(f'Сокращение мейкспэна: {ms_orig - ms_j} мин ({(ms_orig-ms_j)/ms_orig*100:.1f}%)')

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    for ax, ord_, title in [(axes[0], order, f'Порядок Джонсона (makespan={ms_j})'),
                             (axes[1], original_order, f'Исходный порядок (makespan={ms_orig})')]:
        t_a, t_b = 0, 0
        for j in ord_:
            ax.barh(1, A[j], left=t_a, color='steelblue', edgecolor='k', height=0.4)
            ax.text(t_a + A[j]/2, 1, str(j+1), ha='center', va='center', fontsize=8, color='white')
            start_b = max(t_a + A[j], t_b)
            ax.barh(0, B[j], left=start_b, color='coral', edgecolor='k', height=0.4)
            ax.text(start_b + B[j]/2, 0, str(j+1), ha='center', va='center', fontsize=8, color='white')
            t_a += A[j]
            t_b = start_b + B[j]
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Машина B (анализ)', 'Машина A (генерация)'])
        ax.set_title(title); ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'task3_gantt.png', dpi=120)
    plt.close()
    print('  [Диаграмма Ганта сохранена: task3_gantt.png]')

    print(f'\nВыводы: Алгоритм Джонсона сокращает makespan с {ms_orig} до {ms_j} мин'
          f'\n  ({(ms_orig-ms_j)/ms_orig*100:.1f}% экономии). Простой машины B = {idle_B} мин.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 4. Замена оборудования (Беллман)
#  R(t)=32−4t, C(t)=3+3t, N=4, Tmax=6, t0=1
# ═══════════════════════════════════════════════════════════════

def task4():
    section('Задача 4. Замена оборудования (Беллман), N=4, t0=1')

    N    = 4
    Tmax = 6
    t0   = 1

    def R(t): return 32 - 4 * t
    def C(t): return 3 + 3 * t

    ages = list(range(Tmax + 1))

    F = {0: {t: 0.0 for t in ages}}
    policy = {}

    for k in range(1, N + 1):
        F[k] = {}
        policy[k] = {}
        for t in ages:
            keep_val = R(t) + F[k-1].get(t + 1, float('-inf')) if (t + 1) <= Tmax else float('-inf')
            replace_val = -C(t) + R(0) + F[k-1].get(1, 0.0)

            if keep_val >= replace_val and keep_val != float('-inf'):
                F[k][t] = keep_val
                policy[k][t] = 'K'
            else:
                F[k][t] = replace_val
                policy[k][t] = 'R'

    print('\nТаблица F_k(t) (макс. доход за k сезонов, возраст t):')
    header = '  t\\k | ' + ' | '.join(f'F_{k:1d}' for k in range(1, N+1))
    print(header)
    print('  ' + '-' * len(header))
    for t in ages:
        row = f'  t={t} | ' + ' | '.join(f'{F[k].get(t, float("nan")):6.2f}' for k in range(1, N+1))
        print(row)

    print('\nОптимальная политика (K=оставить, R=заменить):')
    header2 = '  t\\k | ' + ' | '.join(f'π_{k:1d}' for k in range(1, N+1))
    print(header2)
    print('  ' + '-' * len(header2))
    for t in ages:
        row = f'  t={t} | ' + ' | '.join(f'  {policy[k].get(t, "-"):1s} ' for k in range(1, N+1))
        print(row)

    print(f'\nОптимальная траектория: t0={t0}, горизонт N={N}:')
    print(f'  {"Сезон":>6} | {"Возраст":>7} | {"Решение":>8} | {"Прибыль":>8}')
    t_cur = t0
    total_profit = 0.0
    for k in range(N, 0, -1):
        dec = policy[k].get(t_cur, 'K')
        if dec == 'K':
            profit = R(t_cur)
            t_next = t_cur + 1
        else:
            profit = -C(t_cur) + R(0)
            t_next = 1
        print(f'  {N-k+1:6d} | t={t_cur:5d} | {dec:>8s} | {profit:8.2f}')
        total_profit += profit
        t_cur = t_next

    print(f'  Итоговая прибыль: {total_profit:.2f}')
    print(f'  (F_{N}({t0}) = {F[N].get(t0, "n/a"):.2f})')

    print('\nВыводы: Беллман позволил оптимально планировать замены оборудования.'
          '\n  Замена выгодна, когда накопленный износ перекрывает стоимость замены.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 5. Распределение ресурсов (ДП)
#  Z=18, N=3, g(Y)=7Y+0.12Y², h(X−Y)=4(X−Y)+0.16(X−Y)²
#  α=0.5, β=0.5, сетка {0,3,6,...,18}, ΔY=3
# ═══════════════════════════════════════════════════════════════

def task5():
    section('Задача 5. Распределение ресурсов (ДП), Z=18, N=3 квартала')

    Z    = 18
    N    = 3
    alp  = 0.5
    bet  = 0.5
    dY   = 3
    grid = np.arange(0, Z + dY, dY, dtype=float)

    def g(Y):     return 7 * Y + 0.12 * Y**2
    def h(Xm):    return 4 * Xm + 0.16 * Xm**2

    print(f'\nСетка X: {[int(x) for x in grid]}')
    print(f'g(Y) = 7Y + 0.12Y², h(X−Y) = 4(X−Y) + 0.16(X−Y)²')
    print(f'Возврат: α={alp}, β={bet}; следующий X = α·Y + β·(X−Y) = 0.5X')

    def next_X(Y, X):
        return alp * Y + bet * (X - Y)   # = 0.5X (т.к. α=β=0.5)

    F = {}
    Ystar = {}

    F[1] = {}
    Ystar[1] = {}
    for X in grid:
        best_v, best_Y = -np.inf, 0.0
        for Y in np.arange(0, X + dY, dY):
            if Y > X + 1e-9:
                continue
            v = g(Y) + h(X - Y)
            if v > best_v:
                best_v, best_Y = v, Y
        F[1][float(X)] = best_v
        Ystar[1][float(X)] = best_Y

    for k in range(2, N + 1):
        F[k] = {}
        Ystar[k] = {}
        F_prev_interp = interp1d(
            [float(x) for x in grid],
            [F[k-1][float(x)] for x in grid],
            kind='linear', fill_value='extrapolate'
        )
        for X in grid:
            best_v, best_Y = -np.inf, 0.0
            for Y in np.arange(0, X + dY, dY):
                if Y > X + 1e-9:
                    continue
                nxt = next_X(Y, X)
                nxt = max(0.0, min(float(Z), nxt))
                v = g(Y) + h(X - Y) + F_prev_interp(nxt)
                if v > best_v:
                    best_v, best_Y = v, Y
            F[k][float(X)] = best_v
            Ystar[k][float(X)] = best_Y

    for k in range(1, N + 1):
        print(f'\nF_{k}(X) и Y*_{k}(X):')
        print(f'  {"X":>4} | {"F_k":>8} | {"Y*":>6} | {"X−Y*":>6} | {"след.X":>7}')
        print(f'  {"-"*4}-+-{"-"*8}-+-{"-"*6}-+-{"-"*6}-+-{"-"*7}')
        for X in grid:
            Xf = float(X)
            y  = Ystar[k][Xf]
            nxt = next_X(y, Xf) if k < N else 0.0
            print(f'  {int(X):4d} | {F[k][Xf]:8.3f} | {y:6.1f} | {Xf-y:6.1f} | {nxt:7.3f}')

    def best_y_for_x(k, X_c, F_prev_vals):
        best_v, best_Y = -np.inf, 0.0
        F_interp = interp1d(
            [float(x) for x in grid], F_prev_vals,
            kind='linear', fill_value='extrapolate')
        for Y in np.arange(0, X_c + dY, dY):
            if Y > X_c + 1e-9:
                break
            nxt = max(0.0, min(float(Z), next_X(Y, X_c)))
            v = g(Y) + h(X_c - Y) + (F_interp(nxt) if k > 1 else 0.0)
            if v > best_v:
                best_v, best_Y = v, Y
        return best_Y, best_v

    print(f'\nОптимальная траектория: Z={Z}, N={N} кварталов:')
    print(f'  {"Квартал":>8} | {"Бюджет X":>9} | {"Y* (на g)":>10} | {"X−Y* (на h)":>11} | {"Доход":>8} | {"след.X":>7}')
    X_cur = float(Z)
    total_income = 0.0
    for k in range(N, 0, -1):
        F_prev_vals = [F[k-1][float(x)] for x in grid] if k > 1 else [0.0]*len(grid)
        y, _ = best_y_for_x(k, X_cur, F_prev_vals)
        income = g(y) + h(X_cur - y)
        X_next = next_X(y, X_cur) if k > 1 else 0.0
        print(f'  {N-k+1:8d} | {X_cur:9.3f} | {y:10.3f} | {X_cur-y:11.3f} | {income:8.3f} | {X_next:7.3f}')
        total_income += income
        X_cur = X_next

    dp_val = F[N][float(Z)]
    print(f'\n  Суммарный доход по траектории: {total_income:.3f}')
    print(f'  Оптимум ДП: F_{N}({Z}) = {dp_val:.3f}')
    if abs(dp_val - total_income) > 1e-6:
        print(f'  Разница {dp_val - total_income:.3f} — артефакт сетки: F вычислялась с интерполяцией,'
              f' а траектория использует дискретный шаг ΔY={dY} на off-grid X.')

    print('\nВыводы: ДП на сетке позволило найти оптимальное квартальное распределение'
          '\n  бюджета между g- и h-проектами с учётом возврата инвестиций.')


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('╔══════════════════════════════════════════════════════════╗')
    print('║  Лабораторная работа №3. Методы оптимизации и ДП        ║')
    print('║  Батраков Я. А., Вариант 6                               ║')
    print('╚══════════════════════════════════════════════════════════╝')

    task1()
    task2()
    task3()
    task4()
    task5()

    print('\n' + '=' * 62)
    print('  Все задачи выполнены.')
    print('=' * 62)
