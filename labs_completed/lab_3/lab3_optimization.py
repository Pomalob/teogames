# -*- coding: utf-8 -*-
"""
Лабораторная работа №3 «Методы оптимизации и динамическое программирование»
Студент: Лобанов Р. Е., Вариант 5

Задача 1. Линейное программирование (scipy.optimize.linprog + двойственная задача)
Задача 2. Нелинейное ДП: оптимизация распределения нагрузки (N=5, X=10, h=1)
Задача 3. Алгоритм Джонсона (8 задач, ETL-пайплайн)
Задача 4. Замена оборудования (Беллман, N=5, t0=2)
Задача 5. Распределение ресурсов (ДП, Z=14, N=3 квартала)
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
#  ЗАДАЧА 1. Линейное программирование (Вариант 5)
#  F = 11x1+14x2+9x3+13x4+12x5+17x6 → max
# ═══════════════════════════════════════════════════════════════

def task1():
    section('Задача 1. ЛП: планирование выпуска микросервисов (Вариант 5)')

    # Коэффициенты целевой функции (минимизируем −F)
    c = [-11, -14, -9, -13, -12, -17]

    # Ограничения ≤ (A_ub × x ≤ b_ub)
    # Ограничения ≥ умножаем на −1
    A_ub = [
        [ 3,  4,  2,  5,  2,  6],   # DevOps-часы ≤ 210
        [ 2,  3,  1,  4,  1,  5],   # бюджет ≤ 150
        [15, 20, 10, 25,  8, 30],   # вычисл. квоты ≤ 800
        [-1, -1, -1,  0,  0,  0],   # x1+x2+x3 ≥ 14  → −(x1+x2+x3) ≤ −14
        [ 0,  0,  0, -1, -1, -1],   # x4+x5+x6 ≥ 11  → -(x4+x5+x6) ≤ −11
        [ 0,  0,  0,  1,  0,  1],   # x4+x6 ≤ 16
    ]
    b_ub = [210, 150, 800, -14, -11, 16]
    bounds = [(0, None)] * 6

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    x = res.x
    F_opt = -res.fun

    print(f'\nПостановка: F = 11x1+14x2+9x3+13x4+12x5+17x6 → max')
    print(f'Оптимальное решение (прямая задача):')
    for i, xi in enumerate(x, 1):
        print(f'  x{i} = {xi:.4f}')
    print(f'  F* = {F_opt:.4f} млн руб./квартал')

    # Двойственные оценки (теневые цены) по ограничениям
    shadow = res.ineqlin.marginals   # dual values (negated for max problem)
    print('\nТеневые цены (двойственные переменные):')
    names = ['DevOps-часы', 'Бюджет облако', 'Вычисл. квоты',
             'x1+x2+x3≥14', 'x4+x5+x6≥11', 'x4+x6≤16']
    for name, s in zip(names, shadow):
        print(f'  {name:20s}: {s:.4f}')

    # Анализ чувствительности: изменение бюджета ±15%
    print('\nАнализ чувствительности: изменение ограничений на ±15%:')
    print(f'  {"Ограничение":20s} | {"−15%":>10} | {"Базовое":>10} | {"+15%":>10}')
    print(f'  {"-"*20}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}')
    for idx, (name, b_base) in enumerate(zip(names, b_ub)):
        results = []
        for factor in [0.85, 1.0, 1.15]:
            b_mod = b_ub.copy()
            b_mod[idx] = b_base * factor
            r = linprog(c, A_ub=A_ub, b_ub=b_mod, bounds=bounds, method='highs')
            results.append(-r.fun if r.status == 0 else float('nan'))
        print(f'  {name:20s} | {results[0]:>10.3f} | {results[1]:>10.3f} | {results[2]:>10.3f}')

    print('\nВыводы:')
    print(f'  Оптимальная маржа = {F_opt:.2f} млн руб./квартал.')
    print('  Наиболее дефицитные ресурсы — те, у которых теневая цена > 0.')
    print('  Увеличение бюджета на 15% позволяет растить маржу.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 2. Нелинейное ДП: оптимизация распределения нагрузки
#  Fk(X) = min_{0≤Y≤X} [3Y² + 4(X−Y)² + F_{k-1}(0.45Y + 0.55(X−Y))]
#  N=5, X=10, h=1
# ═══════════════════════════════════════════════════════════════

def task2():
    section('Задача 2. Нелинейное ДП: распределение нагрузки (N=5, X=10)')

    N = 5
    X_start = 10
    h = 1
    X_grid = np.arange(0, X_start + h, h, dtype=float)   # [0,1,...,10]

    def stage_cost(Y, X):
        return 3 * Y**2 + 4 * (X - Y)**2

    def next_state(Y, X):
        return 0.45 * Y + 0.55 * (X - Y)   # = 0.55X − 0.1Y

    # Stage 1 (последний этап): F1(X) = min_{0≤Y≤X} [3Y²+4(X−Y)²]
    F_vals = [None] * (N + 1)   # F_vals[k][i] = F_k(X_grid[i])
    Y_opt  = [None] * (N + 1)   # optimal Y at each stage / grid point

    # k=1
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

    # k=2,...,N
    for k in range(2, N + 1):
        Fk = np.zeros(len(X_grid))
        Yk = np.zeros(len(X_grid))
        F_interp = interp1d(X_grid, F_vals[k - 1],
                            kind='linear', fill_value='extrapolate')

        for i, X in enumerate(X_grid):
            if X == 0:
                continue

            def obj(Y):
                nxt = next_state(Y, X)
                nxt = max(0.0, nxt)   # clip at 0
                return stage_cost(Y, X) + F_interp(nxt)

            res = minimize_scalar(obj, bounds=(0, X), method='bounded')
            Yk[i] = res.x
            Fk[i] = res.fun

        F_vals[k] = Fk
        Y_opt[k]  = Yk
        print(f'\nF_{k}(X):')
        for X, F, Y in zip(X_grid, Fk, Yk):
            print(f'  X={X:.0f}: F{k}={F:.4f}, Y*={Y:.4f}')

    # Восстановление оптимальной траектории при X=X_start
    print(f'\n--- Оптимальная траектория, начало X={X_start} ---')
    print(f'  Этап | X_вход  | Y*     | X_выход | Затраты_этапа')
    X_cur = float(X_start)
    total_cost = 0.0
    F_interp_N = interp1d(X_grid, F_vals[N], kind='linear', fill_value='extrapolate')
    print(f'  F_{N}({X_start}) = {F_interp_N(X_start):.4f} (оптимальные суммарные затраты)')

    for k in range(N, 0, -1):
        F_interp = interp1d(X_grid, F_vals[k], kind='linear', fill_value='extrapolate')
        if X_cur <= 0:
            break
        Ystar_res = minimize_scalar(
            lambda Y: stage_cost(Y, X_cur) + (
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

    # График F_k(X)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(1, N + 1):
        ax.plot(X_grid, F_vals[k], 'o-', label=f'F_{k}(X)')
    ax.set_xlabel('X (ресурс)'), ax.set_ylabel('Оптимальные затраты F_k(X)')
    ax.set_title('Задача 2. Функции Белмана F_k(X) для нелинейного ДП')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'task2_dp_plot.png', dpi=120)
    plt.close()
    print('  [График сохранён: task2_dp_plot.png]')

    print('\nВыводы: Нелинейное ДП позволило найти оптимальное распределение'
          '\n  нагрузки по 5 этапам при минимизации суммарных затрат.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 3. Задача Джонсона: ETL-пайплайн (8 задач)
#  A=[4,8,2,6,3,7,5,9], B=[6,3,7,2,8,4,1,5]
# ═══════════════════════════════════════════════════════════════

def task3():
    section('Задача 3. Алгоритм Джонсона: ETL-пайплайн (8 задач)')

    A = [4, 8, 2, 6, 3, 7, 5, 9]   # время извлечения (стадия 1)
    B = [6, 3, 7, 2, 8, 4, 1, 5]   # время загрузки (стадия 2)
    n = len(A)

    print(f'\nЗадачи: A={A}')
    print(f'        B={B}')

    def johnson(A, B):
        jobs = list(range(len(A)))
        front, back = deque(), deque()
        while jobs:
            min_val = min(min(A[j], B[j]) for j in jobs)
            # Обрабатываем ВСЕ задачи с текущим минимумом (не только первую)
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
        t_a = 0  # end time on machine A
        t_b = 0  # end time on machine B
        schedule = []
        for j in order:
            t_a += A[j]
            t_b = max(t_b, t_a) + B[j]
            schedule.append((j + 1, t_a, t_b))
        return t_b, schedule

    # Порядок по Джонсону
    order = johnson(A, B)
    ms_j, sched_j = makespan(order, A, B)

    print(f'\nОптимальный порядок (Джонсон): {[j+1 for j in order]}')
    print(f'Таблица расписания:')
    print(f'  {"Задача":>6} | {"A[j]":>5} | {"B[j]":>5} | {"Конец A":>8} | {"Конец B":>8}')
    print(f'  {"-"*6}-+-{"-"*5}-+-{"-"*5}-+-{"-"*8}-+-{"-"*8}')
    for (j, ea, eb) in sched_j:
        print(f'  {j:6d} | {A[j-1]:5d} | {B[j-1]:5d} | {ea:8d} | {eb:8d}')
    print(f'  Makespan (Джонсон): {ms_j}')

    # Простой машины B
    t_a_end = 0
    idle_B = 0
    t_b_prev = 0
    for j in order:
        t_a_end += A[j]
        idle_B += max(0, t_a_end - t_b_prev)
        t_b_prev = max(t_b_prev, t_a_end) + B[j]
    print(f'  Простой машины B: {idle_B} мин')

    # Исходный порядок для сравнения
    original_order = list(range(n))
    ms_orig, _ = makespan(original_order, A, B)
    print(f'\nМейкспэн в исходном порядке (1..8): {ms_orig}')
    print(f'Сокращение мейкспэна: {ms_orig - ms_j} мин ({(ms_orig-ms_j)/ms_orig*100:.1f}%)')

    # Диаграмма Ганта
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
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Машина B (загрузка)', 'Машина A (извлечение)'])
        ax.set_title(title); ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'task3_gantt.png', dpi=120)
    plt.close()
    print('  [Диаграмма Ганта сохранена: task3_gantt.png]')

    print(f'\nВыводы: Алгоритм Джонсона сокращает makespan с {ms_orig} до {ms_j} мин'
          f'\n  ({(ms_orig-ms_j)/ms_orig*100:.1f}% экономии). Простой машины B = {idle_B} мин.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 4. Замена оборудования (Беллман)
#  R(t)=28−2.8t, C(t)=4.5+3.2t, N=5, Tmax=5, t0=2
# ═══════════════════════════════════════════════════════════════

def task4():
    section('Задача 4. Замена оборудования (Беллман), N=5, t0=2')

    N    = 5
    Tmax = 5
    t0   = 2

    def R(t): return 28 - 2.8 * t   # прибыль от эксплуатации
    def C(t): return 4.5 + 3.2 * t  # стоимость замены в год t

    ages = list(range(Tmax + 1))     # 0,1,...,5

    # Таблица F_k(t): max доход за k оставшихся лет при возрасте t
    # F_0(t) = 0 (нет лет)
    F = {0: {t: 0.0 for t in ages}}
    policy = {}   # policy[k][t] = 'K' или 'R'

    for k in range(1, N + 1):
        F[k] = {}
        policy[k] = {}
        for t in ages:
            # KEEP (оставить): прибыль R(t) + F_{k-1}(t+1)
            keep_val = R(t) + F[k-1].get(t + 1, float('-inf')) if (t + 1) <= Tmax else float('-inf')

            # REPLACE (заменить): −C(t)+R(0)+F_{k-1}(1)
            # При замене: платим C(t), получаем новое (возраст 0), за год → возраст 1
            replace_val = -C(t) + R(0) + F[k-1].get(1, 0.0)

            if keep_val >= replace_val and keep_val != float('-inf'):
                F[k][t] = keep_val
                policy[k][t] = 'K'
            else:
                F[k][t] = replace_val
                policy[k][t] = 'R'

    print('\nТаблица F_k(t) (макс. доход за k лет, возраст t):')
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

    # Восстановление оптимальной траектории от t0=2, N=5 лет
    print(f'\nОптимальная траектория: t0={t0}, горизонт N={N}:')
    print(f'  {"Год":>4} | {"Возраст":>7} | {"Решение":>8} | {"Прибыль":>8}')
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
        print(f'  {N-k+1:4d} | t={t_cur:5d} | {dec:>8s} | {profit:8.2f}')
        total_profit += profit
        t_cur = t_next

    print(f'  Итоговая прибыль: {total_profit:.2f}')
    print(f'  (F_{N}({t0}) = {F[N].get(t0, "n/a"):.2f})')

    print('\nВыводы: Беллман позволил оптимально планировать замены оборудования.'
          '\n  Замена выгодна, когда накопленный износ перекрывает стоимость рефакторинга.')


# ═══════════════════════════════════════════════════════════════
#  ЗАДАЧА 5. Распределение ресурсов (ДП)
#  Z=14, N=3, g(Y)=5Y+0.1Y², h(X−Y)=3(X−Y)+0.18(X−Y)²
#  α=0.65, β=0.35, Сетка {0,2,4,...,14}, ΔY=2
# ═══════════════════════════════════════════════════════════════

def task5():
    section('Задача 5. Распределение ресурсов (ДП), Z=14, N=3 квартала')

    Z    = 14
    N    = 3
    alp  = 0.65   # коэффициент возврата из g-проекта
    bet  = 0.35   # коэффициент возврата из h-проекта
    dY   = 2
    grid = np.arange(0, Z + dY, dY, dtype=float)   # {0,2,4,...,14}

    def g(Y):     return 5 * Y + 0.1 * Y**2
    def h(Xm):    return 3 * Xm + 0.18 * Xm**2   # Xm = X − Y

    print(f'\nСетка X: {[int(x) for x in grid]}')
    print(f'g(Y) = 5Y + 0.1Y², h(X−Y) = 3(X−Y) + 0.18(X−Y)²')
    print(f'Возврат: α={alp}, β={bet}; следующий X = α·Y + β·(X−Y)')

    def next_X(Y, X):
        return alp * Y + bet * (X - Y)

    # F_k(X): макс. доход за k кварталов при бюджете X
    # F_1(X): только один квартал — распределяем X между g и h
    F = {}
    Ystar = {}

    # k=1
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

    # k=2,...,N
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

    # Вывод таблиц F_k и Y*_k
    for k in range(1, N + 1):
        print(f'\nF_{k}(X) и Y*_{k}(X):')
        print(f'  {"X":>4} | {"F_k":>8} | {"Y*":>6} | {"X−Y*":>6} | {"след.X":>7}')
        print(f'  {"-"*4}-+-{"-"*8}-+-{"-"*6}-+-{"-"*6}-+-{"-"*7}')
        for X in grid:
            Xf = float(X)
            y  = Ystar[k][Xf]
            nxt = next_X(y, Xf) if k < N else 0.0
            print(f'  {int(X):4d} | {F[k][Xf]:8.3f} | {y:6.1f} | {Xf-y:6.1f} | {nxt:7.3f}')

    # Восстановление оптимальной траектории с Z=14
    # При X вне сетки: ищем Y оптимальный по интерполированной функции ценности
    def best_y_for_x(k, X_c, F_prev_vals):
        """Оптимальный Y для произвольного X_c (не обязательно на сетке)."""
        best_v, best_Y = -np.inf, 0.0
        yd = dY  # шаг дискретизации Y
        F_interp = interp1d(
            [float(x) for x in grid], F_prev_vals,
            kind='linear', fill_value='extrapolate')
        for Y in np.arange(0, X_c + yd, yd):
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

    print(f'\n  Суммарный доход: {total_income:.3f}')
    print(f'  (F_{N}({Z}) = {F[N][float(Z)]:.3f})')

    print('\nВыводы: ДП на сетке позволило найти оптимальное квартальное распределение'
          '\n  бюджета между g- и h-проектами с учётом возврата инвестиций.')


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('╔══════════════════════════════════════════════════════════╗')
    print('║  Лабораторная работа №3. Методы оптимизации и ДП        ║')
    print('║  Лобанов Р. Е., Вариант 5                                ║')
    print('╚══════════════════════════════════════════════════════════╝')

    task1()
    task2()
    task3()
    task4()
    task5()

    print('\n' + '=' * 62)
    print('  Все задачи выполнены.')
    print('=' * 62)
