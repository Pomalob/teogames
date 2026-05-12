import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

np.random.seed(42)

NODES = [
    {'name': 'Web Server',      'V': 500,  'C': 80,  'P_def': 0.05, 'P_no_def': 0.85},
    {'name': 'Auth Service',    'V': 800,  'C': 120, 'P_def': 0.03, 'P_no_def': 0.90},
    {'name': 'Database',        'V': 1000, 'C': 150, 'P_def': 0.02, 'P_no_def': 0.95},
    {'name': 'API Gateway',     'V': 600,  'C': 90,  'P_def': 0.04, 'P_no_def': 0.80},
    {'name': 'File Storage',    'V': 700,  'C': 100, 'P_def': 0.06, 'P_no_def': 0.75},
    {'name': 'Monitoring Node', 'V': 300,  'C': 50,  'P_def': 0.10, 'P_no_def': 0.60},
]

BUDGET = 300
N = len(NODES)
V        = np.array([nd['V']       for nd in NODES])
C        = np.array([nd['C']       for nd in NODES])
P_def    = np.array([nd['P_def']   for nd in NODES])
P_no_def = np.array([nd['P_no_def'] for nd in NODES])


def solve_stackelberg(V, C, P_def, P_no_def, B):
    num_vars = N + 1
    c_obj = np.zeros(num_vars)
    c_obj[-1] = 1.0

    A_ub, b_ub = [], []
    for i in range(N):
        row = np.zeros(num_vars)
        row[i] = V[i] * (P_def[i] - P_no_def[i])
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(-V[i] * P_no_def[i])

    budget_row = np.zeros(num_vars)
    budget_row[:N] = C
    A_ub.append(budget_row)
    b_ub.append(float(B))

    result = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=[(0.0, 1.0)] * N + [(0.0, None)], method='highs')
    if not result.success:
        raise RuntimeError("LP: " + result.message)
    return result.x[:N], result.x[-1]


p_opt, d_opt = solve_stackelberg(V, C, P_def, P_no_def, BUDGET)
damages      = V * (p_opt * P_def + (1.0 - p_opt) * P_no_def)
damages_none = V * P_no_def
target       = int(np.argmax(damages))

print("=== Равновесие Штакельберга ===")
print(f"{'Узел':<22} {'P(защита)':>10} {'Бюджет':>10} {'Ущерб (opt)':>13} {'Ущерб (нет)':>13}")
print("-" * 72)
for i, nd in enumerate(NODES):
    mark = " <-- цель" if i == target else ""
    print(f"{nd['name']:<22} {p_opt[i]:>10.4f} {C[i]*p_opt[i]:>10.2f} {damages[i]:>13.2f} {damages_none[i]:>13.2f}{mark}")

print(f"\nБюджет использован: {np.dot(p_opt, C):.2f} / {BUDGET}")
print(f"Цена игры (мин. ожид. ущерб): {d_opt:.2f}")
print(f"Снижение макс. ущерба: {100*(damages_none.max()-d_opt)/damages_none.max():.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names = [nd['name'] for nd in NODES]
x = np.arange(N)
w = 0.35

ax1 = axes[0]
bars = ax1.bar(x, p_opt, color='steelblue', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
ax1.set_ylabel('Вероятность защиты')
ax1.set_title('Оптимальное распределение защиты')
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, p_opt):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=9)

ax2 = axes[1]
ax2.bar(x - w/2, damages_none, width=w, label='Без защиты', color='tomato', alpha=0.8)
ax2.bar(x + w/2, damages,      width=w, label='С защитой',  color='mediumseagreen', alpha=0.8)
ax2.axhline(y=d_opt, color='navy', linestyle='--', linewidth=1.5,
            label=f'Цена игры = {d_opt:.1f}')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
ax2.set_ylabel('Ожидаемый ущерб')
ax2.set_title('Сравнение ожидаемого ущерба')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Игра Штакельберга — защита облачной инфраструктуры', fontsize=13)
plt.tight_layout()
plt.savefig('task2_stackelberg.png', dpi=150)
plt.show()
