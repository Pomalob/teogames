import numpy as np
import matplotlib.pyplot as plt

TREE = {
    'Быстрый релиз': {
        'Стабильная нагрузка': {
            'prob': 0.50,
            'outcomes': {
                'Успех':           {'prob': 0.70, 'value':  200_000},
                'Частичный успех': {'prob': 0.20, 'value':   50_000},
                'Провал':          {'prob': 0.10, 'value':  -80_000},
            },
        },
        'Пиковая нагрузка': {
            'prob': 0.30,
            'outcomes': {
                'Успех':           {'prob': 0.30, 'value':  150_000},
                'Частичный успех': {'prob': 0.40, 'value':   20_000},
                'Провал':          {'prob': 0.30, 'value': -150_000},
            },
        },
        'Сбой зависимости': {
            'prob': 0.20,
            'outcomes': {
                'Успех':           {'prob': 0.10, 'value':   80_000},
                'Частичный успех': {'prob': 0.30, 'value':  -20_000},
                'Провал':          {'prob': 0.60, 'value': -200_000},
            },
        },
    },
    'Тестирование': {
        'Стабильная нагрузка': {
            'prob': 0.50,
            'outcomes': {
                'Успех':           {'prob': 0.85, 'value':  170_000},
                'Частичный успех': {'prob': 0.10, 'value':   40_000},
                'Провал':          {'prob': 0.05, 'value':  -30_000},
            },
        },
        'Пиковая нагрузка': {
            'prob': 0.30,
            'outcomes': {
                'Успех':           {'prob': 0.60, 'value':  130_000},
                'Частичный успех': {'prob': 0.30, 'value':   10_000},
                'Провал':          {'prob': 0.10, 'value':  -80_000},
            },
        },
        'Сбой зависимости': {
            'prob': 0.20,
            'outcomes': {
                'Успех':           {'prob': 0.40, 'value':   60_000},
                'Частичный успех': {'prob': 0.35, 'value':  -10_000},
                'Провал':          {'prob': 0.25, 'value': -100_000},
            },
        },
    },
    'Отмена': {
        'Стабильная нагрузка': {
            'prob': 0.50,
            'outcomes': {'Провал': {'prob': 1.00, 'value': -15_000}},
        },
        'Пиковая нагрузка': {
            'prob': 0.30,
            'outcomes': {'Провал': {'prob': 1.00, 'value': -15_000}},
        },
        'Сбой зависимости': {
            'prob': 0.20,
            'outcomes': {'Провал': {'prob': 1.00, 'value': -15_000}},
        },
    },
}


def calculate_emv_and_risk(tree):
    results = {}
    for strategy, environments in tree.items():
        leaves = []
        for env_data in environments.values():
            for od in env_data['outcomes'].values():
                jp = env_data['prob'] * od['prob']
                if jp > 0:
                    leaves.append((jp, od['value']))
        emv = sum(p * v for p, v in leaves)
        sigma = np.sqrt(sum(p * (v - emv) ** 2 for p, v in leaves))
        results[strategy] = {'emv': emv, 'sigma': sigma, 'leaves': leaves}
    return results


results = calculate_emv_and_risk(TREE)

print(f"{'Стратегия':<22} {'EMV, руб.':>14} {'sigma, руб.':>14} {'EMV/sigma':>10}")
print("-" * 64)
for s, d in results.items():
    ratio = d['emv'] / d['sigma'] if d['sigma'] > 0 else 0
    print(f"{s:<22} {d['emv']:>14,.0f} {d['sigma']:>14,.0f} {ratio:>10.3f}")

best = max(results, key=lambda s: results[s]['emv'] / results[s]['sigma']
           if results[s]['sigma'] > 0 else float('-inf'))
print(f"\nОптимальная стратегия (EMV/sigma): {best}")

strategies  = list(results.keys())
emv_vals    = [results[s]['emv']   for s in strategies]
sigma_vals  = [results[s]['sigma'] for s in strategies]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x, w = np.arange(len(strategies)), 0.35
colors_emv = ['mediumseagreen' if v >= 0 else 'tomato' for v in emv_vals]

ax1 = axes[0]
b1 = ax1.bar(x - w/2, emv_vals,   w, color=colors_emv,   alpha=0.85, label='EMV')
b2 = ax1.bar(x + w/2, sigma_vals, w, color='darkorange', alpha=0.75, label='sigma')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=10)
ax1.set_ylabel('Значение, руб.')
ax1.set_title('EMV и риск (sigma) по стратегиям')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
for bar in b1:
    h = bar.get_height()
    ax1.annotate(f'{h:,.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4 if h >= 0 else -12),
                 textcoords='offset points', ha='center', fontsize=8)

ax2 = axes[1]
for i, s in enumerate(strategies):
    ax2.scatter(sigma_vals[i], emv_vals[i], s=200, zorder=5)
    ax2.annotate(s, (sigma_vals[i], emv_vals[i]),
                 textcoords='offset points', xytext=(8, 5), fontsize=10)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_xlabel('Риск sigma, руб.')
ax2.set_ylabel('EMV, руб.')
ax2.set_title('Фронт риск-доходность')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))

plt.suptitle('Дерево решений: выпуск нового функционала', fontsize=13)
plt.tight_layout()
plt.savefig('task3_decision_tree.png', dpi=150)
plt.show()
