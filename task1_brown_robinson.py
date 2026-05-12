import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

np.random.seed(42)

N_ROWS, N_COLS = 12, 12
A = np.random.randint(-100, 101, size=(N_ROWS, N_COLS))

print("Платёжная матрица A (12x12):")
print(A)


def brown_robinson(A, num_iterations=5000):
    n, m = A.shape
    row_counts = np.zeros(n)
    col_counts = np.zeros(m)

    row_counts[np.random.randint(0, n)] += 1
    col_counts[np.random.randint(0, m)] += 1

    lower_prices, upper_prices = [], []

    for _ in range(num_iterations):
        p = row_counts / row_counts.sum()
        q = col_counts / col_counts.sum()

        row_payoffs = A @ q
        col_payoffs = p @ A

        lower_prices.append(float(np.min(col_payoffs)))
        upper_prices.append(float(np.max(row_payoffs)))

        row_counts[int(np.argmax(row_payoffs))] += 1
        col_counts[int(np.argmin(col_payoffs))] += 1

    p_final = row_counts / row_counts.sum()
    q_final = col_counts / col_counts.sum()
    game_value = (lower_prices[-1] + upper_prices[-1]) / 2.0

    return p_final, q_final, game_value, lower_prices, upper_prices


def solve_game_lp(A):
    n, m = A.shape

    c1 = np.zeros(n + 1)
    c1[-1] = -1.0

    A_ub1 = np.zeros((m, n + 1))
    for j in range(m):
        A_ub1[j, :n] = -A[:, j]
        A_ub1[j, n] = 1.0

    A_eq1 = np.zeros((1, n + 1))
    A_eq1[0, :n] = 1.0

    res1 = linprog(c1, A_ub=A_ub1, b_ub=np.zeros(m),
                   A_eq=A_eq1, b_eq=np.array([1.0]),
                   bounds=[(0, None)] * n + [(None, None)], method='highs')
    if not res1.success:
        raise RuntimeError("LP (игрок 1): " + res1.message)
    p_lp = res1.x[:n]
    v_lp = res1.x[n]

    c2 = np.zeros(m + 1)
    c2[-1] = 1.0

    A_ub2 = np.zeros((n, m + 1))
    for i in range(n):
        A_ub2[i, :m] = A[i, :]
        A_ub2[i, m] = -1.0

    A_eq2 = np.zeros((1, m + 1))
    A_eq2[0, :m] = 1.0

    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(n),
                   A_eq=A_eq2, b_eq=np.array([1.0]),
                   bounds=[(0, None)] * m + [(None, None)], method='highs')
    if not res2.success:
        raise RuntimeError("LP (игрок 2): " + res2.message)
    q_lp = res2.x[:m]

    return p_lp, q_lp, v_lp


NUM_ITER = 5000
p_br, q_br, v_br, lower_prices, upper_prices = brown_robinson(A, NUM_ITER)
p_lp, q_lp, v_lp = solve_game_lp(A)

print(f"\n=== Метод Брауна-Робинсона ({NUM_ITER} итераций) ===")
print(f"Нижняя цена: {lower_prices[-1]:.4f}")
print(f"Верхняя цена: {upper_prices[-1]:.4f}")
print(f"Цена игры (BR): {v_br:.4f}")
print(f"Стратегия игрока 1: {np.round(p_br, 4)}")
print(f"Стратегия игрока 2: {np.round(q_br, 4)}")

print(f"\n=== Линейное программирование ===")
print(f"Цена игры (LP): {v_lp:.4f}")
print(f"Стратегия игрока 1: {np.round(p_lp, 4)}")
print(f"Стратегия игрока 2: {np.round(q_lp, 4)}")
print(f"\nРазница |BR - LP|: {abs(v_br - v_lp):.4f}")

iterations = np.arange(1, NUM_ITER + 1)
plt.figure(figsize=(12, 6))
plt.plot(iterations, lower_prices, label='Нижняя цена игры', color='blue', alpha=0.7)
plt.plot(iterations, upper_prices, label='Верхняя цена игры', color='red', alpha=0.7)
plt.axhline(y=v_lp, color='green', linestyle='--', linewidth=2,
            label=f'Цена игры (LP) = {v_lp:.2f}')
plt.xlabel('Номер итерации')
plt.ylabel('Цена игры')
plt.title('Сходимость метода Брауна-Робинсона')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1_convergence.png', dpi=150)
plt.show()
