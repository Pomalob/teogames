# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

ALTERNATIVES = ['PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'MySQL', 'DynamoDB']
CRITERIA     = ['Производительность', 'Стоимость', 'Масштабируемость',
                'Простота поддержки', 'Надёжность']


def parse_value(s):
    s = s.strip()
    if '/' in s:
        a, b = s.split('/', 1)
        return float(a) / float(b)
    return float(s)


def priority_vector(matrix):
    vals, vecs = np.linalg.eig(matrix)
    idx = int(np.argmax(vals.real))
    w = vecs[:, idx].real
    return w / w.sum(), float(vals[idx].real)


def consistency(n, lmax):
    ci = (lmax - n) / (n - 1) if n > 1 else 0.0
    ri = RI.get(n, 1.49)
    return ci, ci / ri if ri > 0 else 0.0


def input_matrix(title, labels):
    n = len(labels)
    print(f"\n{'='*55}")
    print(f"Матрица: {title}")
    print("Шкала Саати: 1-9, дроби — как 1/3")
    print('='*55)

    while True:
        matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                while True:
                    try:
                        v = parse_value(input(f"  {labels[i][:16]} vs {labels[j][:16]}: "))
                        if v <= 0:
                            print("  > 0!")
                            continue
                        matrix[i, j], matrix[j, i] = v, 1.0 / v
                        break
                    except (ValueError, ZeroDivisionError):
                        print("  Некорректный ввод.")

        w, lmax = priority_vector(matrix)
        ci, cr = consistency(n, lmax)

        print(f"\n  Вектор приоритетов: {np.round(w, 4)}")
        print(f"  lambda_max={lmax:.4f}  CI={ci:.4f}  CR={cr:.4f}", end="  ")
        if cr <= 0.1:
            print("OK")
            return matrix, w, ci, cr
        print("CR > 0.1 — введите матрицу заново.")


def run_ahp():
    print("AHP — Выбор СУБД для микросервиса")
    print(f"Альтернативы: {', '.join(ALTERNATIVES)}")
    print(f"Критерии:     {', '.join(CRITERIA)}")

    _, w_crit, ci_c, cr_c = input_matrix("Критерии", CRITERIA)

    print(f"\nВеса критериев:")
    for crit, w in zip(CRITERIA, w_crit):
        print(f"  {crit}: {w:.4f}")

    local_w = np.zeros((len(ALTERNATIVES), len(CRITERIA)))
    for k, crit in enumerate(CRITERIA):
        _, w_alt, _, _ = input_matrix(f"Альтернативы / {crit}", ALTERNATIVES)
        local_w[:, k] = w_alt

    global_p = local_w @ w_crit
    ranking  = sorted(zip(ALTERNATIVES, global_p), key=lambda x: x[1], reverse=True)

    print("\n=== ИТОГОВЫЙ РЕЙТИНГ ===")
    print(f"{'Место':<6} {'Альтернатива':<14} {'Приоритет':>12}")
    print("-" * 35)
    for place, (alt, p) in enumerate(ranking, 1):
        mark = "  <<< ВЫБОР" if place == 1 else ""
        print(f"{place:<6} {alt:<14} {p:>12.4f}{mark}")

    col_w = 13
    print("\nЛокальные веса:")
    header = f"{'':14}" + "".join(f"{c[:12]:>{col_w}}" for c in CRITERIA) + f"{'Глобальный':>{col_w}}"
    print(header)
    for i, alt in enumerate(ALTERNATIVES):
        row = f"{alt:<14}" + "".join(f"{local_w[i,k]:{col_w}.4f}" for k in range(len(CRITERIA)))
        row += f"{global_p[i]:{col_w}.4f}"
        print(row)


if __name__ == '__main__':
    run_ahp()
