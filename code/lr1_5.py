"""1.5. QR-разложение Хаусхолдера и QR-итерации без сдвигов."""
import cmath
import numpy as np
from common import input_lines, matrix_input, norm_inf, run


def qr_decompose(a):
    n = len(a)
    r = a.astype(float).copy()
    q = np.eye(n)
    for k in range(n-1):
        x = r[k:, k].copy()
        length = float(np.sqrt(x @ x))
        if length == 0.0:  # Нулевой столбец не требует отражения.
            continue
        # Знак вектора выбираем так, чтобы не вычитать близкие числа.
        x[0] += length if x[0] >= 0 else -length
        x /= float(np.sqrt(x @ x))
        # H = I - 2vv^T; умножаем без построения полной матрицы H.
        r[k:, k:] -= 2 * np.outer(x, x @ r[k:, k:])
        q[:, k:] -= 2 * np.outer(q[:, k:] @ x, x)
        r[k+1:, k] = 0.0
    return q, r


def block_roots(block):
    trace = block[0, 0] + block[1, 1]
    determinant = block[0, 0]*block[1, 1] - block[0, 1]*block[1, 0]
    root = cmath.sqrt(complex(trace*trace - 4*determinant))
    return np.array([(trace+root)/2, (trace-root)/2])


def extract_blocks(a, eps):
    """Выделяем только изолированные блоки 1x1 и комплексные блоки 2x2.

    Проверяем ВСЕ элементы ниже блока, а не только ниже первой поддиагонали.
    Неизолированная пара не даёт оснований завершать QR-итерации.
    """
    n, i = len(a), 0
    values, sizes = [], []
    while i < n:
        if float(np.sqrt(np.sum(a[i+1:, i]**2))) <= eps:
            values.append(complex(a[i, i]))
            sizes.append(1)
            i += 1
        elif i+1 < n and float(np.sqrt(np.sum(a[i+2:, i:i+2]**2))) <= eps:
            roots = block_roots(a[i:i+2, i:i+2])
            if abs(roots[0].imag) <= np.finfo(float).eps * max(1.0, abs(roots[0])):
                return None  # Вещественный 2x2-блок продолжаем диагонализовать.
            values.extend(roots)
            sizes.append(2)
            i += 2
        else:
            return None
    return np.array(values), tuple(sizes)


def qr_algorithm(a, eps, max_iterations=100000):
    if eps <= 0:
        raise ValueError('Точность должна быть положительной.')
    current = a.astype(float).copy()
    previous = extract_blocks(current, eps)
    if previous is not None and all(size == 1 for size in previous[1]):
        return previous[0], 0, current
    for k in range(1, max_iterations+1):
        q, r = qr_decompose(current)
        current = r @ q
        if not np.all(np.isfinite(current)):
            raise RuntimeError('QR-итерации потеряли конечность чисел.')
        blocks = extract_blocks(current, eps)
        if blocks is not None:
            # Для комплексных пар также требуем стабилизации корней блока.
            if all(size == 1 for size in blocks[1]):
                return blocks[0], k, current
            if previous is not None and blocks[1] == previous[1] and np.max(np.abs(blocks[0] - previous[0])) <= eps:
                return blocks[0], k, current
        previous = blocks
    raise RuntimeError('QR без сдвигов не сошёлся за установленное число итераций. '
                       'Для сложного спектра может потребоваться алгоритм со сдвигами.')


def main():
    a, eps = matrix_input(input_lines(__doc__), with_eps=True)
    q, r = qr_decompose(a)
    print(f'Q:\n{q}\nR:\n{r}')
    print(f'||A - QR||inf: {norm_inf(a - q @ r):.3e}')
    print(f'||Q^T Q - I||inf: {norm_inf(q.T @ q - np.eye(len(a))):.3e}')
    values, k, final = qr_algorithm(a, eps)
    print(f'\nСобственные значения: {values}\nQR-итераций: {k}\nИтоговая матрица:\n{final}')
    print(f'След A: {np.trace(a):.10g}; сумма lambda: {sum(values):.10g}')


if __name__ == '__main__':
    run(main)
