"""1.2. Метод прогонки. Вход: нижняя, главная, верхняя диагонали и правая часть."""
import numpy as np
from common import input_lines, vector, norm_inf, run


def tridiagonal_solve(a, b, c, d):
    n = len(d)
    if n == 0 or any(len(v) != n for v in (a, b, c)):
        raise ValueError('Все четыре вектора должны иметь одинаковую ненулевую длину.')
    if a[0] != 0 or c[-1] != 0:
        raise ValueError('Для границ системы нужны a[0] = 0 и c[-1] = 0.')
    p, q, x = np.zeros(n), np.zeros(n), np.zeros(n)
    threshold = np.finfo(float).eps * n * max(np.max(np.abs(v)) for v in (a, b, c))
    for i in range(n):
        denominator = b[i] + (a[i] * p[i-1] if i else 0.0)
        if abs(denominator) <= threshold:
            raise ValueError(f'Нулевой знаменатель прогонки в строке {i+1}.')
        p[i] = -c[i] / denominator
        q[i] = (d[i] - (a[i] * q[i-1] if i else 0.0)) / denominator
    x[-1] = q[-1]
    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]
    return x, p, q


def tridiagonal_product(a, b, c, x):
    result = b * x
    result[1:] += a[1:] * x[:-1]
    result[:-1] += c[:-1] * x[1:]
    return result


def main():
    lines = input_lines(__doc__)
    if len(lines) != 4:
        raise ValueError('Нужны ровно четыре строки: a, b, c, d.')
    a, b, c, d = [vector(line) for line in lines]
    x, p, q = tridiagonal_solve(a, b, c, d)
    print(f'p: {p}\nq: {q}\nx: {x}')
    product = tridiagonal_product(a, b, c, x)
    print(f'Ax: {product}\nd:  {d}')
    print(f'||Ax - d||inf: {norm_inf(product - d):.3e}')


if __name__ == '__main__':
    run(main)
