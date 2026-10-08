"""1.1. LU-разложение с выбором главного элемента по столбцу: PA = LU."""
import numpy as np
from common import input_lines, matrix_input, norm_inf, run


def lu_decompose(a):
    n = len(a)
    u = a.astype(float).copy()
    l = np.eye(n)
    p = np.eye(n)
    swaps = 0
    threshold = np.finfo(float).eps * max(1, n) * norm_inf(u)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(u[k:, k])))
        if abs(u[pivot, k]) <= threshold:
            raise ValueError('Матрица вырождена или численно близка к вырожденной.')
        if pivot != k:
            u[[k, pivot], :] = u[[pivot, k], :]
            p[[k, pivot], :] = p[[pivot, k], :]
            # Меняем только уже вычисленные множители L, а не её диагональ.
            l[[k, pivot], :k] = l[[pivot, k], :k]
            swaps += 1
        for i in range(k+1, n):
            l[i, k] = u[i, k] / u[k, k]
            u[i, k:] -= l[i, k] * u[k, k:]
            u[i, k] = 0.0
    return p, l, u, swaps


def lu_solve(p, l, u, b):
    """Прямой ход Ly = Pb, затем обратный Ux = y. Вход b не изменяется."""
    n = len(b)
    rhs = p @ b
    y, x = np.zeros(n), np.zeros(n)
    for i in range(n):
        y[i] = (rhs[i] - l[i, :i] @ y[:i]) / l[i, i]
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - u[i, i+1:] @ x[i+1:]) / u[i, i]
    return x


def inverse_from_lu(p, l, u):
    identity = np.eye(len(l))
    return np.column_stack([lu_solve(p, l, u, identity[:, j]) for j in range(len(l))])


def main():
    a, b = matrix_input(input_lines(__doc__), with_rhs=True)
    p, l, u, swaps = lu_decompose(a)
    x = lu_solve(p, l, u, b)
    inverse = inverse_from_lu(p, l, u)
    determinant = (-1)**swaps * float(np.prod(np.diag(u)))
    for name, value in [('P', p), ('L', l), ('U', u), ('P @ A', p @ a), ('L @ U', l @ u)]:
        print(f'{name}:\n{value}')
    print(f'Перестановок строк: {swaps}\nx: {x}\ndet(A): {determinant:.12g}')
    print(f'A^(-1):\n{inverse}')
    print(f'||PA - LU||inf: {norm_inf(p @ a - l @ u):.3e}')
    print(f'||Ax - b||inf: {norm_inf(a @ x - b):.3e}')
    print(f'||A A^(-1) - I||inf: {norm_inf(a @ inverse - np.eye(len(a))):.3e}')


if __name__ == '__main__':
    run(main)
