"""1.3. Простые итерации (Якоби) и Зейдель с оценками точности из методички."""
import numpy as np
from common import input_lines, matrix_input, norm_inf, run


def equivalent_system(a, b):
    if np.any(np.diag(a) == 0):
        raise ValueError('Для такого приведения диагональные элементы должны быть ненулевыми.')
    alpha = -a.astype(float) / np.diag(a)[:, None]
    np.fill_diagonal(alpha, 0.0)
    beta = b / np.diag(a)
    return alpha, beta


def iterative_solve(a, b, eps, method='jacobi', max_iterations=100000):
    if method not in ('jacobi', 'seidel') or eps <= 0:
        raise ValueError('Неверный метод или точность.')
    alpha, beta = equivalent_system(a, b)
    contraction = norm_inf(alpha)
    if contraction >= 1:
        raise ValueError('||alpha||inf >= 1: выбранная оценка ошибки неприменима. '
                         'Это само по себе ещё не означает расходимость метода.')
    # Для Зейделя C — строго верхняя часть alpha; для Якоби используется alpha.
    numerator = contraction if method == 'jacobi' else norm_inf(np.triu(alpha, 1))
    coefficient = numerator / (1.0 - contraction)
    x = beta.copy()  # Начальное приближение не включается в число итераций.
    history = []
    for k in range(1, max_iterations+1):
        old = x.copy()
        if method == 'jacobi':
            x = beta + alpha @ old
        else:
            for i in range(len(x)):
                x[i] = beta[i] + alpha[i, :i] @ x[:i] + alpha[i, i+1:] @ old[i+1:]
        estimate = coefficient * norm_inf(x - old)
        history.append((k, estimate, norm_inf(a @ x - b)))
        if not np.all(np.isfinite(x)):
            raise RuntimeError('Итерационный процесс потерял конечность чисел.')
        if estimate <= eps:
            return x, k, estimate, history
    raise RuntimeError('Превышен предел итераций.')


def main():
    a, b, eps = matrix_input(input_lines(__doc__), with_rhs=True, with_eps=True)
    alpha, beta = equivalent_system(a, b)
    print(f'alpha:\n{alpha}\nbeta: {beta}\n||alpha||inf: {norm_inf(alpha):.10f}')
    for method, name in [('jacobi', 'Простые итерации'), ('seidel', 'Зейдель')]:
        x, k, estimate, _ = iterative_solve(a, b, eps, method)
        print(f'\n{name}\nx: {x}\nИтераций: {k}\nОценка ошибки: {estimate:.10g}')
        print(f'||Ax - b||inf: {norm_inf(a @ x - b):.3e}')
    print('\nСравнение точностей: eps | Якоби | Зейдель')
    for accuracy in [1e-2, 1e-4, 1e-6]:
        counts = [iterative_solve(a, b, accuracy, method)[1] for method in ['jacobi', 'seidel']]
        print(f'{accuracy:.0e} | {counts[0]} | {counts[1]}')


if __name__ == '__main__':
    run(main)
