"""Независимые проверки. Только здесь разрешены эталонные numpy.linalg.

Запуск из корня проекта: python checks/verify.py
Никакие эталонные решатели не используются внутри сдаваемых алгоритмов.
"""
import itertools
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from common import matrix_input, vector, norm_inf
from lr1_1 import lu_decompose, lu_solve, inverse_from_lu
from lr1_2 import tridiagonal_solve, tridiagonal_product
from lr1_3 import iterative_solve
from lr1_4 import jacobi_rotations
from lr1_5 import qr_decompose, qr_algorithm


def read(number, **kwargs):
    lines = (ROOT/'tests'/f'lr1_{number}.txt').read_text().splitlines()
    return matrix_input(lines, **kwargs)


def eigen_error(values, reference):
    # Для небольших проверочных матриц сравниваем спектры независимо от порядка.
    return min(max(abs(values[i] - reference[p[i]]) for i in range(len(values)))
               for p in itertools.permutations(range(len(values))))


def require_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Ожидалось отклонение некорректных данных.')


def main():
    rng = np.random.default_rng(20)
    checks = 0
    data = {}
    a, b = read(1, with_rhs=True)
    # Вариант 20, пример одногруппника (вариант 16) и матрицы с перестановками.
    peer = np.array([[-5,-1,-3,-1],[-2,0,8,-4],[-7,-2,2,-2],[2,-4,-4,4]], float)
    matrices = [a, peer, np.array([[0, 1], [2, 3]], float)]
    matrices += [rng.normal(size=(n, n)) for n in (1, 3, 4, 7) for _ in range(4)]
    for mat in matrices:
        rhs = rng.normal(size=len(mat))
        original = rhs.copy()
        p, l, u, swaps = lu_decompose(mat)
        np.testing.assert_allclose(p @ mat, l @ u, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(lu_solve(p, l, u, rhs), np.linalg.solve(mat, rhs), atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(rhs, original)
        np.testing.assert_allclose(inverse_from_lu(p, l, u), np.linalg.inv(mat), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose((-1)**swaps * np.prod(np.diag(u)), np.linalg.det(mat), atol=1e-10, rtol=1e-10)
        checks += 1
    require_error(lambda: lu_decompose(np.array([[1., 2.], [2., 4.]])))
    checks += 1
    data['task1'] = {'x': np.linalg.solve(a, b).tolist(), 'det': float(np.linalg.det(a))}

    diagonals = [vector(line) for line in (ROOT/'tests/lr1_2.txt').read_text().splitlines()]
    low, diagonal, high, rhs = diagonals
    full = np.diag(diagonal) + np.diag(low[1:], -1) + np.diag(high[:-1], 1)
    x, _, _ = tridiagonal_solve(*diagonals)
    np.testing.assert_allclose(x, np.linalg.solve(full, rhs), atol=1e-12)
    np.testing.assert_allclose(tridiagonal_product(low, diagonal, high, x), rhs, atol=1e-12)
    checks += 1
    data['task2'] = {'x': x.tolist()}
    for n in (1, 2, 8):
        low, high = rng.normal(size=n), rng.normal(size=n)
        low[0], high[-1] = 0, 0
        diagonal = np.abs(low) + np.abs(high) + 1
        rhs = rng.normal(size=n)
        full = np.diag(diagonal) + np.diag(low[1:], -1) + np.diag(high[:-1], 1)
        x, _, _ = tridiagonal_solve(low, diagonal, high, rhs)
        np.testing.assert_allclose(x, np.linalg.solve(full, rhs), atol=1e-12)
        checks += 1
    require_error(lambda: tridiagonal_solve(np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2)))
    checks += 1

    a, b, _ = read(3, with_rhs=True, with_eps=True)
    reference = np.linalg.solve(a, b)
    data['task3'] = {'reference': reference.tolist(), 'accuracy': []}
    for eps in (1e-2, 1e-4, 1e-6):
        row = {'eps': eps}
        for method in ('jacobi', 'seidel'):
            x, k, estimate, history = iterative_solve(a, b, eps, method)
            error = norm_inf(x-reference)
            assert error <= estimate + 1e-12 and estimate <= eps
            assert k == len(history)
            row[method] = {'x': x.tolist(), 'iterations': k, 'estimate': estimate, 'error': error}
            checks += 1
        data['task3']['accuracy'].append(row)
    require_error(lambda: iterative_solve(np.array([[1.,2.],[2.,1.]]), np.ones(2), 1e-2))
    checks += 1

    a, _ = read(4, with_eps=True)
    reference = np.linalg.eigvalsh(a)
    data['task4'] = {'reference': reference.tolist(), 'accuracy': []}
    for eps in (1e-2, 1e-4, 1e-6):
        values, vectors, k, history = jacobi_rotations(a, eps)
        error = float(np.max(np.abs(values-reference)))
        assert error <= np.sqrt(2)*history[-1][1] + 1e-12
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(a)), atol=1e-12)
        assert all(history[i+1][1] <= history[i][1] + 1e-12 for i in range(len(history)-1))
        row = {'eps': eps, 'iterations': k, 'off_norm': history[-1][1], 'eigen_error': error}
        data['task4']['accuracy'].append(row)
        checks += 1
    for mat in [np.eye(3), np.array([[2.,-3.],[-3.,2.]]), np.zeros((1,1))]:
        vals, vecs, _, _ = jacobi_rotations(mat, 1e-10)
        np.testing.assert_allclose(vals, np.linalg.eigvalsh(mat), atol=1e-10)
        np.testing.assert_allclose(mat @ vecs, vecs * vals, atol=1e-10)
        checks += 1
    require_error(lambda: jacobi_rotations(np.array([[1.,2.],[0.,1.]]), 1e-2))
    checks += 1

    a, _ = read(5, with_eps=True)
    data['task5'] = {'reference': [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(a)], 'accuracy': []}
    # Пара комплексных корней, нулевой и треугольный столбцы и дополнительный 5x5 тест из примера.
    peer3 = np.array([[1,2,5],[-8,0,-6],[7,-9,-7]], float)
    peer5 = np.array([[2,1,3,4,5],[1,0,2,1,3],[4,1,1,0,2],[3,2,0,1,1],[5,4,3,2,1]], float)
    mats = [a, peer3, peer5, np.array([[0.,-1.],[1.,0.]]),
            np.diag([0.,2.,-3.]), np.zeros((3,3)), np.array([[2.]])]
    for mat in mats:
        q, r = qr_decompose(mat)
        np.testing.assert_allclose(q @ r, mat, atol=1e-11)
        np.testing.assert_allclose(q.T @ q, np.eye(len(mat)), atol=1e-11)
        np.testing.assert_allclose(np.tril(r, -1), 0, atol=1e-11)
        vals, _, _ = qr_algorithm(mat, 1e-9)
        assert eigen_error(vals, np.linalg.eigvals(mat)) < 1e-7
        checks += 1
    for eps in (1e-2, 1e-4, 1e-6):
        vals, k, _ = qr_algorithm(a, eps)
        error = eigen_error(vals, np.linalg.eigvals(a))
        assert error < eps
        data['task5']['accuracy'].append({'eps': eps, 'iterations': k, 'eigen_error': error})
        checks += 1

    data['checked_cases'] = checks
    (ROOT/'results/verification.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    print(f'OK: {checks} проверочных случаев. Эталонные numpy.linalg используются только в checks/verify.py.')
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
