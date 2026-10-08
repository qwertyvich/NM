"""1.4. Вращения Якоби для собственных значений и векторов симметричной матрицы."""
import math
import numpy as np
from common import input_lines, matrix_input, norm_inf, run


def off_diagonal_norm(a):
    return float(np.sqrt(np.sum(np.triu(a, 1)**2)))


def jacobi_rotations(a, eps, max_iterations=100000):
    if eps <= 0 or not np.allclose(a, a.T, rtol=0, atol=1e-12):
        raise ValueError('Нужны симметричная матрица и положительная точность.')
    current = a.astype(float).copy()
    vectors = np.eye(len(a))
    history = [(0, off_diagonal_norm(current))]
    k = 0
    while history[-1][1] > eps:
        if k >= max_iterations:
            raise RuntimeError('Превышен предел вращений.')
        upper = np.abs(np.triu(current, 1))
        i, j = np.unravel_index(np.argmax(upper), upper.shape)
        # atan2 корректно обрабатывает равные диагональные элементы и знаки.
        phi = 0.5 * math.atan2(2 * current[i, j], current[i, i] - current[j, j])
        c, s = math.cos(phi), math.sin(phi)
        rotation = np.eye(len(a))
        rotation[i, i] = rotation[j, j] = c
        rotation[i, j], rotation[j, i] = -s, s
        current = rotation.T @ current @ rotation
        # Удаляем только асимметрию, возникающую из-за округления.
        current = (current + current.T) / 2
        vectors = vectors @ rotation
        k += 1
        history.append((k, off_diagonal_norm(current)))
    order = np.argsort(np.diag(current))
    return np.diag(current)[order], vectors[:, order], k, history


def main():
    a, eps = matrix_input(input_lines(__doc__), with_eps=True)
    values, vectors, k, history = jacobi_rotations(a, eps)
    print(f'Собственные значения: {values}\nСобственные векторы по столбцам:\n{vectors}')
    print(f'Вращений: {k}\nВнедиагональная норма: {history[-1][1]:.3e}')
    for j, value in enumerate(values):
        print(f'||Av{j+1} - lambda{j+1} v{j+1}||inf: {norm_inf(a @ vectors[:, j] - value*vectors[:, j]):.3e}')
    print(f'||V^T V - I||inf: {norm_inf(vectors.T @ vectors - np.eye(len(a))):.3e}')
    print('\nИстория: вращение | внедиагональная норма')
    for step, off in history:
        print(f'{step} | {off:.10g}')
    print('\nТочность | вращений | итоговая внедиагональная норма | макс. невязка Av-lambda*v')
    for accuracy in [1e-2, 1e-4, 1e-6]:
        vals, vecs, steps, hist = jacobi_rotations(a, accuracy)
        residual = max(norm_inf(a @ vecs[:, j] - vals[j]*vecs[:, j]) for j in range(len(a)))
        print(f'{accuracy:.0e} | {steps} | {hist[-1][1]:.3e} | {residual:.3e}')


if __name__ == '__main__':
    run(main)
