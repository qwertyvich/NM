"""Чтение данных из файла или stdin; общие проверки входа."""
import argparse
import sys
from pathlib import Path
import numpy as np


def input_lines(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file', nargs='?', help='файл данных; без аргумента читать stdin')
    args = parser.parse_args()
    data = Path(args.file).read_text(encoding='utf-8-sig') if args.file else sys.stdin.read()
    return [line.strip() for line in data.splitlines() if line.strip()]


def vector(line):
    result = np.array([float(x) for x in line.split()], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError('Все числа должны быть конечными.')
    return result


def matrix_input(lines, with_rhs=False, with_eps=False):
    header = lines[0].split()
    if len(header) != (2 if with_eps else 1):
        raise ValueError('Первая строка: n eps.' if with_eps else 'Первая строка: n.')
    n = int(header[0])
    if n < 1 or len(lines) != 1 + n + int(with_rhs):
        raise ValueError('Неверное число строк данных или размер матрицы.')
    a = np.array([vector(line) for line in lines[1:1+n]])
    if a.shape != (n, n):
        raise ValueError('Матрица должна быть квадратной размера n.')
    result = [a]
    if with_rhs:
        b = vector(lines[-1])
        if b.shape != (n,):
            raise ValueError('Правая часть должна содержать n чисел.')
        result.append(b)
    if with_eps:
        eps = float(header[1])
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('Точность eps должна быть положительной и конечной.')
        result.append(eps)
    return tuple(result)


def norm_inf(a):
    """Максимум модулей для вектора; максимум сумм модулей строк для матрицы."""
    return float(np.max(np.abs(a))) if a.ndim == 1 else float(np.max(np.sum(np.abs(a), axis=1)))


def run(main):
    np.set_printoptions(precision=10, suppress=True, linewidth=110)
    try:
        main()
    except (ValueError, OSError, IndexError, RuntimeError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        sys.exit(1)
