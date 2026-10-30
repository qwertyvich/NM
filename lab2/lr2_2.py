"""Задача 2.2, вариант 20, a=2. Python 3.9+, стандартная библиотека."""

import argparse
import math
from dataclasses import dataclass


Vector = tuple[float, float]
A = 2.0
LOWER = (1.1, 1.4)
UPPER = (1.2, 1.5)
X0 = (1.15, 1.45)
MAX_ITERATIONS = 10000
# ||J_phi||_inf: первая строка максимальна при x2=1.4,
# вторая - при x1=1.1, поскольку 1.1 <= x1 <= 1.2 < sqrt(2).
Q = max(1 / (math.log(10) * LOWER[1] * math.sqrt(1+2*math.log10(LOWER[1]))),
        A / (2 * LOWER[0]**2) - 1/A)


@dataclass
class Result:
    x: Vector
    iterations: int
    history: list[Vector]
    error_bound: float


def norm(v: Vector) -> float:
    return max(abs(v[0]), abs(v[1]))


def difference(u: Vector, v: Vector) -> Vector:
    return u[0]-v[0], u[1]-v[1]


def in_domain(x: Vector) -> bool:
    return all(LOWER[i] <= x[i] <= UPPER[i] for i in range(2))


def f(x: Vector) -> Vector:
    x1, x2 = x
    return x1*x1 - 2*math.log10(x2) - 1, x1*x1 - A*x1*x2 + A


def jacobian(x: Vector) -> tuple[Vector, Vector]:
    x1, x2 = x
    return (2*x1, -2/(x2*math.log(10))), (2*x1-A*x2, -A*x1)


def phi(x: Vector) -> Vector:
    x1, x2 = x
    return math.sqrt(1+2*math.log10(x2)), (x1*x1+A)/(A*x1)


def validate_epsilon(eps: float) -> None:
    if not math.isfinite(eps) or not 1e-12 <= eps < 1:
        raise ValueError("Точность должна удовлетворять 1e-12 <= eps < 1.")


def check_contraction() -> None:
    # phi1 возрастает по x2; phi2 убывает по x1 в выбранном D.
    image_lower = (phi(LOWER)[0], phi(UPPER)[1])
    image_upper = (phi(UPPER)[0], phi(LOWER)[1])
    if not (0 < Q < 1 and in_domain(image_lower) and in_domain(image_upper)):
        raise ValueError("Не выполнены условия сжатия в выбранной области.")


def simple_iterations(eps: float) -> Result:
    validate_epsilon(eps)
    check_contraction()
    x = X0
    history = [x]
    for k in range(1, MAX_ITERATIONS+1):
        # Обе координаты вычисляются по ПРЕДЫДУЩЕМУ вектору.
        next_x = phi(x)
        error_bound = Q/(1-Q) * norm(difference(next_x, x))
        history.append(next_x)
        x = next_x
        if error_bound <= eps:
            return Result(x, k, history, error_bound)
    raise RuntimeError("Превышено допустимое число итераций.")


def solve_2x2(matrix: tuple[Vector, Vector], b: Vector) -> Vector:
    """Решение линейной системы 2x2 по формулам Крамера."""
    (a11, a12), (a21, a22) = matrix
    determinant = a11*a22-a12*a21
    if abs(determinant) < 1e-15:
        raise ArithmeticError("Якобиан вырожден или близок к вырожденному.")
    return ((b[0]*a22-a12*b[1])/determinant,
            (a11*b[1]-b[0]*a21)/determinant)


def newton_method(eps: float) -> Result:
    validate_epsilon(eps)
    check_contraction()
    x = X0
    history = [x]
    for k in range(1, MAX_ITERATIONS+1):
        fx = f(x)
        delta = solve_2x2(jacobian(x), (-fx[0], -fx[1]))
        next_x = (x[0]+delta[0], x[1]+delta[1])
        if not in_domain(next_x):
            raise ArithmeticError("Итерация Ньютона вышла за выбранную область.")
        # Для любого x из D: ||x-x*|| <= ||x-phi(x)||/(1-Q).
        error_bound = norm(difference(next_x, phi(next_x))) / (1-Q)
        history.append(next_x)
        x = next_x
        if norm(delta) <= eps and error_bound <= eps:
            return Result(x, k, history, error_bound)
    raise RuntimeError("Превышено допустимое число итераций.")


def read_epsilon() -> float:
    parser = argparse.ArgumentParser(description="ЛР 2.2, вариант 20, a=2")
    parser.add_argument("eps", nargs="?", help="Точность, например 1e-6")
    args = parser.parse_args()
    try:
        eps = float((args.eps if args.eps is not None else input("Введите точность: ")).replace(",", "."))
        validate_epsilon(eps)
        return eps
    except ValueError as exc:
        parser.error(str(exc))


def main() -> None:
    eps = read_epsilon()
    for name, method in [("Метод простой итерации", simple_iterations),
                         ("Метод Ньютона", newton_method)]:
        result = method(eps)
        print(f"{name}: x1 = {result.x[0]:.12f}; x2 = {result.x[1]:.12f}; "
              f"итераций = {result.iterations}")


if __name__ == "__main__":
    main()
