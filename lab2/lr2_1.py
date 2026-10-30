"""Задача 2.1, вариант 20. Python 3.9+, только стандартная библиотека."""

import argparse
import math
from dataclasses import dataclass


A, B = 0.5, 0.6
X0 = B
MAX_ITERATIONS = 10000
# Верхняя оценка |phi'(x)| на [A, B].
Q = 1 / (2 * math.sqrt(5) * math.cos(B)**2 * math.sqrt(math.tan(A) + 1))
# |f'(x)| = 10*x - sec(x)^2 >= 10*A - sec(B)^2 > 0.
M = 10 * A - 1 / math.cos(B)**2


@dataclass
class Result:
    x: float
    iterations: int
    history: list[float]
    error_bound: float


def f(x: float) -> float:
    return math.tan(x) - 5 * x*x + 1


def derivative(x: float) -> float:
    return 1 / math.cos(x)**2 - 10*x


def second_derivative(x: float) -> float:
    return 2 * math.tan(x) / math.cos(x)**2 - 10


def phi(x: float) -> float:
    return math.sqrt((math.tan(x) + 1) / 5)


def validate_epsilon(eps: float) -> None:
    if not math.isfinite(eps) or not 1e-12 <= eps < 1:
        raise ValueError("Точность должна удовлетворять 1e-12 <= eps < 1.")


def simple_iterations(eps: float) -> Result:
    validate_epsilon(eps)
    if not (0 < Q < 1 and A <= phi(A) <= phi(B) <= B):
        raise ValueError("Не выполнены условия сходимости на [A, B].")
    x = X0
    history = [x]
    for k in range(1, MAX_ITERATIONS + 1):
        next_x = phi(x)
        error_bound = Q / (1-Q) * abs(next_x-x)
        history.append(next_x)
        x = next_x
        if error_bound <= eps:
            return Result(x, k, history, error_bound)
    raise RuntimeError("Превышено допустимое число итераций.")


def newton_method(eps: float) -> Result:
    validate_epsilon(eps)
    if f(X0) * second_derivative(X0) <= 0:
        raise ValueError("Начальная точка не удовлетворяет f(x0)*f''(x0)>0.")
    x = X0
    history = [x]
    for k in range(1, MAX_ITERATIONS + 1):
        d = derivative(x)
        if abs(d) < 1e-15:
            raise ArithmeticError("Производная близка к нулю.")
        next_x = x - f(x)/d
        if not A <= next_x <= B:
            raise ArithmeticError("Итерация Ньютона вышла за выбранный отрезок.")
        step = abs(next_x-x)
        # По теореме Лагранжа |x-x*| <= |f(x)|/M.
        error_bound = abs(f(next_x)) / M
        history.append(next_x)
        x = next_x
        if step <= eps and error_bound <= eps:
            return Result(x, k, history, error_bound)
    raise RuntimeError("Превышено допустимое число итераций.")


def read_epsilon() -> float:
    parser = argparse.ArgumentParser(description="ЛР 2.1, вариант 20")
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
        print(f"{name}: x = {result.x:.12f}; итераций = {result.iterations}")


if __name__ == "__main__":
    main()
