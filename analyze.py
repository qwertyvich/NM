"""Создать графики и CSV для анализа сходимости. Нужен matplotlib.

Запуск: python analyze.py 1e-6
Результаты сохраняются в папке results рядом с этим файлом.
"""

import argparse
import csv
import math
from decimal import Decimal, localcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lr2_1 as scalar
import lr2_2 as system


OUT = Path(__file__).resolve().parent / "results"
METHODS = [("Простая итерация", "iterations", "#2563a6"),
           ("Ньютон", "newton", "#bc552c")]


def bisect(function, a, b):
    """Независимый эталон для анализа; не используется основными методами."""
    fa = function(a)
    if fa * function(b) >= 0:
        raise ValueError("Нет смены знака.")
    for _ in range(220):
        midpoint = (a+b)/2
        if midpoint == a or midpoint == b:
            break
        fm = function(midpoint)
        if fm == 0:
            return midpoint
        if fa*fm < 0:
            b = midpoint
        else:
            a, fa = midpoint, fm
    return (a+b)/2


def reference_solutions():
    # Decimal с 60 значащими цифрами позволяет оценивать даже ошибки
    # порядка 1e-16, которые теряются при сравнении двух float.
    with localcontext() as context:
        context.prec = 60

        def tangent(x):
            sin_term, cos_term = x, Decimal(1)
            sin_sum, cos_sum = sin_term, cos_term
            for n in range(1, 100):
                sin_term *= -x*x / ((2*n)*(2*n+1))
                cos_term *= -x*x / ((2*n-1)*(2*n))
                sin_sum += sin_term
                cos_sum += cos_term
                if max(abs(sin_term), abs(cos_term)) < Decimal("1e-65"):
                    break
            return sin_sum/cos_sum

        root1 = bisect(lambda x: tangent(x)-5*x*x+1, Decimal("0.5"), Decimal("0.6"))
        # Из второго уравнения x2=(x1^2+2)/(2*x1).
        def reduced(x1):
            x2 = (x1*x1+2)/(2*x1)
            return x1*x1 - 2*x2.log10() - 1
        x1 = bisect(reduced, Decimal("1.1"), Decimal("1.2"))
        return root1, (x1, (x1*x1+2)/(2*x1))


def actual_error(value, reference):
    if isinstance(value, tuple):
        return max(float(abs(Decimal.from_float(v)-r)) for v, r in zip(value, reference))
    return float(abs(Decimal.from_float(value)-reference))


def grid(a, b, count=500):
    return [a+(b-a)*i/(count-1) for i in range(count)]


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT/filename, dpi=190, facecolor="white")
    plt.close(fig)


def initial_plots(root1, root2):
    root1, root2 = float(root1), tuple(map(float, root2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, limits in zip(axes, [(-1, 1), (0.48, 0.62)]):
        xs = grid(*limits)
        ax.plot(xs, [scalar.f(x) for x in xs], color="#2563a6",
                label=r"$f(x)=\tan x-5x^2+1$")
        ax.axhline(0, color="#555555", lw=.8)
        ax.axvspan(.5, .6, color="#dce9f5", alpha=.7, label="Отрезок [0.5; 0.6]")
        ax.plot(root1, 0, "o", color="#bc552c", label="Положительный корень")
        ax.plot(scalar.X0, scalar.f(scalar.X0), "s", color="#333333", label=r"$x^{(0)}=0.6$")
        ax.set(xlabel="x", ylabel="f(x)")
        ax.grid(alpha=.25)
    axes[0].set_title("Задача 2.1: заданный отрезок")
    axes[1].set_title("Выбор начального приближения")
    axes[0].legend(fontsize=7, loc="lower center")
    finish(fig, "01_scalar_initial.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, limits in zip(axes, [(.65, 1.65), (1.08, 1.22)]):
        xs = grid(*limits)
        ax.plot(xs, [10**((x*x-1)/2) for x in xs], color="#2563a6",
                label=r"$x_2=10^{(x_1^2-1)/2}$")
        ax.plot(xs, [(x*x+2)/(2*x) for x in xs], color="#bc552c",
                label=r"$x_2=(x_1^2+2)/(2x_1)$")
        ax.plot(root2[0], root2[1], "o", color="#222222", label="Решение")
        ax.plot(*system.X0, marker="x", markersize=7, color="#222222", label=r"$x^{(0)}=(1.15;1.45)$")
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
        ax.grid(alpha=.25)
    axes[0].set_title("Задача 2.2: пересечение кривых")
    axes[0].set_ylim(1.1, 2.3)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Область D = [1.1; 1.2] × [1.4; 1.5]")
    axes[1].set_ylim(1.39, 1.51)
    finish(fig, "02_system_initial.png")


def histories(eps, root1, root2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for problem, module, reference, ax in [(1, scalar, root1, axes[0]),
                                            (2, system, root2, axes[1])]:
        for (label, slug, color), method in zip(METHODS, [module.simple_iterations, module.newton_method]):
            result = method(eps)
            rows, errors = [], []
            for k, value in enumerate(result.history):
                if problem == 1:
                    error = actual_error(value, reference)
                    step = abs(value-result.history[k-1]) if k else None
                    bound = scalar.Q/(1-scalar.Q)*step if k and slug == "iterations" else abs(scalar.f(value))/scalar.M
                    row = [k, value, error, step, bound]
                else:
                    error = actual_error(value, reference)
                    step = system.norm(system.difference(value, result.history[k-1])) if k else None
                    bound = system.Q/(1-system.Q)*step if k and slug == "iterations" else system.norm(system.difference(value, system.phi(value)))/(1-system.Q)
                    row = [k, *value, error, step, bound]
                if k == 0 and slug == "iterations":
                    row[-1] = None
                errors.append(error)
                rows.append(row)
            header = ["k", "x", "error", "step", "error_bound"] if problem == 1 else ["k", "x1", "x2", "error_inf", "step_inf", "error_bound"]
            with (OUT/f"task{problem}_{slug}.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)
            # Нулевую ошибку относительно double-эталона нельзя показать в логарифмическом масштабе.
            visible = [(k, error) for k, error in enumerate(errors) if error > 5e-15]
            ax.semilogy([v[0] for v in visible], [v[1] for v in visible], "o-", color=color, label=label)
        ax.axhline(eps, color="#555555", ls="--", lw=.8, label=r"Заданная $\varepsilon$")
        ax.set(title=f"Задача 2.{problem}", xlabel="Номер итерации k",
               ylabel=r"$|x^{(k)}-x^*|$" if problem == 1 else r"$\|x^{(k)}-x^*\|_\infty$")
        ax.set_xticks(range(max(len(module.simple_iterations(eps).history), len(module.newton_method(eps).history))))
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=7)
    finish(fig, "03_errors.png")


def precision_comparison(root1, root2):
    epsilons = [1e-2, 1e-4, 1e-6, 1e-8]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    all_rows = []
    for problem, module, reference, ax in [(1, scalar, root1, axes[0]), (2, system, root2, axes[1])]:
        for (label, slug, color), method in zip(METHODS, [module.simple_iterations, module.newton_method]):
            counts = []
            for eps in epsilons:
                result = method(eps)
                error = actual_error(result.x, reference)
                all_rows.append([f"2.{problem}", label, eps, result.iterations, error, result.error_bound])
                counts.append(result.iterations)
            ax.plot(range(4), counts, "o-", color=color, label=label)
        ax.set(title=f"Задача 2.{problem}", xlabel=r"Требуемая точность $\varepsilon$", ylabel="Количество итераций")
        ax.set_xticks(range(4), ["10⁻²", "10⁻⁴", "10⁻⁶", "10⁻⁸"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    with (OUT/"precision_comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["task", "method", "eps", "iterations", "error", "error_bound"])
        writer.writerows(all_rows)
    finish(fig, "04_iteration_counts.png")


def main():
    parser = argparse.ArgumentParser(description="Графики и анализ ЛР 2, вариант 20")
    parser.add_argument("eps", type=float, nargs="?", default=1e-6)
    eps = parser.parse_args().eps
    scalar.validate_epsilon(eps)
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    root1, root2 = reference_solutions()
    initial_plots(root1, root2)
    histories(eps, root1, root2)
    precision_comparison(root1, root2)
    (OUT/"analysis_precision.txt").write_text(f"eps = {eps:g}\n", encoding="utf-8")
    print(f"Графики и таблицы сохранены в {OUT}")


if __name__ == "__main__":
    main()
