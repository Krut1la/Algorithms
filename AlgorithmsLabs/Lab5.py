"""
Prog:   Lab5.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 5. Var 10. 2021
"""
import math

from AlgorithmsLabs.Input import get_input_source, InputError, get_option

import numpy as np


def gauss_seidel_method(system, eq, epsilon):
    """
    Solves system of linear equations
    :param system: x[] coefficients, left parts
    :param eq: right parts
    :param epsilon: precision
    :return: x vector of results
    """
    n = len(system)
    x = np.zeros(n)

    max_iterations = 100
    iteration = 1

    full = False
    while not full or iteration > max_iterations:
        x_new = np.copy(x)
        for i in range(0, n):
            s1 = 0.0
            for j in range(0, i):
                s1 = s1 + system[i][j] * x_new[j]

            s2 = 0.0
            for j in range(i + 1, n):
                s1 = s1 + system[i][j] * x[j]

            x_new[i] = (eq[i] - s1 - s2) / system[i][i]

        ss = 0.0
        for i in range(0, n):
            ss = ss + (x_new[i] - x[i]) ** 2

        full = np.sqrt(ss) <= epsilon
        x = x_new

        iteration = iteration + 1

    if not full:
        print("Max iteration reached. Precision still not reached!")

    return x


def is_converging(a):
    n = len(a)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            s = s + math.fabs(a[i][j])

        if math.fabs(a[i][i]) < s:
            return False

    return True


def main():
    print("Lab 5. Solving of linear equations systems. Var. 10")

    input_source = get_input_source("lab5_system_linear_equations")

    try:
        n = input_source.read_var("n", int, 2, 10)
        epsilon = m = input_source.read_var("epsilon", float, 1e-6, 1e-1)

        a = []
        for i in range(0, n):
            line = []
            for j in range(0, n):
                line.append(input_source.read_var("a[{},{}]".format(i, j), float, float("-inf"), float("inf")))

            a.append(line)

        b = []
        for i in range(0, n):
            b.append(input_source.read_var("b[{}]".format(i), float, float("-inf"), float("inf")))

    except InputError as ir:
        print("Error wile reading data from file: ", ir)
        return

    if not is_converging(a):
        print("Convergence condition is false. Method might fail.")
        option = get_option("(1 - continue, 2 - exit", 2)
        if option == 2:
            return

    try:
        x = gauss_seidel_method(a, b, epsilon)
        print(x)
    except ValueError:
        print("Calculation failed.")


main()