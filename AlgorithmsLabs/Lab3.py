"""
Prog:   Lab3.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 3. Var 10. 2021
"""

import matplotlib.pyplot as plt
import math


def poly_newton_coefficient(x_nodes, y_nodes):
    """
    Cals coefficients of Newton polynomial using the general formula (1.6)
    :param x_nodes: x values in nodes
    :param y_nodes: y values in nodes
    :return: array of coefficients
    """
    m = len(x_nodes)

    coefficients = []

    for n in range(1, m + 1):
        S = 0
        for j in range(1, n + 1):

            P = 1
            for i in range(1, n + 1):
                if i == j:
                    continue

                P = P * (x_nodes[j - 1] - x_nodes[i - 1])

            S = S + y_nodes[j - 1] / P

        coefficients.append(S)

    return coefficients


def newton_polynomial(x_nodes, x, coefficients):
    """
    Evaluates Newton polynomial in x point
    :param x_nodes: x values in nodes
    :param x: point to evaluate
    :param coefficients: Newton polynomial coefficients
    :return: y value
    """
    n = len(x_nodes) - 1
    p = coefficients[n]
    for k in range(1, n + 1):
        p = coefficients[n - k] + (x - x_nodes[n - k]) * p
    return p


def generate_interpolated_nodes(a, b, x_nodes, y_nodes, n):
    """
    Calculates n equidistant interpolated points between a and b
    :param a: start of range
    :param b: end of range
    :param x_nodes: x values in nodes
    :param y_nodes: y values in nodes
    :param n: number of points to calculate
    :return: arrays of n x and n y values
    """
    ax = poly_newton_coefficient(x_nodes, y_nodes)

    x = []
    y = []
    step = (b - a) / n

    for i in range(0, n):
        xi = a + i * step
        x.append(xi)
        y.append(newton_polynomial(x_nodes, xi, ax))

    return x, y


def generate_nodes(a, b, n):
    """
    Calculates n equidistant points for given function between a and b.
    Var 10. f(x) = cos(x + e^cos(x)
    :param a: start of range
    :param b: and of range
    :param n: number of nodes
    :return: arrays of n x and n y values in nodes
    """
    x = []
    y = []
    step = (b - a) / n

    for i in range(0, n):
        xi = a + i * step
        x.append(xi)
        y.append(math.cos(xi + math.exp(math.cos(xi))))

    return x, y


def main():
    print("Lab 3. Newton polynomial. Var. 10")
    fig, axs = plt.subplots()
    fig.canvas.set_window_title('Lab 3. Newton polynomial. Var. 10')
    axs.set_title('cos(x + e^cos(x))')
    axs.set_ylabel('f(x)')
    axs.set_xlabel('x')

    a = 3.0
    b = 6.0

    x_nodes, y_nodes = generate_nodes(a, b, 30)

    axs.plot(x_nodes, y_nodes, 'ro', color='green')

    x, y = generate_interpolated_nodes(a, b, x_nodes, y_nodes, 80)

    axs.plot(x, y, color='red')

    plt.show()


main()
