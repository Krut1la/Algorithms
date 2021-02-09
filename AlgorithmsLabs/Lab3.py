"""
Prog:   Lab3.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 3. Var 10. 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce
import operator

from pip._vendor.urllib3.connectionpool import xrange


def interpolate(x, x_values, y_values):
    def _basis(j):
        p = [(x - x_values[m])/(x_values[j] - x_values[m]) for m in xrange(k) if m != j]
        return reduce(operator.mul, p)
    assert len(x_values) != 0 and (len(x_values) == len(y_values)), 'x and y cannot be empty and must have the same length'
    k = len(x_values)
    return sum(_basis(j)*y_values[j] for j in xrange(k))


def poly_newton_coefficient(x, y):
    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (x[k:m] - x[k - 1])

    return a


def __poly_newton_coefficient(x_nodes, y_nodes):
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


def generate_interpolated_y_range(x_nodes, x_range, a):
    y = []
    for x in x_range:
        y.append(newton_polynomial(x_nodes, x, a))

    return y


def generate_interpolated_y_range_lagrange(x_nodes, y_nodes, x_range):
    y = []
    for x in x_range:
        y.append(interpolate(x, x_nodes, y_nodes))

    return y




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


def generate_x_range(a, b, m):
    x = []
    for i in range(0, m + 1):
        x.append(a + i * (b - a) / m)

    return x


def generate_y_range(x_range, func):
    y = []
    for x in x_range:
        y.append(func(x))

    return y


def create_axs(func_expression):
    fig, axs = plt.subplots(2, 4)
    fig.canvas.set_window_title('Lab 3. Newton polynomial. Var. 10')
    axs[0, 0].set_title('Precise')
    axs[0, 0].set_ylabel(func_expression)
    axs[0, 0].set_xlabel('x')
    axs[0, 0].grid()

    axs[0, 1].set_title('Interpolated')
    axs[0, 1].set_ylabel('Pn(x)')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].grid()

    axs[0, 2].set_title('Error')
    axs[0, 2].set_ylabel('-lg|Pn(x) - Pn+1(x)|')
    axs[0, 2].set_xlabel('x_m')
    axs[0, 2].grid()

    axs[0, 3].set_title('Error precise')
    axs[0, 3].set_ylabel('-lg|Pn(x) - f(x)|')
    axs[0, 3].set_xlabel('x_m')
    axs[0, 3].grid()

    axs[1, 0].set_title('Error evaluation')
    axs[1, 0].set_ylabel('-lg|Pn(x) - Pn+1(x)|')
    axs[1, 0].set_xlabel('x_m')
    axs[1, 0].grid()

    axs[1, 1].set_title('Error evaluation precise')
    axs[1, 1].set_ylabel('-lg|Pn(x) - f(x)|')
    axs[1, 1].set_xlabel('x_m')
    axs[1, 1].grid()

    axs[1, 2].set_title('sin(x)')
    axs[1, 2].set_ylabel('f(x)')
    axs[1, 2].set_xlabel('x')

    axs[1, 3].set_title('sin(x)')
    axs[1, 3].set_ylabel('f(x)')
    axs[1, 3].set_xlabel('x')

    return axs


def get_delta_n(x, x_nodes, func):
    n = len(x_nodes)
    delta_n = 0.0

    for j in range(0, n):
        sigma_j = func(x_nodes[j])
        A_j = 1.0

        for i in range(0, n):
            if i == j:
                continue

            A_j = A_j * (x - x_nodes[i]) / (x_nodes[j] - x_nodes[i])

        delta_n = delta_n + sigma_j * A_j

    return delta_n


def analyze_func(a, b, m, func, axs):
    # draw original func
    x_range = generate_x_range(a, b, 1000)
    y_range = generate_y_range(x_range, func)

    axs[0, 0].plot(x_range, y_range, color='green')

    # draw polynomial
    x_nodes = generate_x_range(a, b, m)
    y_nodes = generate_y_range(x_nodes, func)

    axs[0, 1].plot(x_nodes, y_nodes, 'ro', color='blue')

    coefficients = poly_newton_coefficient(x_nodes, y_nodes)

    # y_interpolated = generate_interpolated_y_range(x_nodes, x_range, coefficients)
    y_interpolated = generate_interpolated_y_range_lagrange(x_nodes, y_nodes, x_range)

    axs[0, 1].plot(x_range, y_interpolated, color='red')

    # draw error
    j = 2

    for n in range(1, m):
        x_nodes_test = x_nodes[:n + 1]
        y_nodes_test = y_nodes[:n + 1]
        coefficients_test = poly_newton_coefficient(x_nodes_test, y_nodes_test)

        x_nodes_plus_1 = x_nodes[:n + 1 + 1]
        y_nodes_plus_1 = y_nodes[:n + 1 + 1]
        coefficients_plus_1 = poly_newton_coefficient(x_nodes_plus_1, y_nodes_plus_1)

        x_range_test = generate_x_range(x_nodes[j], x_nodes[j + 1], 100)

        x_error = []
        y_error_polynomial = []
        y_error_func = []
        y_error_func_float = []
        for x in x_range_test:
            try:
                xp = (x - x_nodes[j]) / (x_nodes[j + 1] - x_nodes[j])

                # Px = newton_polynomial(x_nodes_test, x, coefficients_test)
                # P_1x = newton_polynomial(x_nodes_plus_1, x, coefficients_plus_1)
                Px = interpolate(x, x_nodes_test, y_nodes_test)
                P_1x = interpolate(x, x_nodes_plus_1, y_nodes_plus_1)
                Fx = func(x)
                delta_n = get_delta_n(x, x_nodes_test, func)
                delta_n_1 = get_delta_n(x, x_nodes_plus_1, func)

                log_delta_P_P1 = -math.log(math.fabs(Px - P_1x), 10)
                log_delta_P_func = -math.log(math.fabs(Px - Fx), 10)
                log_func_float = -math.log(math.fabs(delta_n - delta_n_1), 10)
            except ValueError:
                pass
            else:
                x_error.append(xp)
                y_error_polynomial.append(log_delta_P_P1)
                y_error_func.append(log_delta_P_func)
                y_error_func_float.append(log_func_float)

        axs[0, 2].plot(x_error, y_error_polynomial, color='black')
        axs[0, 2].annotate(j, xy=(0, 0), xycoords='data',
                           xytext=(0.0, 0.0), textcoords='axes fraction',
                           arrowprops=dict(facecolor='black', shrink=0.05),
                           horizontalalignment='right', verticalalignment='top',
                           )
        axs[0, 3].plot(x_error, y_error_func, color='black')
        axs[1, 0].plot(x_error, y_error_func_float, color='black')

    # draw error evaluation

    # draw table
    axs[1, 2].axis('tight')
    axs[1, 2].axis('off')
    clust_data = np.random.random((10, 4))
    col_label = ("n", "Dn", "Dn Exact", "k")
    the_table = axs[1, 2].table(cellText=clust_data, colLabels=col_label, loc='center')


def main():
    print("Lab 3. Newton polynomial. Var. 10")

    axs = create_axs("cos(x + e^cos(x))")

    # sin(x)
    a = 0.0
    b = math.pi / 2
    m = 20

    def sin_x(arg):
        return math.sin(arg)

    analyze_func(a, b, m, sin_x, axs)

    # cos(x + e^cos(x))
    # a = 3
    # b = 6
    # m = 10
    #
    # def sin_x(arg):
    #     return math.sin(arg)
    #
    # analyze_func(a, b, m, sin_x, axs)

    plt.show()


main()
