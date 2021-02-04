"""
Prog:   Lab3.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 3. Var 10. 2021
"""

import numpy as np
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


def generate_interpolated_y_range(x_nodes, x_range, a):
    y = []
    for x in x_range:
        y.append(newton_polynomial(x_nodes, x, a))

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


def generate_sin_x_nodes(m):
    x = []
    y = []

    for j in range(0, m + 1):
        xj = (j * math.pi) / (2 * m)
        x.append(xj)
        y.append(math.sin(xj))

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


def analyze_func(a, b, m, func, axs):
    # draw original func
    x_range = generate_x_range(a, b, 100)
    y_range = generate_y_range(x_range, func)

    axs[0, 0].plot(x_range, y_range, color='green')

    # draw polynomial
    x_nodes = generate_x_range(a, b, m)
    y_nodes = generate_y_range(x_nodes, func)

    axs[0, 1].plot(x_nodes, y_nodes, 'ro', color='blue')

    coefficients = poly_newton_coefficient(x_nodes, y_nodes)

    y_interpolated = generate_interpolated_y_range(x_nodes, x_range, coefficients)

    axs[0, 1].plot(x_range, y_interpolated, color='red')

    # draw error

    x_nodes_plus_1 = generate_x_range(a, b, m + 1)
    y_nodes_plus_1 = generate_y_range(x_nodes_plus_1, func)

    coefficients_plus_1 = poly_newton_coefficient(x_nodes_plus_1, y_nodes_plus_1)

    for j in range(0, m):
        x_error = []
        y_error_polynomial = []
        y_error_func = []
        for x in x_range:
            try:
                xp = (x - x_nodes[j]) / (x_nodes[j + 1] - x_nodes[j])
                if 1.0 < xp or xp < 0.0:
                    continue
                log_delta_P_P1 = -math.log(math.fabs(newton_polynomial(x_nodes, x, coefficients) -
                                                     newton_polynomial(x_nodes_plus_1, x, coefficients_plus_1)), 10)
                log_delta_P_func = -math.log(math.fabs(newton_polynomial(x_nodes, x, coefficients) - func(x)), 10)
            except ValueError:
                pass
            else:
                x_error.append(xp)
                y_error_polynomial.append(log_delta_P_P1)
                y_error_func.append(log_delta_P_func)

        axs[0, 2].plot(x_error, y_error_polynomial, color='black')
        axs[0, 3].plot(x_error, y_error_func, color='black')

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
    # a = 0
    # b = math.pi/2
    # m = 20

    # def sin_x(arg):
    #        return math.sin(arg)

    #   analyze_func(a, b, m, sin_x, 0, axs, "sin(x)")

    # cos(x + e^cos(x))
    a = 3
    b = 6
    m = 10

    def sin_x(arg):
        return math.sin(arg)

    analyze_func(a, b, m, sin_x, axs)

    plt.show()


main()
