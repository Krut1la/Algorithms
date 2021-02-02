"""
Prog:   Lab3.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 3. Var 10. 2021
"""

import matplotlib.pyplot as plt
import math
import numpy as np

def _poly_newton_coefficient(x, y):
    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (x[k:m] - x[k - 1])

    return a


def newton_polynomial(x_data, y_data, x, a):

    n = len(x_data) - 1
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k]) * p
    return p


def generate_interpolated_nodes(a, b, x_nodes, y_nodes, n):

    ax = _poly_newton_coefficient(x_nodes, y_nodes)

    x = []
    y = []
    step = (b - a) / n

    for i in range(0, n):
        xi = a + i * step
        x.append(xi)
        y.append(newton_polynomial(x_nodes, y_nodes, xi, ax))

    return x, y


def generate_nodes(a, b, n):
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
