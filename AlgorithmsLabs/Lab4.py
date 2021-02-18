"""
Prog:   Lab4.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 4. Var 10. 2021
"""

import matplotlib.pyplot as plt
import math

import numpy as np

from Input import get_input_source, get_option


def f_x(x):
    return x ** 3 + 8 * x - 6


def drv_f_x(x):
    return 3 * x ** 2 + 8


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


def create_axs(method_name, func, func_expression, a, b):
    # A4 page
    fig_width_cm = 21
    fig_height_cm = 29.7

    extra_margin_cm = 1
    margin_left_cm = 2 + extra_margin_cm
    margin_right_cm = 1.0
    margin_bottom_cm = 1.0 + extra_margin_cm
    margin_top_cm = 3.0 + extra_margin_cm
    inches_per_cm = 1 / 2.54

    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches

    margin_left = margin_left_cm * inches_per_cm
    margin_right = margin_right_cm * inches_per_cm
    margin_bottom = margin_bottom_cm * inches_per_cm
    margin_top = margin_top_cm * inches_per_cm

    left_margin = margin_left / fig_width
    right_margin = 1 - margin_right / fig_width
    bottom_margin = margin_bottom / fig_height
    top_margin = 1 - margin_top / fig_height

    fig_size = [fig_width, fig_height]

    plt.rc('figure', figsize=fig_size)

    fig_page1, axs_page1 = plt.subplots(1, 1)
    fig_page1.suptitle(method_name + ' nonlinear equation solving results: ' + func_expression +
                       '\n a={:.2f}, b={:.2f}'.format(a, b),
                       fontsize=16, style='normal')
    fig_page1.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
                              wspace=0.3, hspace=0.2)
    fig_page1.canvas.set_window_title('Lab 4. Nonlinear equation solving. Var. 10')
    axs_page1.set_title(r'$f(x)=$' + func_expression, fontsize=12, color='gray')
    axs_page1.set_ylabel(r'$f(x)$', rotation=0, loc='top', fontsize=10, color='gray')
    axs_page1.set_xlabel(r'$x$', fontsize=10, loc='right', color='gray')
    axs_page1.minorticks_on()
    axs_page1.grid(which='minor', alpha=0.2)
    axs_page1.grid(which='major', alpha=0.5)

    return axs_page1, fig_page1


def solve(a, b, func, func_drv, axs_page1):
    x_range = generate_x_range(a, b, 100)

    # draw original func
    y_range = generate_y_range(x_range, func)

    axs_page1.plot(x_range, y_range, color='green')

    # draw derivative func
    y_drv_range = generate_y_range(x_range, func_drv)

    axs_page1.plot(x_range, y_drv_range, color='blue')


def main():
    print("Lab 4. Nonlinear equations. Var. 10")

    equation_coefficients = ()

    a = -3.0
    b = 3.0

    ttt = f_x(0.0)

    method_name = "Newton"
    func_expression = r'$x^3 + 8x - 6$'

    axs_page1, fig_page1 = create_axs(method_name, f_x, func_expression, a, b)

    solve(a, b, f_x, drv_f_x, axs_page1, fig_page1)

    plt.show()


main()
