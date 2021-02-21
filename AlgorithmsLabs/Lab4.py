"""
Prog:   Lab4.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 4. Var 10. 2021
"""

import matplotlib.pyplot as plt
import math

import numpy as np

from Input import get_input_source, get_option


# def f_x(x):
#     return x ** 3 + 8 * x - 6
#
#
# def drv_f_x(x):
#     return 3 * x ** 2 + 8


def alg_equation(x, a):
    n = len(a)
    y = 0.0
    for i in range(0, n):
        y = y + a[i] * (x ** i)
    return y


def drv_alg_equation(a):
    n = len(a)
    dvr_coefficients = []
    for i in range(1, n):
        dvr_coefficients.append(a[i] * i)

    return dvr_coefficients


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


def create_axs(method_name, func_expression, a, b):
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

    fig_page1, axs_page1 = plt.subplots(2, 1)
    fig_page1.suptitle(method_name + ' nonlinear equation solving results:\n' + func_expression +
                       '\n a={:.2f}, b={:.2f}'.format(a, b),
                       fontsize=16, style='normal')
    fig_page1.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
                              wspace=0.3, hspace=0.2)
    fig_page1.canvas.set_window_title('Lab 4. Nonlinear equation solving. Var. 10')
    axs_page1[0].set_title(func_expression, fontsize=12, color='gray')
    axs_page1[0].set_ylabel(r'$f(x)$', rotation=0, loc='top', fontsize=10, color='gray')
    axs_page1[0].set_xlabel(r'$x$', fontsize=10, loc='right', color='gray')
    axs_page1[0].minorticks_on()
    axs_page1[0].grid(which='minor', alpha=0.2)
    axs_page1[0].grid(which='major', alpha=0.5)

    axs_page1[1].set_title('Table', fontsize=12, color='gray')

    return axs_page1, fig_page1


def find_root_precise(boundary, func, func_first_drv, func_second_drv):
    if func(boundary[0]) * func(boundary[1]) > 0.0:
        raise ValueError("Func must change sign between {:.2f} and {:.2f}".format(boundary[0], boundary[1]))

    return 0.0, 5


def find_all_roots_boundaries(a):
    return [(-3.0, 3.0)]


def get_equation_expression(a):
    """
    a[n-1]x^(n-1) + a[n-2]x^(n-2) ... a[1]x + a[0]
    :param a: coefficients
    :return: expression string
    """
    expression = ""
    for n in range(0, len(a)):
        if math.fabs(a[n]) < 1e-3:
            continue

        sign = ""
        if a[n] < 0.0:
            sign = "-"
        elif n != len(a) - 1:
            sign = "+"

        ua = math.fabs(a[n])

        if n == 0:
            expression = \
                "{}{}".format(sign, "" if (ua - 1.0) < 1e-3 else str(round(ua, 2) if ua % 1 else int(ua))) + expression
        elif n == 1:
            expression = \
                "{}{}x".format(sign, "" if (ua - 1.0) < 1e-3 else str(round(ua, 2) if ua % 1 else int(ua))) + expression
        else:
            expression = \
                "{}{}x^{}".format(sign, "" if (ua - 1.0) < 1e-3 else str(round(ua, 2) if ua % 1 else int(ua)), n) \
                + expression

    expression = expression

    return expression


def main():
    print("Lab 4. Nonlinear equations. Var. 10")

    v_10_equation = (-6.0, 8.0, 0.0, 1.0)

    equation = v_10_equation
    first_drv_equation = drv_alg_equation(equation)
    second_drv_equation = drv_alg_equation(first_drv_equation)

    boundaries = find_all_roots_boundaries(equation)

    min_a = min(boundaries, key=lambda boundary: boundary[0])
    max_b = max(boundaries, key=lambda boundary: boundary[1])

    method_name = "Newton"
    func_expression = get_equation_expression(equation)
    first_drv_expression = get_equation_expression(first_drv_equation)
    second_drv_expression = get_equation_expression(second_drv_equation)

    axs_page1, fig_page1 = create_axs(method_name, r"$f(x)={eq}, f'(x)={eq1d}, f''(x)={eq2d}$"
                                      .format(eq=func_expression, eq1d=first_drv_expression,
                                              eq2d=second_drv_expression), min_a[0], max_b[1])

    x_range = generate_x_range(min_a[0], max_b[1], 100)

    # draw original func
    y_range = generate_y_range(x_range, lambda x: alg_equation(x, equation))

    axs_page1[0].plot(x_range, y_range, color='green')

    # draw derivative func
    y_drv_range = generate_y_range(x_range, lambda x: alg_equation(x, first_drv_equation))

    axs_page1[0].plot(x_range, y_drv_range, color='blue', alpha=0.5)

    # draw 2nd derivative func
    y_2nd_drv_range = generate_y_range(x_range, lambda x: alg_equation(x, second_drv_equation))

    axs_page1[0].plot(x_range, y_2nd_drv_range, color='yellow', alpha=0.5)

    table_data = [["" for j in range(5)] for i in range(len(boundaries))]

    for n in range(0, len(boundaries)):
        table_data[n][0] = str(n + 1)
        table_data[n][1] = "{:.2f}".format(boundaries[n][0])
        table_data[n][2] = "{:.2f}".format(boundaries[n][1])
        x_root, iterations = find_root_precise(boundaries[n],
                                               lambda x: alg_equation(x, equation),
                                               lambda x: alg_equation(x, first_drv_equation),
                                               lambda x: alg_equation(x, second_drv_equation))

        table_data[n][3] = "{:.2f}".format(x_root)
        table_data[n][4] = str(iterations)

        # draw root finding process

    # draw table
    axs_page1[1].axis('off')
    col_label = (r'$root$', r'$a$', r'$b$', r'$x$', "iterations")
    the_table = axs_page1[1].table(cellText=table_data, colLabels=col_label, loc='center')
    the_table.set_fontsize(10)
    the_table.scale(1.0, 1.35)

    plt.show()


main()
