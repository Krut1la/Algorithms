"""
Prog:   Lab4.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 4. Var 10. 2021
"""

import matplotlib.pyplot as plt
import math

from AlgorithmsLabs.Input import get_input_source, InputError


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


def create_axs(method_name, func_expression):
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
    fig_page1.suptitle(method_name + ' nonlinear equation solving results:',
                       fontsize=16, style='normal')
    fig_page1.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
                              wspace=0.3, hspace=0.3)
    fig_page1.canvas.set_window_title('Lab 4. Nonlinear equation solving. Var. 10')
    axs_page1[0].set_title(func_expression, fontsize=12, color='gray')
    axs_page1[0].set_ylabel(r'$f(x)$', rotation=0, loc='top', fontsize=10, color='gray')
    axs_page1[0].set_xlabel(r'$x$', fontsize=10, loc='right', color='gray')
    axs_page1[0].set_ylim([-100, 100])
    axs_page1[0].minorticks_on()
    axs_page1[0].grid(which='minor', alpha=0.2)
    axs_page1[0].grid(which='major', alpha=0.5)

    return axs_page1, fig_page1


def find_root_precise(boundary, func, func_first_drv, func_second_drv, epsilon):
    """
    Newton method
    :param boundary:
    :param func:
    :param func_first_drv:
    :param func_second_drv:
    :param epsilon:
    :return:
    """
    a = boundary[0]
    b = boundary[1]

    if func(a) * func(b) > 0.0:
        raise ValueError("Func must change sign between {:.2f} and {:.2f}".format(a, b))

    k = []

    if math.fabs(b - a) < epsilon:
        return (a + b) / 2, ()

    if func(b) * func_second_drv(b) > 0.0:
        z = b
        b = a
        a = z

    while len(k) < 100:
        x = b - func(b) / func_first_drv(b)
        k.append((-func_first_drv(x) * x + func(x), func_first_drv(x)))

        if math.fabs(x - b) < epsilon:
            return x, k

        b = x


def lagrange_upper_limit(a):
    """
    Lagrange theorem
    :param a: equation
    :return: Upper limit of positive roots
    """
    a_pos = [i for i in a]
    n = len(a) - 1
    if a[n] < 0.0:
        a_pos = [-i for i in a]

    try:
        i_first_negative = a_pos.index(next(el for el in reversed(a_pos) if el < 0.0))
    except StopIteration:
        raise ValueError

    C = math.fabs(max(filter(lambda el: el < 0.0, a_pos), key=lambda el: math.fabs(el)))

    return 1 + (C / a_pos[n]) ** (1 / (n - i_first_negative))


def find_all_roots_ranges(a, func, func_first_drv):
    """
    Finds root ranges using theorems
    :param a: equation
    :param func: function
    :param func_first_drv: first derivative
    :return: ranges
    """
    n = len(a) - 1

    P1_x = [i for i in reversed(a)]
    P2_x = [-a[i] if i % 2 > 0 and i > 0 else a[i] for i in range(0, len(a))]
    P3_x = [i for i in reversed(P2_x)]

    try:
        R = lagrange_upper_limit(a)

        R1 = lagrange_upper_limit(P1_x)
        R2 = lagrange_upper_limit(P2_x)
        R3 = lagrange_upper_limit(P3_x)

        x_plus_min = 1 / R1
        x_plus_max = R

        x_minus_min = -R2
        x_minus_max = -1 / R3
    except ValueError:
        max_abs_a = math.fabs(max(a[:-1], key=lambda el: math.fabs(el)))
        max_abs_b = math.fabs(max(a[1:], key=lambda el: math.fabs(el)))
        x_plus_min = 1 / (1 + max_abs_b / math.fabs(a[0]))
        x_plus_max = 1 + max_abs_a / math.fabs(a[n])

        x_minus_min = -(1 + max_abs_a / math.fabs(a[n]))
        x_minus_max = -(1 / (1 + max_abs_b / math.fabs(a[0])))

    has_min_pair_complex_pair = False

    for i in range(1, len(a) - 1):
        if (a[i] ** 2) < a[i - 1] * a[i + 1]:
            has_min_pair_complex_pair = True
            break

    S_pos = count_roots(a)

    S_neg = count_roots(P2_x)

    # find single root ranges
    ranges = []

    if S_pos + S_neg > 0:
        a = x_plus_min
        b = x_plus_max
        find_single_root_ranges(a, b, func, func_first_drv, ranges)

        a = x_minus_min
        b = x_minus_max
        find_single_root_ranges(a, b, func, func_first_drv, ranges)

    return ranges, has_min_pair_complex_pair


def count_roots(a):
    """
    Counts all roots
    :param a: equation
    :return: roots count
    """
    roots_count = 0

    n = len(a) - 1
    start_sign = a[n] > 0.0

    for ai in reversed(a):
        if math.fabs(ai) < 1e-6:
            continue
        if (ai > 0.0) != start_sign:
            start_sign = ai > 0.0
            roots_count = roots_count + 1

    return roots_count


def find_single_root_ranges(a, b, func, func_first_drv, ranges):
    """
    Splits range into ranges containing only single real root
    :param a: range start
    :param b: range end
    :param func: function
    :param func_first_drv: first derivative
    :param ranges: output container
    :return:
    """
    h = 0.01
    x = a
    func_sign_a = func(a) > 0.0
    first_drv_sign_a = func_first_drv(a) > 0.0
    while x <= b:
        func_sign_b = func(x) > 0.0
        first_drv_sign_b = func_first_drv(x) > 0.0

        if first_drv_sign_b != first_drv_sign_a:
            a = x
            func_sign_a = func(a) > 0.0
            first_drv_sign_a = func_first_drv(a) > 0.0

        if func_sign_a != func_sign_b:
            ranges.append((a, x))
            func_sign_a = func(x) > 0.0
            first_drv_sign_a = func_first_drv(x) > 0.0

        x = x + h


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
                "{}{}".format(sign, str(round(ua, 2) if ua % 1 else int(ua))) + expression
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

    input_source = get_input_source("lab4_nonlinear_equation")
    try:
        n = input_source.read_var("n", int, 1, 10)

        a = []
        for i in range(0, n + 1):
            a.append(input_source.read_var("a[{}]".format(i), float, float("-inf"), float("inf")))

    except InputError as ir:
        print("Error wile reading data from file: ", ir)
        return

    # test_equation = (-3.0, -7.0, 8.0, -5.0, 2.0, 1.0)
    # test_equation_no_roots = (3.0, 7.0, 8.0, 5.0, 2.0, 0.0, 1.0)

    equation = a
    # equation = test_equation
    # equation = test_equation_no_roots
    first_drv_equation = drv_alg_equation(equation)
    second_drv_equation = drv_alg_equation(first_drv_equation)

    ranges, has_complex_roots = find_all_roots_ranges(equation,
                                                      lambda x: alg_equation(x, equation),
                                                      lambda x: alg_equation(x, first_drv_equation))

    if len(ranges) == 0:
        print("No real roots found!")
        return

    min_a = min(ranges, key=lambda boundary: boundary[0])
    max_b = max(ranges, key=lambda boundary: boundary[1])

    method_name = "Newton"
    func_expression = get_equation_expression(equation)
    first_drv_expression = get_equation_expression(first_drv_equation)
    second_drv_expression = get_equation_expression(second_drv_equation)

    axs_page1, fig_page1 = create_axs(method_name, r"$f(x)={eq}$".format(eq=func_expression) + "\n" +
                                      r"$f'(x)={eq1d}$".format(eq1d=first_drv_expression) + "\n" +
                                      r"$f''(x)={eq2d}$".format(eq2d=second_drv_expression))

    x_range = generate_x_range(min_a[0] - 5.0, max_b[1] + 5.0, 100)

    # draw original func
    y_range = generate_y_range(x_range, lambda x: alg_equation(x, equation))

    axs_page1[0].plot(x_range, y_range, color='green')

    # draw derivative func
    y_drv_range = generate_y_range(x_range, lambda x: alg_equation(x, first_drv_equation))

    axs_page1[0].plot(x_range, y_drv_range, color='blue', linestyle="--", alpha=0.4)

    # draw 2nd derivative func
    y_2nd_drv_range = generate_y_range(x_range, lambda x: alg_equation(x, second_drv_equation))

    axs_page1[0].plot(x_range, y_2nd_drv_range, color='brown', linestyle=":", alpha=0.4)

    table_data = [["" for j in range(5)] for i in range(len(ranges))]

    for n in range(0, len(ranges)):
        table_data[n][0] = str(n + 1)
        table_data[n][1] = "{:.2f}".format(ranges[n][0])
        table_data[n][2] = "{:.2f}".format(ranges[n][1])
        x_root, tangents = find_root_precise(ranges[n],
                                             lambda x: alg_equation(x, equation),
                                             lambda x: alg_equation(x, first_drv_equation),
                                             lambda x: alg_equation(x, second_drv_equation), 1e-5)

        # draw root finding process
        axs_page1[0].axvspan(x_root, x_root, alpha=0.5, color='black')

        x_tangent_range = generate_x_range(ranges[n][0], ranges[n][1], 50)

        for i in range(0, len(tangents)):
            y_tangent_range = generate_y_range(x_tangent_range, lambda x: alg_equation(x, tangents[i]))
            axs_page1[0].plot(x_tangent_range, y_tangent_range, color='red', alpha=0.5)

        table_data[n][3] = "{:.5f}".format(x_root)
        table_data[n][4] = str(len(tangents))

    # draw table
    if has_complex_roots:
        axs_page1[1].set_title('Table, has at least 2 complex roots ', fontsize=12, color='gray')
    else:
        axs_page1[1].set_title('Table', fontsize=12, color='gray')
    axs_page1[1].axis('off')
    col_label = (r'$root$', r'$a$', r'$b$', r'$x$', "Iterations")
    the_table = axs_page1[1].table(cellText=table_data, colLabels=col_label, loc='center')
    the_table.set_fontsize(10)
    the_table.scale(1.0, 1.35)

    plt.show()


main()
