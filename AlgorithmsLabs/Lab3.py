"""
Prog:   Lab3.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 3. Var 10. 2021
"""

import matplotlib.pyplot as plt
import math
from Input import get_input_source, get_option


class Interpolator(object):
    def __init__(self, x_nodes, y_nodes):
        """
        :param x_nodes: x values in nodes
        :param y_nodes: y values in nodes
        """
        self._x_nodes = x_nodes
        self._y_nodes = y_nodes

    @classmethod
    def get_method_name(cls):
        pass

    def interpolate(self, x):
        """
        Evaluates polynomial in x point
        :param x: point to evaluate
        :return: y value
        """
        pass

    def generate_y_range(self, x_range):
        """

        """
        y = []
        for x in x_range:
            y.append(self.interpolate(x))

        return y


class InterpolatorLagrange(Interpolator):
    def __init__(self, x_nodes, y_nodes):
        super(InterpolatorLagrange, self).__init__(x_nodes, y_nodes)

    @classmethod
    def get_method_name(cls):
        return "Lagrange"

    def interpolate(self, x):
        n = len(self._x_nodes)

        S = 0.0
        for j in range(1, n + 1):
            P = 1.0

            for i in range(1, n + 1):
                if i == j:
                    continue

                P = P * (x - self._x_nodes[i - 1]) / (self._x_nodes[j - 1] - self._x_nodes[i - 1])

            S = S + P * self._y_nodes[j - 1]

        return S


class InterpolatorNewton(Interpolator):
    def __init__(self, x_nodes, y_nodes):
        super(InterpolatorNewton, self).__init__(x_nodes, y_nodes)
        self.method_name = "Newton"
        self.__end_diffs = self.__get_poly_newton_end_diffs()

    @classmethod
    def get_method_name(cls):
        return "Newton"

    def __get_poly_newton_end_diffs(self):
        """
        Cals end diffs of Newton polynomial using the general formula (1.6)
        :return: array of coefficients
        """
        m = len(self._x_nodes)

        end_diffs = []

        for n in range(1, m + 1):
            S = 0
            for j in range(1, n + 1):
                P = 1
                for i in range(1, n + 1):
                    if i == j:
                        continue

                    P = P * (self._x_nodes[j - 1] - self._x_nodes[i - 1])

                S = S + self._y_nodes[j - 1] / P

            end_diffs.append(S)

        return end_diffs

    def interpolate(self, x):
        n = len(self._x_nodes) - 1
        p = self.__end_diffs[n]
        for k in range(1, n + 1):
            p = self.__end_diffs[n - k] + (x - self._x_nodes[n - k]) * p

        return p


def get_delta_n(x, x_nodes, func):
    n = len(x_nodes)
    delta_n = 0.0

    for j in range(0, n):
        sigma_j = math.fabs(func(x_nodes[j])) * 1e-15
        A_j = 1.0

        for i in range(0, n):
            if i == j:
                continue

            A_j = A_j * math.fabs((x - x_nodes[i]) / (x_nodes[j] - x_nodes[i]))

        delta_n = delta_n + sigma_j * A_j

    return delta_n


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


def create_axs(method_name, func_expression, a, b, m, j):
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
    fig_page1.suptitle(method_name + ' interpolation method results: ' + func_expression +
                       '\n a={:.2f}, b={:.2f}, m={}, j={}'.format(a, b, m, j),
                       fontsize=16, style='normal')
    fig_page1.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
                              wspace=0.3, hspace=0.2)
    fig_page1.canvas.set_window_title('Lab 3. Polynomial interpolation. Var. 10')
    axs_page1[0].set_title('Precise: ' + r'$f(x)=$' + func_expression, fontsize=12, color='gray')
    axs_page1[0].set_ylabel(r'$f(x)$', rotation=0, loc='top', fontsize=10, color='gray')
    axs_page1[0].set_xlabel(r'$x$', fontsize=10, loc='right', color='gray')
    axs_page1[0].grid()

    axs_page1[1].set_title('Nodes, interpolated', fontsize=12, color='gray')
    axs_page1[1].set_ylabel(r'$P_{{{n}}}(x)$'.format(n=m), rotation=0, loc='top', fontsize=10, color='gray')
    axs_page1[1].set_xlabel(r'$x$', fontsize=10, style='italic', loc='right', color='gray')
    axs_page1[1].grid()

    fig_page2, axs_page2 = plt.subplots(2, 2)
    fig_page2.suptitle(method_name + ' interpolation method error evaluation: ' + func_expression +
                       '\n a={:.2f}, b={:.2f}, m={}, j={}'.format(a, b, m, j),
                       fontsize=16, style='normal')
    fig_page2.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
                              wspace=0.3, hspace=0.3)
    fig_page2.canvas.set_window_title('Lab 3. Polynomial interpolation. Var. 10')

    axs_page2[0, 0].set_title('Error: ' + r'$\Delta_{n}=P_{n}(x) - P_{n+1}(x)$', fontsize=12, color='gray')
    axs_page2[0, 0].set_ylabel(r'$-lg|\Delta_{n}|$', rotation=0, loc='top', fontsize=8, color='gray')
    axs_page2[0, 0].set_xlabel(r'$\overline{x}=\frac{x-x_{j}}{x_{j+1}-x_{j}}$', fontsize=10, loc='right', color='gray')
    axs_page2[0, 0].grid()

    axs_page2[0, 1].set_title('Error: ' + r'$\Delta_{n}=P_{n}(x) - f(x)$', fontsize=12, color='gray')
    axs_page2[0, 1].set_ylabel(r'$-lg|\Delta_{n}|$', rotation=0, loc='top', fontsize=8, color='gray')
    axs_page2[0, 1].set_xlabel(r'$\overline{x}=\frac{x-x_{j}}{x_{j+1}-x_{j}}$', fontsize=10, loc='right', color='gray')
    axs_page2[0, 1].grid()

    axs_page2[1, 0].set_title('FP error: ' + r'$\sigma=$' + func_expression + r'$10^{-15}$', fontsize=12, color='gray')
    axs_page2[1, 0].set_ylabel(r'$-lg|\Delta_{n}|$', rotation=0, loc='top', fontsize=10, color='gray')
    axs_page2[1, 0].set_xlabel(r'$\overline{x}=\frac{x-x_{j}}{x_{j+1}-x_{j}}$', fontsize=10, loc='right', color='gray')
    axs_page2[1, 0].set_ylim([0.0, 15.0])
    axs_page2[1, 0].grid()

    axs_page2[1, 1].set_title('Table', fontsize=12, color='gray')

    return axs_page1, axs_page2, fig_page1, fig_page2


def analyze_func(a, b, m, j, func, interpolator_type, axs_page1, axs_page2):
    x_points = 300

    # draw original func
    x_range = generate_x_range(a, b, 1000)
    y_range = generate_y_range(x_range, func)

    axs_page1[0].plot(x_range, y_range, color='green')

    # draw nodes
    x_nodes = generate_x_range(a, b, m)
    y_nodes = generate_y_range(x_nodes, func)

    x_middle = x_nodes[j] + (x_nodes[j + 1] - x_nodes[j]) / 2

    axs_page1[1].plot(x_nodes, y_nodes, 'ro', color='blue')

    # draw polynomial

    p_m = interpolator_type(x_nodes, y_nodes)

    y_interpolated = p_m.generate_y_range(x_range)

    axs_page1[1].plot(x_range, y_interpolated, color='red')

    axs_page1[1].axvspan(x_nodes[j], x_nodes[j + 1], alpha=0.5, color='yellow')
    axs_page1[1].axvspan(x_middle, x_middle, alpha=0.5, color='blue')

    # draw error

    k = j
    table_data = [["" for j in range(4)] for i in range(m - k - 1)]
    for n in range(1, m - k):

        table_data[n - 1][0] = "{}".format(n)

        x_nodes_n = x_nodes[k:n + k + 1]
        y_nodes_n = y_nodes[k:n + k + 1]

        x_nodes_n_1 = x_nodes[k:n + k + 2]
        y_nodes_n_1 = y_nodes[k:n + k + 2]

        p_n = interpolator_type(x_nodes_n, y_nodes_n)
        p_n_1 = interpolator_type(x_nodes_n_1, y_nodes_n_1)

        p_n_y_middle = p_n.interpolate(x_middle)
        p_n_1_y_middle = p_n_1.interpolate(x_middle)
        f_y_middle = func(x_middle)

        table_data[n - 1][1] = "{:e}".format(p_n_y_middle - p_n_1_y_middle)
        table_data[n - 1][2] = "{:e}".format(p_n_y_middle - f_y_middle)
        table_data[n - 1][3] = "{:.2f}".format(1 - (p_n_y_middle - f_y_middle)/(p_n_y_middle - p_n_1_y_middle))

        x_range_test = generate_x_range(x_nodes[j], x_nodes[j + 1], x_points)

        x_error = []
        y_error_polynomial = []
        y_error_func = []
        y_error_func_float = []
        for x in x_range_test:
            try:
                xp = (x - x_nodes[j]) / (x_nodes[j + 1] - x_nodes[j])

                p_n_y = p_n.interpolate(x)
                p_n_1_y = p_n_1.interpolate(x)
                f_y = func(x)

                log_delta_p_p_1 = -math.log(math.fabs(p_n_y - p_n_1_y), 10)
                log_delta_p_func = -math.log(math.fabs(p_n_y - f_y), 10)

                delta_n = get_delta_n(x, x_nodes_n, func)

                log_func_float = -math.log(math.fabs(delta_n), 10)
            except ValueError:
                pass
            else:
                x_error.append(xp)
                y_error_polynomial.append(log_delta_p_p_1)
                y_error_func.append(log_delta_p_func)
                y_error_func_float.append(log_func_float)

        # draw error evaluation
        axs_page2[0, 0].plot(x_error, y_error_polynomial, color='black')
        axs_page2[0, 0].axvspan(0.5, 0.5, alpha=0.5, color='blue')

        if n < 10:
            axs_page2[0, 0].annotate('n={}'.format(n),
                               xy=(x_error[len(x_error) - 1], y_error_polynomial[len(y_error_polynomial) - 1]),
                               xycoords='data', xytext=(1.1, 0.1 * n), textcoords='axes fraction', fontsize=8,
                               color='gray', arrowprops=dict(color='gray', shrink=0.05, width=0.01, headwidth=0.2),
                               horizontalalignment='right', verticalalignment='top',
                               )
        axs_page2[0, 1].plot(x_error, y_error_func, color='black')
        axs_page2[0, 1].axvspan(0.5, 0.5, alpha=0.5, color='blue')
        axs_page2[1, 0].plot(x_error, y_error_func_float, color='black')
        axs_page2[1, 0].axvspan(0.5, 0.5, alpha=0.5, color='blue')

    # draw table
    axs_page2[1, 1].axis('off')
    col_label = (r'$n$', r'$\Delta_{n}$', r'$\Delta_{n}exact$', r'$k$')
    the_table = axs_page2[1, 1].table(cellText=table_data, colLabels=col_label, loc='center')
    the_table.set_fontsize(20)
    the_table.scale(1.0, 1.35)


def main():
    print("Lab 3. Polynomial interpolation. Var. 10")

    option_func = get_option("Select function (1 - sin(x), 2 - cos(x + e^cos(x)):", 2)
    option_method = get_option("Select polynomial (1 - Lagrange, 2 - Newton:", 2)

    def sin_x(arg):
        return math.sin(arg)

    def v_10(arg):
        return math.cos(arg + math.exp(math.cos(arg)))

    if option_func == 1:
        func = sin_x
        func_expression = r'$sin(x)$'
        input_source = get_input_source("lab3_sin_interpolation")
    else:
        func = v_10
        func_expression = r'$cos(x + e^{cos(x)})$'
        input_source = get_input_source("lab3_v_10_interpolation")

    if option_method == 1:
        interpolator = InterpolatorLagrange
    else:
        interpolator = InterpolatorNewton

    a = input_source.read_var("a", float, float("-inf"), float("inf"))
    b = input_source.read_var("b", float, a, float("inf"))
    m = input_source.read_var("m", int, 5, 20)
    j = input_source.read_var("j", int, 0, m - 1)

    axs_page1, axs_page2, fig_page1, fig_page2 = create_axs(interpolator.get_method_name(), func_expression, a, b, m, j)

    analyze_func(a, b, m, j, func, interpolator, axs_page1, axs_page2)

    # fig_page2.savefig("E:\Krut1la\KPI\Grade 2\Part 2\Algorithms\Labs\Lab3\lab3_sin_fig_4.png")
    # fig_page1.savefig("E:\Krut1la\KPI\Grade 2\Part 2\Algorithms\Labs\Lab3\lab3_sin_fig_3.png")

    plt.show()


main()
