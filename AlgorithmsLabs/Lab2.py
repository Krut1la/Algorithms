"""
Prog:   Lab2.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 2. Var 10. 2021
"""

import random

import matplotlib.pyplot as plt
import time
from Input import get_input_source, InputError, get_option


def fast_bubble_sort(a):
    n = len(a)
    k = 1

    while True:
        P = 0
        for i in range(0, n - k):
            if a[i] > a[i + 1]:
                t = a[i]
                a[i] = a[i + 1]
                a[i + 1] = t
                P = 1

        if P == 0:
            break

        k = k + 1


def analyze(a, num_iterations):
    """
    Runs sorting num_iterations times and calculates average time spent
    :param a: array with data
    :param num_iterations:
    :return: average time in ms
    """

    diff_ns = 0.0
    for i in range(0, num_iterations):
        a_copy = a.copy()
        start_time = time.process_time()

        fast_bubble_sort(a_copy)

        end_time = time.process_time()

        diff_ns = diff_ns + (end_time - start_time)

    diff_arg_ms = 1000 * diff_ns/num_iterations

    return diff_arg_ms


def main():
    option = get_option("Select job (1 - demo, 2 - speed analyze):", 2)

    if option == 1:
        input_source = get_input_source("lab2_fast_bubble")
        print("Lab 2. Fast bubble sort demo. Variant 10.")
        try:
            n = input_source.read_var("n", int, 1, 10000)

            a = []
            for i in range(0, n):
                a.append(input_source.read_var("a[{}]".format(i + 1), float, float("-inf"), float("inf")))

        except InputError as ir:
            print("Error wile reading data from file: ", ir)
        else:
            print("Sorting...")

            fast_bubble_sort(a)

            for i in range(0, n):
                print("a[{}] = {}".format(i + 1, a[i]))
    elif option == 2:
        print("Lab 2. Fast bubble sort speed analyze. Variant 10.")
        fig, axs = plt.subplots()
        fig.canvas.set_window_title('Lab 2. Fast bubble sort speed analyze. Variant 10.')
        axs.set_title('g(n)')
        axs.set_ylabel('time (ms)')
        axs.set_xlabel('n (elements)')

        print("Worst case. Elements presorted wrong way")
        n_step = 10
        x = []
        y = []
        for n in range(n_step, n_step*10, n_step):
            a = []

            a_value = -n/2
            for i in range(0, n):
                a.append(-(a_value + i))

            time_ms = analyze(a, 1)

            x.append(n)
            y.append(time_ms)

            axs.plot(x, y, color='red')

        print("Best case. Elements presorted right way")

        x = []
        y = []
        for n in range(n_step, n_step*10, n_step):
            a = []
            a_value = -n / 2
            for i in range(0, n):
                a.append(a_value + i)

            time_ms = analyze(a, 1)

            x.append(n)
            y.append(time_ms)

            axs.plot(x, y, color='green')

        print("Average case. Elements not sorted")

        x = []
        y = []
        for n in range(n_step, n_step*10, n_step):
            a = []
            for i in range(0, n):
                a.append((random.random() - 0.5) * n / 2)

            time_ms = analyze(a, 1)

            x.append(n)
            y.append(time_ms)

            axs.plot(x, y, color='brown')

        plt.show()


main()
