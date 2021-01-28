"""
Prog:   Lab2.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 2. Var 10. 2021
"""

import math
from Utils import input_from_file, get_option


def fast_bubble_sort(input_source):
    print("Lab 2. Fast bubble sort. Variant 10.")

    n = input_source("n", int)

    a = [n]
    for i in range(0, n):
        a.append(input_source("a[{}]".format(i + 1), float))

    print("Sorting...")

    k = 1

    while k < n:
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

    for i in range(0, n):
        print("a[{}] = {}".format(i + 1, a[i]))


def main():
    # input_source = get_input_source()
    input_source = input_from_file

    fast_bubble_sort(input_source)


main()
