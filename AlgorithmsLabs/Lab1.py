"""
Prog:   Lab1.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 1. Var 10. 2021
"""

import math
from Utils import get_input_source, get_option


def linear_algo(input_source):
    print("Lab 1. Linear algorithm. Variant 10.")
    a = input_source("a", float)
    b = input_source("b", float)
    c = input_source("c", float)
    d = input_source("d", float)

    try:
        Y1 = ((math.sqrt(a) + b ** 2) / (math.sqrt(b) - a ** 2)) + (math.sqrt((a * b) / (c * d)))
        print("Y1 = {}".format(Y1))
    except ZeroDivisionError:
        print("Calculation failed!")


def condition_algo(input_source):
    print("Lab 1. Condition algorithm. Variant 10.")
    a = input_source("a", float)
    c = input_source("c", float)
    k = input_source("k", float)

    if k < 10:
        y = (a + c) ** 4 + (a - c) ** 2
    else:
        y = (a - c) ** 3 + (a + c) ** 2

    print("y = {}".format(y))


def cyclic_algo(input_source):
    print("Lab 1. Cyclic algorithm. Variant 10.")
    a = input_source("a", float)
    b = input_source("b", float)
    p = input_source("p", int)

    f = 0

    for i in range(1, p + 1):
        for j in range(1, p + 1):
            for k in range(1, p + 1):
                f = f + i * i * j * i * j * k * math.sqrt(a + b)

    print("f = {}".format(f))


def main():
    input_source = get_input_source()

    option = get_option("Select algorithm (1 - Linear, 2 - Condition, 3 - Cyclic):", 3)

    if option == 1:
        linear_algo(input_source)
    elif option == 2:
        condition_algo(input_source)
    elif option == 3:
        cyclic_algo(input_source)


main()
