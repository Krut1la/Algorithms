"""
Prog:   Lab1.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 1. Var 10. 2021
"""

import math

from Input import get_input_source, get_option, InputError


def linear_algo(a, b, c, d):
    try:
        Y1 = ((math.sqrt(a) + b ** 2) / (math.sqrt(b) - a ** 2)) + (math.sqrt((a * b) / (c * d)))
    except ZeroDivisionError:
        print("Calculation failed!")
        Y1 = float('nan')

    return Y1


def condition_algo(a, c, k):
    if k < 10:
        y = (a + c) ** 4 + (a - c) ** 2
    else:
        y = (a - c) ** 3 + (a + c) ** 2

    return y


def cyclic_algo(a, b, p):
    f = 0

    for i in range(1, p + 1):
        for j in range(1, p + 1):
            for k in range(1, p + 1):
                f = f + i * i * j * i * j * k * math.sqrt(a + b)

    return f


def main():
    option = get_option("Select algorithm (1 - Linear, 2 - Condition, 3 - Cyclic):", 3)

    if option == 1:
        input_source = get_input_source("lab1_linear_algo")
        print("Lab 1. Linear algorithm. Variant 10.")
        try:
            a = input_source.read_var("a", float, float("-inf"), float("inf"))
            b = input_source.read_var("b", float, float("-inf"), float("inf"))
            c = input_source.read_var("c", float, float("-inf"), float("inf"))
            d = input_source.read_var("d", float, float("-inf"), float("inf"))
        except InputError as ir:
            print("Error wile reading data from file: ", ir)
        else:
            print("Y1 = {:.2f}".format(linear_algo(a, b, c, d)))
    elif option == 2:
        input_source = get_input_source("lab1_condition_algo")
        print("Lab 1. Condition algorithm. Variant 10.")
        try:
            a = input_source.read_var("a", float, float("-inf"), float("inf"))
            c = input_source.read_var("c", float, float("-inf"), float("inf"))
            k = input_source.read_var("k", float, float("-inf"), float("inf"))
        except InputError as ir:
            print("Error wile reading data from file: ", ir)
        else:
            print("y = {:.2f}".format(condition_algo(a, c, k)))
    elif option == 3:
        input_source = get_input_source("lab1_cyclic_algo")
        print("Lab 1. Cyclic algorithm. Variant 10.")
        try:
            a = input_source.read_var("a", float, float("-inf"), float("inf"))
            b = input_source.read_var("b", float, float("-inf"), float("inf"))
            p = input_source.read_var("p", int, 1, 10000)
        except InputError as ir:
            print("Error wile reading data from file: ", ir)
        else:
            print("f = {:.2f}".format(cyclic_algo(a, b, p)))


main()
