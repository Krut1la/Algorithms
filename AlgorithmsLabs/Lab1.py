"""
Prog:   Lab1.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms Lab 1. 2021
"""


def input_repeat(var_name, type):
    var_new = None
    while type(var_new) is not type:
        try:
            var_new = input("Enter {}:".format(var_name))
        except ValueError:
            print("Type {} expected. {} got".format(type, type(var_new)))

    return var_new


def main():
    a = input_repeat("Enter a:", type(float))
    b = input_repeat("Enter b:", type(float))
    print(a + b)


main()
