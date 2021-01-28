"""
Prog:   Utils.py
Auth:   Oleksii Krutko, IO-z91
Desc:   Algorithms. 2021
"""


def input_from_keyboard_repeat(var_name, var_type):
    while True:
        try:
            var_new = var_type(input("Enter {}:".format(var_name)))
        except ValueError:
            print("Type {} expected.".format(var_type))
        else:
            break

    return var_new


def input_from_file(var_name, var_type):
    with open("../venv/input_files/labs_input.txt") as reader:
        line = reader.readline()
        var_new = None

        while line != '':
            var_name_file, var_value_file = line.split(' ')
            if var_name_file == var_name:
                try:
                    var_new = var_type(var_value_file)
                except ValueError:
                    print("Type of variable {} must be {}.".format(var_name, var_type))
                else:
                    break

            line = reader.readline()

    if var_new is not None:
        print("{} = {}".format(var_name, var_new))
        return var_new
    else:
        print("Variable with name {} not found in {}".format(var_name, "input_files/labs_input.txt"))


def get_input_source():
    source_type = int(input("Input source type (1 - keyboard, 2 - input.txt):"))

    if source_type == 1:
        return input_from_keyboard_repeat
    elif source_type == 2:
        return input_from_file
    else:
        raise Exception("Wrong input type!")


def get_option(text, count):
    option = int(input(text))

    if option in range(1, count + 1):
        return option
    else:
        raise Exception("Wrong option!")
