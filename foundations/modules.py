import sys

from lists import sorted_set as sorted_set_example


def usage():
    """
    A sample description
    """
    pass


def fibonacci(max):
    a = 0
    b = 1
    series = [a, b]
    while (a + b) <= max:
        c = a + b
        a = b
        b = c
        series.append(c)

    return series


if __name__ == "__main__":
    for arg in sys.argv:
        print(f"Argument: {arg}")

    print(f"Fibonacci series: {fibonacci(int(sys.argv[1]))}")

    sorted_set_example()
    print(usage.__doc__)
