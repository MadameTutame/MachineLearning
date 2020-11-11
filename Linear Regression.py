import numpy as np
import matplotlib.pyplot as plt

"""
    We have to approximate the dependence of x-values and t-values in this task.
"""


def calculate_parameters(length, degree, x_values, y_values):
    """
    plan_matrix: plan matrix
    w: weight
    the basic function: exponentiation
    """

    plan_matrix = np.zeros((length, degree + 1))
    for i in range(degree + 1):
        plan_matrix[:, i] = x_values ** i

    w = np.linalg.inv(plan_matrix.T @ plan_matrix) @ plan_matrix.T @ y_values
    return w, plan_matrix


def get_approximation(length, degree, x_values, y_values):
    """
    :param length: size of sample
    :param degree: degree for basis function calculating
    :param x_values: x values
    :param y_values: y values
    """

    w, F = calculate_parameters(length, degree, x_values, y_values)
    F = np.array(F)

    Y = F @ w.T

    plt.title("For N = " + str(length))
    plt.plot(x, z, color='red', linestyle='-', label="z(x)")
    plt.plot(x, Y, color='green', linestyle='-', label="Degree = " + str(degree))
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.legend(loc="lower right")
    plt.show()


def calculate_error(length, x_values, y_values):

    errors = []
    deg = np.arange(1, 100, 1)

    for d in deg:

        w, plan_matrix = calculate_parameters(length, d, x_values, y_values)
        plan_matrix = np.array(plan_matrix)

        sum = 0.0
        for i in range(length):
            sum += (plan_matrix[i, :] @ w - y_values[i]) ** 2
        sum /= d
        errors.append(sum)

    errors = np.array(errors)

    plt.title("Error dependence on the degree of polynomial")
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.plot(deg, errors, color='blue', linestyle='-')
    plt.show()


N = 1000
x = np.linspace(0, 1, N)

# Here we create our function. You can use your mathematical function
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

get_approximation(length=N,
                  degree=1,
                  x_values=x,
                  y_values=t
                  )
get_approximation(length=N,
                  degree=10,
                  x_values=x,
                  y_values=t
                  )
get_approximation(length=N,
                  degree=100,
                  x_values=x,
                  y_values=t
                  )

calculate_error(N, x, t)
