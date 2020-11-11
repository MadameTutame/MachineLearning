import matplotlib.pyplot as plt
import numpy as np


# Shuffle indices
def split_on_samples(dataset_size, train_part, validation_part):
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_size, validation_size = np.int32(train_part * dataset_size), np.int32(validation_part * dataset_size)
    test_size = train_size + validation_size

    return indices[:train_size], indices[train_size:test_size], indices[test_size:]


# Plan matrix and weights calculation
def calculate_parameters(x_values, y_values, sample_length, basis_functions, how_many_functions, lambda_value, w):
    plan_matrix = np.zeros((sample_length, how_many_functions))

    for j in range(how_many_functions):
        plan_matrix[:, j] = basis_functions[j](x_values)

    if lambda_value is not None and w is None:
        t_plan_matrix = plan_matrix.T
        parameter = np.linalg.inv(
            t_plan_matrix @ plan_matrix + lambda_value * np.eye(how_many_functions)) @ t_plan_matrix @ y_values

    if lambda_value is None and w is not None:
        parameter = plan_matrix @ w.T

    return parameter


def calculate_error(x_values, y_values, sample_length, basis_functions, how_many_functions, lambda_value, w):
    parameters = calculate_parameters(x_values, y_values, sample_length, basis_functions, how_many_functions, None, w)

    return (np.sum((parameters - y_values) ** 2) + lambda_value * (w @ w.T)) / 2


def display(x_values, y_values, sample_length, basis_functions, how_many_functions, w, lambda_value):
    X = calculate_parameters(x_values, y_values, sample_length, basis_functions, how_many_functions, None, w)

    plt.plot(x, t, color='violet', marker='o', markersize='0.7', linestyle=' ', label="Error values")
    plt.plot(x, z, color='red', linestyle='-', label="z(x)")
    plt.plot(x_values, X, color='blue', linestyle='-', label="Lambda = " + str(lambda_value))
    plt.xlabel("x values")
    plt.ylabel("Error values")
    plt.legend(loc="lower right")
    plt.show()


# main function
def main(iterations, x_values, y_values, basis_functions, how_many_functions, lambdas, dataset_size, train_part):

    train_ind, valid_ind, test_ind = split_on_samples(dataset_size, train_part, (1 - train_part) / 2)

    x_train, t_train = np.sort(x_values[train_ind]), np.sort(y_values[train_ind])
    x_valid, t_valid = np.sort(x_values[valid_ind]), np.sort(y_values[valid_ind])
    x_test, t_test = np.sort(x_values[test_ind]), np.sort(y_values[test_ind])
    len_train, len_valid, len_test = len(x_train), len(x_valid), len(x_test)

    minimal_error = 10 ** 10
    best_w, best_lambda = 0.0, 0.0
    best_basis_func = []

    for i in range(iterations):

        lambda_cur = np.random.choice(lambdas)
        phi_cur = np.random.choice(basis_functions, how_many_functions, replace=False)

        # Calculating weights for train sample and error value for validation sample
        w_cur = calculate_parameters(x_train, t_train, len_train, phi_cur, how_many_functions, lambda_cur, None)
        error_cur = calculate_error(x_valid, t_valid, len_valid, phi_cur, how_many_functions, lambda_cur, w_cur)

        # Validation
        if minimal_error > error_cur:
            best_w = w_cur
            best_lambda = lambda_cur
            best_basis_func = phi_cur
            minimal_error = error_cur

    display(x_test, t_test, len_test, best_basis_func, how_many_functions, best_w, best_lambda)

    return best_w, best_lambda, best_basis_func, x_test, t_test, len_test


def generate_data(size):
    x = np.linspace(0, 1, size)
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(size)
    t = z + error

    return x, z, t


# Size of dataset
size = 5000

# Setting a lists of functions and lambdas
basis_functions = [np.sin, np.cos, np.exp, np.sqrt, lambda y: np.sin(6 * np.pi * y)]
lambda_values = [0, 0.0001, 0.00000000000000000001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

x, z, t = generate_data(size)

train_part = 0.8

# Randomly select the number of basis functions
how_many_functions = np.random.randint(1, np.size(basis_functions) + 1)
w_best, lambda_best, phi_best, x_test, t_test, len_test = main(iterations=1000,
                                                               x_values=x,
                                                               y_values=t,
                                                               basis_functions=basis_functions,
                                                               how_many_functions=how_many_functions,
                                                               lambdas=lambda_values,
                                                               dataset_size=size,
                                                               train_part=train_part)

print('Number of functions:', how_many_functions)
for i in range(len(basis_functions)):
    if basis_functions[i] in phi_best:
        print(basis_functions[i].__name__, ' function was used.')
print('Error on the training sample:',
      calculate_error(x_test, t_test, len_test, phi_best, how_many_functions, lambda_best, w_best))
print('Best lambda:', lambda_best)
print('Best weights:', w_best)
