from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np


def split_data(x_values, y_values, train_part):
    train_part = np.int32(len(y_values) * train_part)
    return x_values[:train_part], x_values[train_part:], y_values[:train_part], y_values[train_part:]


def std_data(x_values):
    return (x_values - np.mean(x_values, axis=0)) / np.std(x_values, axis=0)


def get_plan_matrix(x_values, basis_function):

    plan_matrix = np.zeros((x_values[:, 0].size, 13))

    # Objects are represented by vectors of size 13
    len_columns = 13

    for i in range(len_columns):
        plan_matrix[:, i] = basis_function(x_values[:, i])

    return plan_matrix


def get_gradient(x_values, y_values, w_par, lambda_value, basis_function):
    plan_matrix = get_plan_matrix(x_values, basis_function)
    return np.dot(w_par, (np.dot(plan_matrix.T, plan_matrix) + lambda_value * np.eye(13))) - np.dot(y_values.T, plan_matrix)


def calculate_error(sample_f, sample_p, basic_function, w, lamb):

    plan_matrix = get_plan_matrix(sample_f, basic_function)
    parameter = np.dot(plan_matrix, w.T)

    return (np.sum((parameter - sample_p) ** 2) + lamb * np.dot(w, w.T)) / 2


def descent(learning_rate, iterations, epsilon, sample_f, sample_p,  lamb, basic_function):

    w = np.random.normal(size=13, loc=0, scale=0.01)

    error_value = []
    counter = 0

    for i in range(iterations):

        grad = get_gradient(sample_f, sample_p, w, lamb, basic_function)

        # Stopping criteria
        if np.linalg.norm(grad) < epsilon:
            print('break')
            break
        if np.linalg.norm(learning_rate * grad) < epsilon:
            break

        w -= learning_rate * grad
        counter += 1
        error_value.append(calculate_error(sample_f, sample_p, basic_function, w, lamb))

    return w, np.array(error_value), counter


def display_error(error_values, iterations):

    x = np.arange(0, iterations)

    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.plot(x, error_values)
    plt.show()


def main_function(x_values, y_values, train_part, learning_rate, iterations, epsilon, lambda_value, basis_function):

    x_values = std_data(x_values)
    sf_train, sf_test, sp_train, sp_test = split_data(x_values, y_values, train_part)

    w_cur, error_values, count = descent(learning_rate, iterations, epsilon, sf_train, sp_train, lambda_value, basis_function)

    display_error(error_values, count)

    train_error = calculate_error(sf_train, sp_train, basis_function, w_cur, lambda_value)
    test_error = calculate_error(sf_test, sp_test, basis_function, w_cur, lambda_value)

    print('Error value on training sample:', train_error)
    print('Error value on test sample', test_error)


data = load_boston()

features = data["data"]
prices = data.target


main_function(x_values=features,
              y_values=prices,
              train_part=0.9,
              learning_rate=0.00001,
              iterations=100000,
              epsilon=0.00001,
              lambda_value=0,
              basis_function=lambda x: np.sin(x)
              )
