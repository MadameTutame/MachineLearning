from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


def split_indices(dataset_size, train_part, validation_part):
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_size, validation_size = np.int32(train_part * dataset_size), np.int32(validation_part * dataset_size)
    test_size = train_size + validation_size

    return indices[:train_size], indices[train_size:test_size], indices[test_size:]


def split_data(x_values, y_values, train_part):
    train_part = np.int32(len(y_values) * train_part)
    return x_values[:train_part], x_values[train_part:], y_values[:train_part], y_values[train_part:]


def get_one_hot_encoding(sample_p):
    t = np.zeros((1797, 10))
    for i in range(1797):
        t[i][sample_p[i]] = 1
    return t


def std_data(sample_f):
    means = []
    stds = []
    for i in range(64):
        means.append(np.mean(sample_f[:, i]))
        stds.append(np.std(sample_f[:, i]))

    for i in range(1797):
        for k in range(64):
            sample_f[i][k] = sample_f[i][k] - means[k]
            if stds[k] != 0:
                sample_f[i][k] /= stds[k]
    return sample_f


def softmax(vector):
    vector -= np.max(vector)
    vector = np.exp(vector)
    vector /= np.sum(vector)

    return vector


def softmax_matrix(matrix):
    for i in range(np.max(matrix.shape)):
        buf = softmax(matrix[i])
        matrix[i] = buf

    return matrix


def get_plan_matrix(x_values, basis_function):
    plan_matrix = np.zeros(x_values.shape)

    for i in range(64):

        if len(basis_function) > 1:
            plan_matrix[:, i] = basis_function[i](x_values[:, i])
        else:
            plan_matrix[:, i] = basis_function[0](x_values[:, i])

    return plan_matrix


def gradient(x_values, y_values, w_par, lambda_value, basis_function, bias):

    plan_matrix = get_plan_matrix(x_values, basis_function)
    Y = softmax_matrix(plan_matrix @ w_par.T + bias)

    buf = (Y - y_values).T

    grad_w = buf @ plan_matrix + lambda_value * w_par
    grad_b = buf @ np.ones(len(y_values))

    return grad_w, grad_b


def descent(learning_rate, iterations, epsilon, x_values, y_values, lambda_value, basis_function):

    w = np.random.normal(size=64, loc=0, scale=0.01)
    bias = np.random.normal(size=10, loc=0, scale=0.01)

    W = []
    for i in range(10):
        W.append(w)
    W = np.array(W)

    values = []

    for i in range(iterations):

        grad_w, grad_b = gradient(x_values, y_values, W, lambda_value, basis_function, bias)

        # Stop criteria
        if np.linalg.norm(grad_b) < epsilon and np.linalg.norm(grad_w[0]) < epsilon:
            print('break')
            break
        if np.linalg.norm(learning_rate * grad_b) < epsilon and np.linalg.norm(learning_rate * grad_w[0]) < epsilon:
            break

        W -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        values.append(softmax_matrix(x_values @ W.T + bias))

    return W, bias, np.array(values)


def display_accuracy(values, sample_p):
    size = len(values)
    x = np.arange(0, size) + 1
    value_array = np.zeros(size)

    for i in range(size):
        value_array[i] = accuracy(values[i], sample_p)
    plt.xlabel('Iterations')
    plt.ylabel('accuracy')
    plt.plot(x, value_array)
    plt.show()


def accuracy(values, sample_p):
    value = 0.0
    for j in range(len(values)):
        if sample_p[j][np.where(values[j] == np.max(values[j]))] == 1:
            value += 1
    value /= len(values)

    return value


def image_values(values, y_values, lambda_value, w):
    size = len(values)
    x = np.arange(0, size) + 1
    value_array = np.zeros(size)

    for i in range(size):
        for j in range(len(values[0])):
            for k in range(10):
                value_array[i] -= y_values[j][k] * np.log(values[i][j][k])
        value_array[i] += lambda_value * np.sum(np.sum(w ** 2)) / 2

    plt.xlabel('Iterations')
    plt.ylabel('Error values')
    plt.plot(x, value_array)
    plt.show()


# Bigger model with validation
def bigger_model(x_values, y_values, train_part, learning_rate, iterations, epsilon, lambdas, basis_functions):

    x_values = std_data(x_values)
    y_values = get_one_hot_encoding(y_values)

    train_ind, valid_ind, test_ind = split_indices(dataset_size=len(y_values),
                                                   train_part=train_part,
                                                   validation_part=(1 - train_part) / 2)

    x_train, t_train = x_values[train_ind], y_values[train_ind]
    x_valid, t_valid = x_values[valid_ind], y_values[valid_ind]
    x_test, t_test = x_values[test_ind], y_values[test_ind]

    max_accuracy = -1
    w_best, lambda_best = [], 0.0
    basis_best = []
    values_best = []
    b_best = []

    for i in range(iterations):

        lambda_cur = np.random.choice(lambdas)
        basis_cur = np.random.choice(basis_functions, 64, replace=True)

        w_cur, bias, values = descent(learning_rate=learning_rate,
                                      iterations=iterations * 10,
                                      epsilon=epsilon,
                                      x_values=x_train,
                                      y_values=t_train,
                                      lambda_value=lambda_cur,
                                      basis_function=basis_cur
                                      )

        pl = get_plan_matrix(x_valid, basis_cur)
        Y = softmax_matrix(pl @ w_cur.T + bias)
        error = accuracy(Y, t_valid)

        if max_accuracy < error:
            w_best = w_cur
            b_best = bias
            values_best = values
            lambda_best = lambda_cur
            basis_best = basis_cur
            max_accuracy = error

    test = get_plan_matrix(x_test, basis_best)
    test = softmax_matrix(test @ w_best.T + b_best)

    print('Accuracy on test:', accuracy(test, t_test))
    print('Accuracy on valid:', accuracy(softmax_matrix(get_plan_matrix(x_valid, basis_best) @ w_best.T + bias), t_valid))
    print('Accuracy on train:', accuracy(softmax_matrix(get_plan_matrix(x_train, basis_best) @ w_best.T + bias), t_train))

    print('Best lambda:', lambda_best)

    image_values(values_best, t_train, lambda_best, w_best)
    display_accuracy(values_best, t_train)

    for i in range(len(basis_functions)):
        if basis_functions[i] in basis_best:
            print(basis_functions[i].__name__, ' function was used.')


# Model without validation
def smaller_model(x_values, y_values, part, learning_rate, iterations, epsilon, lambda_value, basis_function):

    x_values = std_data(x_values)
    y_values = get_one_hot_encoding(y_values)

    x_train, x_test, t_train, t_test = split_data(x_values, y_values, part)

    w_cur, b, values = descent(learning_rate=learning_rate,
                               iterations=iterations,
                               epsilon=epsilon,
                               x_values=x_train,
                               y_values=t_train,
                               lambda_value=lambda_value,
                               basis_function=basis_function
                               )

    test = get_plan_matrix(x_test, basis_function)
    test = softmax_matrix(test @ w_cur.T + b)

    print()
    print('Accuracy on test without validation', accuracy(test, t_test))

    image_values(values, t_train, lambda_value, w_cur)
    display_accuracy(values, t_train)


digits = load_digits()

basis_functions = [np.sin, np.cos, lambda y: np.sin(6 * np.pi * y), np.sin]
lambda_set = [0, 0.0001, 0.00000000000000000001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

learning_rate = 0.001
part = 0.9
iterations = 10
epsilon = 0.001

bigger_model(x_values=digits.data,
             y_values=digits.target,
             train_part=part,
             learning_rate=learning_rate,
             iterations=iterations,
             epsilon=epsilon,
             lambdas=lambda_set,
             basis_functions=basis_functions
             )

smaller_model(x_values=digits.data,
              y_values=digits.target,
              part=part,
              learning_rate=learning_rate,
              iterations=iterations * 10,
              epsilon=epsilon,
              lambda_value=0,
              basis_function=[lambda x: x]
              )
