from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


"""
    Decision tree model with validation for scikit-learn digits dataset
"""


def split_indices(dataset_size, train_part, validation_part):
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_size, validation_size = np.int32(train_part * dataset_size), np.int32(validation_part * dataset_size)
    test_size = train_size + validation_size

    return indices[:train_size], indices[train_size:test_size], indices[test_size:]


def entropy(y_values):
    classes = []
    for i in range(10):

        k = 0
        for j in range(len(y_values)):

            if y_values[j] == i:
                k += 1

        classes.append(k)
    classes = np.array(classes)

    H = 0.0
    for i in range(10):
        if classes[i] != 0:
            buf = classes[i] / len(y_values)
            H -= buf * np.log2(buf)

    return H


def gini(y_values):
    classes = []
    for i in range(10):

        k = 0
        for j in range(len(y_values)):

            if y_values[j] == i:
                k += 1

        classes.append(k)
    classes = np.array(classes)

    H = 1.0
    for i in range(10):
        if classes[i] != 0:
            buf = classes[i] / len(y_values)
            H -= buf ** 2

    return H


def classification_mistake(y_values):
    classes = []
    for i in range(10):

        k = 0
        for j in range(len(y_values)):

            if y_values[j] == i:
                k += 1

        classes.append(k)
    classes = np.array(classes)

    H = 1.0

    buf = np.max(classes)
    if buf != 0.0:
        H -= buf / len(y_values)
    else:
        H = 0.0

    return H


def get_information_gain(y_values, node_left_child, node_right_child, criteria):

    if criteria == 'Entropy':
        return entropy(y_values) - entropy(node_left_child) * len(node_left_child) / len(y_values) - \
               entropy(node_right_child) * len(node_right_child) / len(y_values)

    if criteria == 'Gini':
        return gini(y_values) - gini(node_left_child) * len(node_left_child) / len(y_values) - \
               gini(node_right_child) * len(node_right_child) / len(y_values)

    if criteria == 'Classification_mistake':
        return classification_mistake(y_values) - classification_mistake(node_left_child) * len(node_left_child) / len(
            y_values) - \
               classification_mistake(node_right_child) * len(node_right_child) / len(y_values)


def get_parameters(x_values, y_values, tau_values, criteria):

    best_tau = 0
    best_I = 0
    best_coord = 0

    for j in range(len(x_values[0])):

        for tau in tau_values:

            node_left_child = []
            node_right_child = []

            for i in range(len(x_values)):

                if x_values[i][j] > tau:
                    node_left_child.append(y_values[i])
                else:
                    node_right_child.append(y_values[i])

            cur_I = get_information_gain(y_values, node_left_child, node_right_child, criteria)

            if cur_I > best_I:
                best_I = cur_I
                best_tau = tau
                best_coord = j

    return best_coord, best_tau


def separate_data(sample_f, sample_p, params):

    k, tau = params

    node_left_f = []
    node_left_p = []

    node_right_f = []
    node_right_p = []

    for i in range(len(sample_f)):

        if sample_f[i][k] > tau:
            node_left_f.append(sample_f[i])
            node_left_p.append(sample_p[i])
        else:
            node_right_f.append(sample_f[i])
            node_right_p.append(sample_p[i])

    return np.array(node_left_f), np.array(node_left_p), np.array(node_right_f), \
           np.array(node_right_p)


def create_separating_node(params, depth):
    node = {'ind': params[0], 'tau': params[1], 'isterminal': False, 'left': None, 'right': None, 'depth': depth}
    return node


def create_terminal_node(sample_f, sample_p, depth):
    node = {'isterminal': True, 'sample_f': sample_f, 'sample_p': sample_p, 'depth': depth, 't': probability_vector(sample_p)}
    return node


def create_tree(x_values, y_values, node_size, epsilon, depth, level, tau_values, criteria):

    k = depth
    k += 1

    if not stop_criteria(y_values, node_size, epsilon, k, level):

        params = get_parameters(x_values, y_values, tau_values, criteria)
        left_sample_f, left_sample_p, right_sample_f, right_sample_p = separate_data(x_values, y_values, params)
        node = create_separating_node(params, k)
        node['left'] = create_tree(left_sample_f, left_sample_p, node_size, epsilon, k, level, tau_values, criteria)
        node['right'] = create_tree(right_sample_f, right_sample_p, node_size, epsilon, k, level, tau_values, criteria)

    else:
        node = create_terminal_node(x_values, y_values, k)

    return node


def stop_criteria(y_values, node_size, epsilon, depth, level):
    if len(y_values) <= node_size or entropy(y_values) < epsilon or depth > level:
        return True
    return False


def print_tree(node):

    if node['isterminal']:
        print(node['sample_p'].shape)
    else:
        print_tree(node['left'])
        print_tree(node['right'])


def probability_vector(sample_p):

    if len(sample_p) == 0:
        return np.zeros(10)

    vector = []
    for i in range(10):

        k = 0
        for j in range(len(sample_p)):

            if sample_p[j] == i:
                k += 1

        vector.append(k / len(sample_p))
    vector = np.array(vector)

    return vector


def test_tree(node, vector_f, pos, conf_matrix, to_hist_list):
    if not node['isterminal']:

        if vector_f[node['ind']] > node['tau']:
            test_tree(node['left'], vector_f, pos, conf_matrix, to_hist_list)

        else:
            test_tree(node['right'], vector_f, pos, conf_matrix, to_hist_list)

    else:

        arr = np.array(node['t'])

        k = max_ind(node['t'])
        conf_matrix[k][pos] += 1

        to_hist_list.append([pos, arr[k], k])

    return conf_matrix, to_hist_list


def get_confusion_matrix_from_test_or_valid(node, x_value, y_value, conf_matrix, hist_list):
    for i in range(len(y_value)):
        conf_matrix, hist_list = test_tree(node, x_value[i], y_value[i], conf_matrix, hist_list)

    return conf_matrix, hist_list


def get_conf_matrix_from_train(node, conf_matrix):
    if not node['isterminal']:

        get_conf_matrix_from_train(node['left'], conf_matrix)
        get_conf_matrix_from_train(node['right'], conf_matrix)

    else:

        k = max_ind(node['t'])
        for val in node['sample_p']:
            conf_matrix[k][val] += 1

    return conf_matrix


def max_ind(node):
    array = np.array(node)
    max = np.max(array)
    k = 0
    for i in range(array.size):
        if array[i] == max:
            k = i

    return k


def get_accuracy(matrix):

    s = 0.0
    for i in range(len(matrix)):
        s += matrix[i][i]

    s /= np.sum(matrix)

    return s


def display_hist(to_hist_list):
    right = []
    no_right = []

    for i in range(len(to_hist_list)):

        if to_hist_list[i][0] == to_hist_list[i][2]:
            right.append(to_hist_list[i][1])
        else:
            no_right.append(to_hist_list[i][1])

    right = np.array(right)
    no_right = np.array(no_right)

    plt.hist(right, 10)
    plt.title('Correctly identified')
    plt.show()

    plt.hist(no_right, 10)
    plt.title('Incorrectly identified')
    plt.show()


def main_function(iterations, x_values, y_values, train_part, tau_values, levels, criteria, epsilons, node_sizes):

    train_ind, valid_ind, test_ind = split_indices(dataset_size=len(y_values),
                                                   train_part=train_part,
                                                   validation_part=(1 - train_part) / 2)

    x_train, t_train = x_values[train_ind], y_values[train_ind]
    x_valid, t_valid = x_values[valid_ind], y_values[valid_ind]
    x_test, t_test = x_values[test_ind], y_values[test_ind]

    max_error = -1

    tau_values = np.random.choice(tau_values, np.random.randint(8, 16), replace=False)
    tau_values = np.sort(tau_values)

    best_criteria = ''
    best_level = 0
    best_criteria_val = 0.0
    best_size = 0
    best_node = None

    for i in range(iterations):

        criteria_cur = np.random.choice(criteria)  # Criteria
        criteria_val_cur = np.random.choice(epsilons)  # Criteria value
        size_cur = np.random.choice(node_sizes)  # Node size
        level_cur = np.random.choice(levels)  # Tree's depth
        to_hist_list_valid = []  # list of probabilities from terminal nodes
        conf_mtrx_valid = np.zeros((10, 10)).astype(int)  # Confusion matrix for validation sample

        node = create_tree(x_values=x_train,
                           y_values=t_train,
                           node_size=size_cur,
                           epsilon=criteria_val_cur,
                           depth=0,
                           level=level_cur,
                           tau_values=tau_values,
                           criteria=criteria_cur
                           )

        conf_mtrx_valid, to_hist_list_valid = get_confusion_matrix_from_test_or_valid(node=node,
                                                                                      x_value=x_valid,
                                                                                      y_value=t_valid,
                                                                                      conf_matrix=conf_mtrx_valid,
                                                                                      hist_list=to_hist_list_valid
                                                                                      )
        accuracy = get_accuracy(conf_mtrx_valid)

        if max_error < accuracy:
            max_error = accuracy
            best_criteria = criteria_cur
            best_level = level_cur
            best_criteria_val = criteria_val_cur
            best_size = size_cur
            best_node = node

    conf_mtrx_test = np.zeros((10, 10)).astype(int)
    conf_mtrx_train = np.zeros((10, 10)).astype(int)
    conf_mtrx_valid = np.zeros((10, 10)).astype(int)

    to_hist_list_test = []
    to_hist_list_valid = []

    conf_mtrx_train = get_conf_matrix_from_train(node=best_node, conf_matrix=conf_mtrx_train)

    conf_mtrx_valid, to_hist_list_valid = get_confusion_matrix_from_test_or_valid(node=best_node,
                                                                                  x_value=x_valid,
                                                                                  y_value=t_valid,
                                                                                  conf_matrix=conf_mtrx_valid,
                                                                                  hist_list=to_hist_list_valid)

    conf_mtrx_test, to_hist_list_test = get_confusion_matrix_from_test_or_valid(node=best_node,
                                                                                x_value=x_test,
                                                                                y_value=t_test,
                                                                                conf_matrix=conf_mtrx_test,
                                                                                hist_list=to_hist_list_test
                                                                                )
    train_acc = get_accuracy(conf_mtrx_train)
    valid_acc = get_accuracy(conf_mtrx_valid)
    test_acc = get_accuracy(conf_mtrx_test)

    display_hist(to_hist_list_test)

    print('Confusion matrix for training sample:\n', conf_mtrx_train)
    print('Confusion matrix for validation sample:\n', conf_mtrx_valid)
    print('Confusion matrix for test sample:\n', conf_mtrx_test)

    print('Accuracy on test sample:', train_acc)
    print('Accuracy on validation sample:', valid_acc)
    print('Accuracy on test sample:', test_acc)

    print('Best criteria:', best_criteria)
    print('Best minimal criteria val:', best_criteria_val)
    print('Best maximum depth:', best_level)
    print('Best node size:', best_size)

    return


digits = load_digits()
features = digits.data
targets = digits.target

part = 0.7
taus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
levels = [2, 4, 6, 7, 10]
criteria = ['Entropy', 'Gini', 'Classification_mistake']
epsilons = [0.001, 0.0001, 0.01, 0.1]
sizes = [2, 4, 8, 16, 32]

main_function(iterations=5,
              x_values=features,
              y_values=targets,
              train_part=part,
              tau_values=taus,
              levels=levels,
              criteria=criteria,
              epsilons=epsilons,
              node_sizes=sizes
              )
