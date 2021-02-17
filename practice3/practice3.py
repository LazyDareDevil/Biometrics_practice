import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
from face_validation import get_dct, get_dft, get_gradient, get_scale, get_histogram
from collections import Counter


def get_trains_tests():
    data_images = fetch_olivetti_faces()
    data_faces = data_images.images
    data_target = data_images.target

    images_all = 400
    images_per_person = 10
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, images_all, images_per_person):
        num_of_samples = 8
        indices = list(range(i, i + num_of_samples))

        indices = rnd.sample(indices, num_of_samples)
        x_train.extend(data_faces[index] for index in indices)
        y_train.extend(data_target[index] for index in indices)

        remaining_indices = set(range(i, i + images_per_person)) - set(indices)
        x_test.extend(data_faces[index] for index in remaining_indices)
        y_test.extend(data_target[index] for index in remaining_indices)

    return x_train, x_test, y_train, y_test

def predict_method(method, new_face, train, points=None, param=None):
    meth_name = method.__name__[4:]
    hist = method(new_face, param)
    dists = [np.linalg.norm(hist - res, ord=2) for res, _, _ in train]
    win = dists.index(min(dists))
    return train[win][1], train[win][2], train[win][0]

def test_method(method, arg_x_tr, arg_x, arg_y_tr, arg_y):
    meth_name = method.__name__[4:]
    print(meth_name)

    train_res = []
    for image, class_face in zip(arg_x_tr, arg_y_tr):
		train_res.append((method(image), class_face, image))
    
    right, counter = 0, 0
    fig = plt.figure(num=meth_name)
    x_acc, y_acc = [], []
    for image, answer in zip(arg_x, arg_y):
		test_res = method(image)
		res, img_closest, method_res = predict_method(method, image, train_res)
        is_chart = meth_name != 'random_points' and len(test_res.shape) == 1  # if chart -> len 1, if image -> len 2
        if res == answer:
            right += 1
            fig.patch.set_facecolor('xkcd:light green')
        else:
            fig.patch.set_facecolor('xkcd:salmon')

        plt.title("Closest images")

        ax2 = plt.subplot(421)
        ax2.imshow(img_closest, cmap='gray')
        ax2.set_title(f'test ({answer})')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax4 = plt.subplot(423)

        ax4.plot(test_res) if is_chart else ax4.imshow(test_res, cmap='gray')

        ax3 = plt.subplot(422)
        ax3.imshow(image, cmap='gray')
        ax3.set_title(f'closest ({res})')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax5 = plt.subplot(424)
        ax5.plot(method_res) if is_chart else ax5.imshow(method_res, cmap='gray')

        counter += 1
        x_acc.append(counter)
        curr_accuracy = right / counter
        y_acc.append(curr_accuracy)
        ax1 = plt.subplot(414)
        ax1.plot(x_acc, y_acc)
        ax1.set_title(f'Current {meth_name} accuracy: {curr_accuracy:.2f}')

        plt.draw()
        plt.pause(0.01)
        plt.clf()

    accuracy = right / len(arg_x)
    print(accuracy)
    plt.close(fig)


def vote(image, trains):
    results = []
    for method, train in zip(methods, trains):
        meth_name = method.__name__[4:]
        res, img_closest, method_res = predict_method(method, image, train_res)
        results.append(res)
    print(results)
    win = Counter(results).most_common()[0][0]
    return win

def test_vote(trained, arg_x, arg_y):
    fig = plt.figure(num='VOTE')
    x_acc, y_acc = [], []
    counter = 0
    for image, answer in zip(arg_x, arg_y):
        res = vote(image, trained)
        right = 0
        if res == answer:
            right += 1
            fig.patch.set_facecolor('xkcd:light green')
        else:
            fig.patch.set_facecolor('xkcd:salmon')
        counter += 1
        accuracy = right / counter
        x_acc.append(counter)
        y_acc.append(accuracy)
        plt.plot(x_acc, y_acc)
        plt.draw()

def dif_params(method, start_param, stop_param, step_param, arg_x_tr, arg_x, arg_y_tr, arg_y):
    meth_name = method.__name__[4:]
    print(meth_name)
    fig = plt.figure(num=meth_name)
    x_acc, y_acc = [], []

    for param in np.arange(start_param, stop_param, step_param):
        train_res = []
        for image, class_face in zip(arg_x_tr, arg_y_tr):
            train_res.append((method(image, param), class_face, image))

        right = 0
        for image, answer in zip(arg_x, arg_y):
            res, img_closest, method_res = predict_method(method, image, train_res, param=param)
            if res == answer:
                right += 1
        accuracy = right / len(arg_x)
        print(accuracy, right, len(arg_x))
        x_acc.append(param)
        y_acc.append(accuracy)

    plt.plot(x_acc, y_acc, '.-')
    plt.xlabel('Parameter')
    plt.ylabel('Accuracy')

    plt.show()

def k_fold_get_indices(k=3):
    images_all = 400
    images_per_person = 10
    train_ind_all, test_ind_all = [], []
    num_test = int(images_per_person / k)
    num_train = images_per_person - num_test

    chunks = [[] for _ in range(k)]
    for i in range(0, images_all, images_per_person):
        indices = list(range(i, i + images_per_person))
        rnd.shuffle(indices)
        res = np.array_split(np.array(indices), k)
        for j, chunk in enumerate(res):
            chunks[j].extend(sorted(chunk))

    for i in range(k):
        cur_test = i
        test_indices = chunks[cur_test]
        train_indices = []
        for j in range(i):
            train_indices.extend(chunks[j])
        for j in range(i+1, k):
            train_indices.extend(chunks[j])

        train_ind_all.append(train_indices)
        test_ind_all.append(test_indices)
    return test_ind_all, train_ind_all


def k_fold_get_data(k=3):
    test_indices, train_indices = k_fold_get_indices(k)
    data_images = fetch_olivetti_faces()
    data_faces = data_images.images
    data_target = data_images.target

    x_train, y_train, x_test, y_test = [[] for _ in range(k)], [[] for _ in range(k)],\
                                       [[] for _ in range(k)], [[] for _ in range(k)]

    for i in range(k):
        indices = train_indices[i]
        x_train[i].extend(data_faces[index] for index in indices)
        y_train[i].extend(data_target[index] for index in indices)

        remaining_indices = test_indices[i]
        x_test[i].extend(data_faces[index] for index in remaining_indices)
        y_test[i].extend(data_target[index] for index in remaining_indices)

    return x_train, y_train, x_test, y_test


def k_fold(k = 3):
    # scale: 0.05, 1.0, 0.05
    # hist: 5, 60, 5
    # dft: 2, 20, 1
    # dct: 2, 20, 1
    # grad: 1, 10, 1
    x_train_list, y_train_list, x_test_list, y_test_list = k_fold_get_data(k)
    methods = [get_scale, get_gradient, get_histogram, get_dct, get_dft]
    method = get_gradient

    meth_name = method.__name__[4:]
    fig = plt.figure(num=meth_name)

    start_param, stop_param, step_param = 5, 170, 5
    x_graph, y_graph = [], []

    for param in np.arange(start_param, stop_param, step_param):
        num_points = param
        points = np.array([random.randint(0, 64, (1, 2)) for _ in range(num_points)])

        accuracies = []
        for i in range(k):
            train_res = []
            for image, class_face in zip(x_train_list[i], y_train_list[i]):
                train_res.append((method(image, points), class_face, image))
            right = 0
            for image, answer in zip(x_test_list[i], y_test_list[i]):
                res, img_closest, method_res = predict_method(method, image, train_res, points=points)
                if res == answer:
                    right += 1
            accuracy = right / len(x_test_list[i])
            accuracies.append(accuracy)

        x_graph.append(param)
        y_graph.append(np.mean(accuracies))
        print(param, np.mean(accuracies))
        plt.clf()
        plt.plot(x_graph, y_graph, '.-')

        plt.draw()
        plt.pause(0.01)
    plt.show()



if __name__ == '__main__':
    k_fold(3)
# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.