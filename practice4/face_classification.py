import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
import os
from scipy.fftpack import dct
from numpy import random
import cv2


def get_histogram(image, param = 30):
    hist, bins = np.histogram(image, bins=np.linspace(0, 1, param))
    return [hist, bins]

def get_dft(image, mat_side = 13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)

def get_dct(image, mat_side = 13):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c

def get_gradient(image, n = 2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result

def get_scale(image, scale = 0.35):
	h = image.shape[0]
	w = image.shape[1]
	new_size = (int(h * scale), int(w * scale))
	return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def get_faces():
	data_images = fetch_olivetti_faces()
	return [data_images.images, data_images.target]

def read_faces_from_disk():
	data_faces = []
	data_target = []
	data_folder = os.path.dirname(os.path.abspath(__file__)) + "/faces/s"
	for i in range(1, 41):
		for j in range(1, 11):
			image = cv2.cvtColor(cv2.imread(data_folder + str(i) + "/" + str(j) + ".pgm"), cv2.COLOR_BGR2GRAY)
			data_faces.append(image/255)
			data_target.append(i-1)
	return [data_faces, data_target]

def split_data(data, images_per_person_in_train=5):
	images_per_person = 10
	images_all = len(data[0])
	if images_per_person_in_train > 9:
		images_per_person_in_train = 9
	if images_per_person_in_train < 1:
		images_per_person_in_train = 1

	x_train, x_test, y_train, y_test = [], [], [], []

	for i in range(0, images_all, images_per_person):
		x_train.extend(data[0][i: i+images_per_person_in_train])
		y_train.extend(data[1][i: i+images_per_person_in_train])

		x_test.extend(data[0][i+images_per_person_in_train: i+images_per_person])
		y_test.extend(data[1][i+images_per_person_in_train: i+images_per_person])
	
	return x_train, x_test, y_train, y_test

def mesh_data(data):
	indexes = rnd.sample(range(0, len(data[0])), len(data[0]))
	return [data[0][index] for index in indexes], [data[1][index] for index in indexes]

def create_feature(data, method, parameter):
	result = []
	for element in data:
		if method == get_histogram:
			result.append(method(element, parameter)[0])
		else:
			result.append(method(element, parameter))
	return result

def distance(el1, el2):
	return np.linalg.norm(np.array(el1) - np.array(el2))

def classifier(data, new_elements, method, parameter):
	if method not in [get_histogram, get_dft, get_dct, get_gradient, get_scale]:
		return []
	featured_data = create_feature(data[0], method, parameter)
	featured_elements = create_feature(new_elements, method, parameter)
	result = []
	for element in featured_elements:
		min_el = [1000, -1]
		for i in range(len(featured_data)):
			dist = distance(element, featured_data[i])
			if dist < min_el[0]:
				min_el = [dist, i]
		if min_el[1] < 0:
			result.append(0)
		else:
			result.append(data[1][min_el[1]])
	return result

def test_classifier(data, test_elements, method, parameter):
	if method not in [get_histogram, get_dft, get_dct, get_gradient, get_scale]:
		return []
	answers = classifier(data, test_elements[0], method, parameter)
	correct_answers = 0
	for i in range(len(test_elements[1])):
		if answers[i] == test_elements[1][i]:
			correct_answers += 1
	return correct_answers/len(test_elements[1])

def teach_parameter(data, test_elements, method):
	if method not in [get_histogram, get_dft, get_dct, get_gradient, get_scale]:
		return []
	image_size = min(data[0][0].shape)
	param = (0, 0, 0)
	if method == get_histogram:
		param = (10, 300, 3)
	if method == get_dft or method == get_dct:
		param = (2, image_size, 1)
	if method == get_gradient:
		param = (2, int(data[0][0].shape[0]/2 - 1), 1)
	if method == get_scale:
		param = (0.05, 1, 0.05)
	
	best_param = param[0]
	classf = test_classifier(data, test_elements, method, best_param)
	stat = [[best_param], [classf]]

	for i in np.arange(param[0] + param[2], param[1], param[2]):
		new_classf = test_classifier(data, test_elements, method, i)
		stat[0].append(i)
		stat[1].append(new_classf)
		if new_classf > classf:
			classf = new_classf
			best_param = i
	
	return [best_param, classf], stat

def voting(data, new_elements, parameters):
	methods = [get_histogram, get_dft, get_dct, get_gradient, get_scale]
	res = {}
	for method in methods:
		res[method.__name__] = classifier(data, new_elements, method, parameters[method.__name__])
	tmp = []
	for i in range(len(new_elements)):
		temp = {}
		for method in res:
			t = res[method][i]
			if t in temp:
				temp[t] += 1
			else:
				temp[t] = 1
		best_size = sorted(temp.items(), key=lambda item: item[1], reverse=True)[0]
		tmp.append(best_size[0])
	return tmp

def test_voting(train, test, parameters):
	res = voting(train, test[0], parameters)
	sum = 0
	for i in range(len(test[0])):
		if test[1][i] == res[i]:
			sum += 1
	return sum/len(test[0])

def cross_validation(data):
	methods = [get_histogram, get_dft, get_dct, get_gradient, get_scale]
	res = []
	start = 5
	end = 8
	for size in range(start, end):
		print(str(size) + " face from class")
		X_train, X_test, y_train, y_test = split_data(data, size)
		train = mesh_data([X_train, y_train])
		test = mesh_data([X_test, y_test])
		parameters = {}
		for method in methods:
			print(method.__name__)
			parameters[method.__name__] = teach_parameter(train, test, method)[0][0]
		print(parameters)
		classf = test_voting(train, test, parameters)
		print(classf)
		res.append([parameters, classf])
		
	best_res = [[], 0]
	best = 0
	for i in range(start, end):
		if res[i-start][1] > best:
			best = res[i-start][1]
			best_res[0] = res[i-start][0]
			best_res[1]= i
	best_res.append(best)
	return best_res

def vote_classifier(data):
	parameters, train_size = cross_validation(data)
	x_train, y_train, x_test, _ = split_data(data, train_size)
	train = mesh_data([x_train, y_train])
	return voting(train, x_test, parameters)
		
# data = read_faces_from_disk()
# parameters, train_size = cross_validation(data)
# print(parameters)
# print(train_size)
# x_train, x_test, y_train, y_test = split_data(data, train_size)
# train = mesh_data([x_train, y_train])
# v = voting(train, x_test, parameters)
# print(y_test[:10])
# print(v[:10])