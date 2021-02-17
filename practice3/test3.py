import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
import os
from face_validation import get_dct, get_dft, get_gradient, get_scale, get_histogram

def get_faces():
	data_images = fetch_olivetti_faces()
	images = data_images.images
	targets = data_images.target
	indexes = rnd.sample(range(0, len(images)), len(images))
	return [images[index] for index in indexes], [targets[index] for index in indexes]

def read_faces_from_disk():
	data_faces = []
	data_target = []
	data_folder = os.path.dirname(os.path.abspath(__file__)) + "/faces/s"
	for i in range(1, 41):
		for j in range(1, 11):
			data_faces.append(cv2.cvtColor(cv2.imread(data_folder + i + "/" + j + ".pgm"), cv2.COLOR_BGR2GRAY)/255)
			data_target.append(i)
	indexes = rnd.sample(range(0, len(data_faces)), len(data_faces))
	return [data_faces[index] for index in indexes], [data_target[index] for index in indexes]

def mesh_data(features, targets):
	indexes = rnd.sample(range(0, len(features)), len(features))
	return [features[index] for index in indexes], [targets[index] for index in indexes]

def split_data(data_faces, data_target, images_per_person = 10, images_per_person_in_train=7, images_per_person_in_test=3):
	images_all = len(data_faces)
	if images_per_person_in_train > 9:
		images_per_person_in_train = 9
	if images_per_person_in_test > 10 - images_per_person_in_train:
		images_per_person_in_test = 10 - images_per_person_in_train
	
	x_train, x_test, y_train, y_test, x_free, y_free = [], [], [], [], [], []

	for i in range(0, images_all, images_per_person):
		indices = list(range(i, i + images_per_person))
		indices_train = rnd.sample(indices, images_per_person_in_train)
		x_train.extend(data_faces[index] for index in indices_train)
		y_train.extend(data_target[index] for index in indices_train)

		indices_test = rnd.sample(set(indices) - set(indices_train), images_per_person_in_test)
		x_test.extend(data_faces[index] for index in indices_test)
		y_test.extend(data_target[index] for index in indices_test)

		indices_free = set(indices) - set(indices_train) - set(indices_test)
		if len(indices_free > 0):
			x_free.extend(data_faces[index] for index in indices_free)
			y_free.extend(data_target[index] for index in indices_free)

	return x_train, x_test, y_train, y_test, x_free, y_free

def create_feature(data, method, parameter):
	result = []
	for element in data:
		result.append(method(element, parameter))
	return result

def distance(el1, el2):
	return np.linalg.norm(el1 - el2)

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
		param = (2, int(image_size/2 - 1), 1)
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

def cross_validation(data, method, folds=3):
	if folds < 3:
		folds = 3
	per_fold = int(len(data[0])/folds)
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	for step in range(0, folds):
		if step == 0:
			x_train = data[0][per_fold:]
			x_test = data[0][:per_fold]
			y_train = data[1][per_fold:]
			y_test = data[1][:per_fold]
		else:
			if step == folds - 1:
				x_train = data[0][:step*per_fold]
				x_test = data[0][step*per_fold:]
				y_train = data[1][:step*per_fold]
				y_test = data[1][step*per_fold:]
			else:
				x_train = data[0][(step-1)*per_fold:step*per_fold].append(data[0][(step+1)*per_fold:])
				x_test = data[0][step*per_fold:(step+1)*per_fold]
				y_train = data[1][(step-1)*per_fold:step*per_fold].append(data[1][(step+1)*per_fold:])
				y_test = data[1][step*per_fold:(step+1)*per_fold]
		

images, targets = get_faces()
x_train, x_test, y_train, y_test = split_data(images, targets)
print(teach_parameter([x_train, y_train], [x_test, y_test], get_scale)[0])

