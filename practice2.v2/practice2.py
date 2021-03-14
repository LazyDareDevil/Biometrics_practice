import cv2
import os
import sys
import numpy as np

def face_symmetry(face_cropped):
		size = face_cropped.shape
		w = int(size[1]/10)
		if w < 3:
			w = 3
		line = w
		best_central = [w, 1000000]
		while line < size[1] - w:
			tmp = 0
			for i in range(1, w+1):
				for j in range(size[0]):
					tmp += np.abs(int(face_cropped[j][line-i]) - int(face_cropped[j][line+i]))
			if tmp < best_central[1]:
				best_central = [line, tmp]
			line += 2
		w1 = int(w/2)
		if w1 < 3:
			w1 = 3
		best_left = [w1, 1000000]
		best_right = [best_central[0] + w1, 1000000]
		line1 = w1
		line2 = best_central[0] + w1
		while line1 < best_central[0] - w1 and line2 < size[1] - w1:
			tmp1 = 0
			tmp2 = 0
			for i in range(1, w1+1):
				for j in range(int(size[0]/4), int(3*size[0]/4)):
					tmp1 += np.abs(int(face_cropped[j][line1-i]) - int(face_cropped[j][line1+i]))
					tmp2 += np.abs(int(face_cropped[j][line2-i]) - int(face_cropped[j][line2+i]))
			if tmp1 < best_left[1]:
				best_left = [line1, tmp1]
			if tmp2 < best_right[1]:
				best_right = [line2, tmp2]
			line1 += 2
			line2 += 2
		return best_central[0], best_left[0], best_right[0]

dataPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
resultPath = os.path.dirname(os.path.abspath(__file__)) + "/results"
data_faces = []
data_folder = os.path.dirname(os.path.abspath(__file__)) + "/faces/s"
for i in range(1, 41):
	for j in range(1, 11):
		image = cv2.cvtColor(cv2.imread(data_folder + str(i) + "/" + str(j) + ".pgm"), cv2.COLOR_BGR2GRAY)
		data_faces.append(image)

print("Starting computing template mathching...")
print("Otput will be in folder 'restults'")

for i in range(1, 5):
	print("Computing for template {} ...".format(i))
	if i == 1:
		print("\tComputing symmetry lines...")
	filename = "template{}.jpg".format(i)
	template = cv2.cvtColor(cv2.imread(os.path.sep.join((dataPath, filename))), cv2.COLOR_BGR2GRAY)
	typeDir = os.path.sep.join((resultPath, "template{}".format(i)))
	method = cv2.TM_SQDIFF
	count = 1
	w, h = template.shape
	for j in data_faces[1:]:
		res = cv2.matchTemplate(j, template, method)
		_, _, top_left, _ = cv2.minMaxLoc(res)
		bottom_right = (top_left[0] + h, top_left[1] + w)
		res = j.copy()
		if i == 1:
			c, l, r = face_symmetry(j[top_left[1]:top_left[1]+w, top_left[0]:top_left[0]+h])
			res = cv2.line(res, (top_left[0]+c, top_left[1]), (top_left[0]+c, top_left[1]+w), (255,255,255), 2) 
			res = cv2.line(res, (top_left[0]+l, top_left[1] + int(h/4)), (top_left[0]+l, top_left[1] + int(3*h/4)), (255,255,255), 2) 
			res = cv2.line(res, (top_left[0]+r, top_left[1] + int(h/4)), (top_left[0]+r, top_left[1] + int(3*h/4)), (255,255,255), 2)
		res = cv2.rectangle(res, top_left, bottom_right, (255, 0, 0), 2)
		filename = "{}.jpg".format(count)
		count+=1
		p = os.path.sep.join((typeDir, filename))
		cv2.imwrite(p, cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

print("Computing for Viola & Jonson (Haar cascades) and symmetry lines...")
face_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)) + '/haar/haarcascade_frontalface_default.xml')
count = 1
typeDir = os.path.sep.join((resultPath, "haar".format(i)))
for j in data_faces[1:]:
	faces = face_cascade.detectMultiScale(j, 1.3, 5)
	res = j.copy()
	for (x, y, w, h) in faces:
		borders = int(w/10)
		c, l, r = face_symmetry(j[y:y+h, x+borders:x+w-borders])
		res = cv2.rectangle(res, (x+borders, y), (x + w-borders, y + h), (255, 0, 0), 2)
		res = cv2.line(res, (x+borders+c, y), (x+borders+c, y+h), (255,255,255), 2) 
		res = cv2.line(res, (x+borders+l, y + int(h/4)), (x+borders+l, y + int(3*h/4)), (255,255,255), 2) 
		res = cv2.line(res, (x+borders+r, y + int(h/4)), (x+borders+r, y + int(3*h/4)), (255,255,255), 2)
	filename = "{}.jpg".format(count)
	count+=1
	p = os.path.sep.join((typeDir, filename))
	cv2.imwrite(p, cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))

print("Computing finished.")
