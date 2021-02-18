from PIL import Image, ImageTk
import tkinter as tki
from tkinter import filedialog
import threading
import datetime
import time
import imutils
import cv2
import os
import sys
from face_classification import *
from tkinter import messagebox as mb
import io

class PhotoBoothApp:
	def __init__(self):
		self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
		self.thread = None
		self.stopEvent = None
		self.root = tki.Tk()
		self.canvas = tki.Canvas(self.root)
		self.root.title("Paractice 3. Face classificator")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.root.quit())
		self.root.geometry("1350x700+50+50")
		self.root.resizable(False, False)
		self.canvas.grid(row=0, column=0)
		
		self.frame_settings = tki.Frame(self.canvas, bg='peach puff', height=100, width=1350)
		self.frame_settings.grid(row=0)
		self.frame_example = tki.Frame(self.canvas, height=200, width=1350)
		self.frame_example.grid(row=1, sticky = 'N')
		self.frame_stats = tki.Frame(self.canvas, width=1350)
		self.frame_stats.grid(row=2)
		
		lbl1 = tki.Label(self.frame_settings, text="Step 1. Choose dataset")
		lbl1.grid(row=0, column=0, padx = 20, pady = 10)

		self.list = tki.Listbox(self.frame_settings, selectmode = "single", width=30, height=2)
		self.list.grid(row=1, column=0, sticky='N', padx = 20, pady=8)
		m = ["The ORL face database (64x64)", "The ORL face database (112x92)"]
		for each_item in m: 
			self.list.insert("end", each_item)  

		btn1 = tki.Button(self.frame_settings, text="Load data", command=self.load_data)
		btn1.grid(row=2, column=0, sticky='N', padx = 20, pady = 8)

		self.label_data = tki.Label(self.frame_settings)
		self.label_data.grid(row=3, column=0, sticky='N', padx = 20, pady = 2)

		lbl3 = tki.Label(self.frame_settings, text="Step 2. Choose number faces in test")
		lbl3.grid(row=0, column=2, padx = 25, pady = 10)

		l1 = tki.Label(self.frame_settings, text="Low range (>=10)")
		l1.grid(row=1, column=2, padx = 25, pady = 8, sticky='NW')
		self.e1 = tki.Entry(self.frame_settings, width=10)
		self.e1.grid(row=1, column=2, padx = 25, pady = 8, sticky='NE')
		l2 = tki.Label(self.frame_settings, text="High range (<=200)")
		l2.grid(row=1, column=2, padx = 25, pady = 40, sticky='NW')
		self.e2 = tki.Entry(self.frame_settings, width=10)
		self.e2.grid(row=1, column=2, padx = 25, pady = 40, sticky='NE')

		lbl4 = tki.Label(self.frame_settings, text="Step 3. Choose method")
		lbl4.grid(row=0, column=4, padx = 25, pady = 10)

		self.list1 = tki.Listbox(self.frame_settings, selectmode = "single", width=20, height=5)
		self.list1.grid(row=1, column=4, sticky='N', padx = 25, pady=8)
		self.methods = ["histogram", "dft", "dct", "scale", "gradient"]
		for each_item in self.methods:
			self.list1.insert("end", each_item)

		btn4 = tki.Button(self.frame_settings, text="Confirm", command=self.set_feature)
		btn4.grid(row=2, column=4, sticky='N', padx = 25, pady = 8)

		self.label_method = tki.Label(self.frame_settings, text="")
		self.label_method.grid(row=3, column=4, padx = 25, pady = 2)

		lbl2 = tki.Label(self.frame_settings, text="Step 4.")
		lbl2.grid(row=0, column=5, padx = 25, pady = 10)

		l3 = tki.Label(self.frame_settings, text="Input parameter for feature")
		l3.grid(row=1, column=5, padx = 25, pady = 8, sticky='NW')
		self.e3 = tki.Entry(self.frame_settings, width=10)
		self.e3.grid(row=1, column=5, padx = 25, pady = 30, sticky='NE')

		btn2 = tki.Button(self.frame_settings, text="Show feature examples", command=self.feature_example)
		btn2.grid(row=2, column=5, sticky='N', padx = 25, pady = 8)

		lbl5 = tki.Label(self.frame_settings, text="Step 5.")
		lbl5.grid(row=0, column=6, padx = 25, pady = 10)

		lbl8 = tki.Label(self.frame_settings, text="Test will be calculated by 3, 5, 7 folds.\n Best result will be chosen")
		lbl8.grid(row=1, column=6, padx = 25, pady = 8, sticky='N')

		btn3 = tki.Button(self.frame_settings, text="Start computing", command=self.start_computing)
		btn3.grid(row=2, column=6, sticky='N', padx = 25, pady = 8)

		lbl9 = tki.Label(self.frame_settings, text="Best results for train")
		lbl9.grid(row=0, column=7, padx = 25, pady = 8, sticky='N')

		lbl10 = tki.Label(self.frame_settings, text="Best parameter")
		lbl10.grid(row=1, column=7, padx = 25, pady = 8, sticky='N')
		self.label_parameter = tki.Label(self.frame_settings, text="")
		self.label_parameter.grid(row=1, column=8, pady = 8, sticky='NW')

		lbl11 = tki.Label(self.frame_settings, text="Best score")
		lbl11.grid(row=2, column=7, padx = 25, sticky='N')
		self.label_score = tki.Label(self.frame_settings, text="")
		self.label_score.grid(row=2, column=8, sticky='NW')

		lbl12 = tki.Label(self.frame_settings, text="Best num folds")
		lbl12.grid(row=3, column=7, padx = 25, sticky='N')
		self.label_folds = tki.Label(self.frame_settings, text="")
		self.label_folds.grid(row=3, column=8, sticky='NW')

		lbl6 = tki.Label(self.frame_example, text="Feature example:")
		lbl6.grid(row=0, column=0, padx = 10, sticky='NW')	

		lbl7 = tki.Label(self.frame_stats, text="Test data computing stats:")
		lbl7.grid(row=0, column=0, padx = 10, sticky='NW')

		self.data = None
		self.number_face_test = [10, 200]
		self.examples = [[], []]
		self.parameter = 0
		self.method = None
		
		self.images = [tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example)]
		for i in range(len(self.images)):
			self.images[i].grid(row = 0, column = i+1, sticky='N', padx=10, pady=2)
		self.features = [tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example)]
		for i in range(len(self.images)):
			self.features[i].grid(row = 1, column = i+1, sticky='N', padx=10, pady=2)

		lll1 = tki.Label(self.frame_stats, text="train with 3 folds")
		lll1.grid(row=0, column=1, padx = 10)
		lll2 = tki.Label(self.frame_stats, text="train with 5 folds")
		lll2.grid(row=0, column=2, padx = 10)
		lll3 = tki.Label(self.frame_stats, text="train with 7 folds")
		lll3.grid(row=0, column=3, padx = 10)
		lll4 = tki.Label(self.frame_stats, text="test with best parameter and different test sizes")
		lll4.grid(row=0, column=4, padx = 10)
		self.stats = [tki.Label(self.frame_stats), tki.Label(self.frame_stats), tki.Label(self.frame_stats), tki.Label(self.frame_stats)]
		for i in range(len(self.stats)):
			self.stats[i].grid(row = 1, column = i+1, sticky='N', padx=10)
	
	def load_data(self):
		try:
			if self.list.curselection()[0] >= 0:
				if self.list.curselection()[0] == 0:
					self.data = get_faces()
					self.label_data.configure(text="64x64")
				if self.list.curselection()[0] == 1:
					self.data = read_faces_from_disk()
					self.label_data.configure(text="112x92")
		except Exception as e:
			print(e)
			mb.showinfo("Attention", "Please, select database")

	def set_feature(self):
		try:
			if self.list1.curselection()[0] >= 0:
				if self.list1.curselection()[0] == 0:
					self.method = get_histogram
				if self.list1.curselection()[0] == 1:
					self.method = get_dft
				if self.list1.curselection()[0] == 2:
					self.method = get_dct
				if self.list1.curselection()[0] == 3:
					self.method = get_scale
				if self.list1.curselection()[0] == 4:
					self.method = get_gradient
				self.label_method.configure(text = self.method.__name__)
		except Exception:
			mb.showinfo("Attention", "Please, select method")
			return

	def feature_example(self):
		if self.data is None:
			mb.showinfo("Attention", "Please, select database")
			return
		if self.method is None:
			self.set_feature()
		self.examples[0] = data_for_example(self.data)
		try:
			self.parameter = float(self.e3.get())
		except Exception:
			self.parameter = 0
		if self.parameter != 0:
			if self.method == get_histogram:
				self.parameter = int(self.parameter)
				if self.parameter > 300:
					self.parameter = 300
				if self.parameter < 10:
					self.parameter = 10
			if self.method == get_dft or self.method == get_dct:
				self.parameter = int(self.parameter)
				image_size = min(self.data[0][0].shape)
				if self.parameter > image_size - 1:
					self.parameter = image_size - 1
				if self.parameter < 2:
					self.parameter = 2
			if self.method == get_scale:
				if self.parameter > 1:
					self.parameter = 1
				if self.parameter < 0.05:
					self.parameter = 0.05
			if self.method == get_gradient:
				self.parameter = int(self.parameter)
				image_size = self.data[0][0].shape[0]
				if self.parameter > int(image_size/2 - 1):
					self.parameter = int(image_size/2 - 1)
				if self.parameter < 2:
					self.parameter = 2
			self.e3.delete(0, "end")
			self.e3.insert(0, self.parameter)
			self.examples[1] = [self.method(ex, self.parameter) for ex in self.examples[0]]
		else:
			self.examples[1] = [self.method(ex) for ex in self.examples[0]]

		for i in range(len(self.examples[0])):
			image = Image.fromarray(self.examples[0][i]*255)
			image = ImageTk.PhotoImage(image)
			self.images[i].configure(image=image)
			self.images[i].image = image
			if self.method == get_histogram:
				hist, bins = self.examples[1][i]
				hist = np.insert(hist, 0, 0.0)
				fig = plt.figure(figsize=(1.1,1.1))
				ax = fig.add_subplot(111)
				ax.plot(bins, hist)
				plt.xticks(color='w')
				plt.yticks(color='w')
				buf = io.BytesIO()
				fig.savefig(buf)
				buf.seek(0)
				image = Image.open(buf)
				image = ImageTk.PhotoImage(image)
				self.features[i].configure(image=image)
				self.features[i].image = image
			if self.method == get_dft or self.method == get_dct:
				fig = plt.figure(figsize=(1.1,1.1))
				ax = fig.add_subplot(111)
				ax.pcolormesh(range(self.examples[1][i].shape[0]),
									range(self.examples[1][i].shape[0]),
									np.flip(self.examples[1][i], 0), cmap="Greys")
				plt.xticks(color='w')
				plt.yticks(color='w')
				buf = io.BytesIO()
				fig.savefig(buf)
				buf.seek(0)
				image = Image.open(buf)
				image = ImageTk.PhotoImage(image)
				self.features[i].configure(image=image)
				self.features[i].image = image
			if self.method == get_scale:
				image = Image.fromarray(self.examples[1][i]*255)
				image = ImageTk.PhotoImage(image)
				self.features[i].configure(image=image)
				self.features[i].image = image
			if self.method == get_gradient:
				image_size = self.data[0][0].shape[0]
				fig = plt.figure(figsize=(1.1,1.1))
				ax = fig.add_subplot(111)
				ax.plot(range(0, len(self.examples[1][i])), self.examples[1][i])
				plt.xticks(color='w')
				plt.yticks(color='w')
				buf = io.BytesIO()
				fig.savefig(buf)
				buf.seek(0)
				image = Image.open(buf)
				image = ImageTk.PhotoImage(image)
				self.features[i].configure(image=image)
				self.features[i].image = image

	def start_computing(self):
		if self.data is None:
			mb.showinfo("Attention", "Please, select database")
			return
		try:
			self.number_face_test = [int(self.e1.get()), int(self.e2.get())]
		except Exception:
			mb.showinfo("Attention", "Please, input integer numbers in range [10, 200]")
			return
		if self.number_face_test[0] < 10:
			self.number_face_test[0] = 1
			self.e1.delete(0, "end")
			self.e1.insert(0, self.number_face_test[0])
		if self.number_face_test[1] > 200:
			self.number_face_test[1] = 200
			self.e2.delete(0, "end")
			self.e2.insert(0, self.number_face_test[1])
		if self.method is None:
			self.set_feature()
		results = [0, 0, 0]
		x_train, x_test, y_train, y_test = split_data(self.data, images_per_person_in_test=5)
		train = mesh_data([x_train, y_train])
		test = mesh_data([x_test, y_test])
		count = 0
		for f in [3, 5, 7]:
			res = cross_validation(train, self.method, folds=f)
			if res[0][1] > results[1]:
				results = [res[0][0], res[0][1], f]
			plt.rcParams["font.size"] = "5"
			fig = plt.figure(figsize=(2.5, 2))
			ax = fig.add_subplot(111)
			ax.plot(res[1][0], res[1][1])
			buf = io.BytesIO()
			fig.savefig(buf)
			buf.seek(0)
			image = Image.open(buf)
			image = ImageTk.PhotoImage(image)
			self.stats[count].configure(image=image)
			self.stats[count].image = image
			count += 1
		
		self.label_parameter.configure(text = str(results[0]))
		self.label_score.configure(text = str(results[1]))
		self.label_folds.configure(text = str(results[2]))

		sizes = range(int(self.number_face_test[0]), int(self.number_face_test[1]), 10)
		test_results = [sizes, []]
		for size in sizes:
			test_results[1].append(test_classifier(train, choose_n_from_data(test, size), self.method, results[0]))
		plt.rcParams["font.size"] = "5"
		fig = plt.figure(figsize=(2.5, 2))
		ax = fig.add_subplot(111)
		ax.plot(test_results[0], test_results[1])
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.stats[count].configure(image=image)
		self.stats[count].image = image

	def onClose(self):
		print("[INFO] closing...")
		# self.stopEvent.set()
		self.root.quit()


pba = PhotoBoothApp()
pba.root.mainloop()
