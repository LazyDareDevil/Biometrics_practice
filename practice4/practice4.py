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
import random as rnd

class PhotoBoothApp:
	def __init__(self):
		self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
		self.thread = None
		self.stopEvent = None
		self.root = tki.Tk()
		self.canvas = tki.Canvas(self.root)
		self.root.title("Paractice 4. Parallel face classifier. Vote.")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.root.quit())
		self.root.geometry("1050x700+50+50")
		self.root.resizable(False, False)
		self.canvas.grid(row=0, column=0)
		
		self.frame_settings = tki.Frame(self.canvas, bg='peach puff', height=100)
		self.frame_settings.grid(row=0)
		self.frame_example = tki.Frame(self.canvas, height=600)
		self.frame_example.grid(row=1, sticky = 'N')

		lbl1 = tki.Label(self.frame_settings, text="Choose dataset")
		lbl1.grid(row=0, column=0, padx = 20, pady = 10)

		self.list = tki.Listbox(self.frame_settings, selectmode = "single", width=30, height=2)
		self.list.grid(row=1, column=0, sticky='N', padx = 20, pady=8)
		m = ["The ORL face database (64x64)", "The ORL face database (112x92)"]
		for each_item in m: 
			self.list.insert("end", each_item)  
		self.label_data = tki.Label(self.frame_settings)
		self.label_data.grid(row=3, column=0, sticky='N', padx = 20, pady = 2)

		btn1 = tki.Button(self.frame_settings, text="Load data", command=self.load_data)
		btn1.grid(row=2, column=0, sticky='N', padx = 20, pady = 8)

		btn2 = tki.Button(self.frame_settings, text="Start train computing", command=self.computing)
		btn2.grid(row=1, column=1, sticky='N', padx = 20, pady = 8)

		self.methods = [get_histogram, get_dft, get_dct, get_gradient, get_scale]
		self.parameters = None
		self.size_train = None
		self.best_score = 0

		btn3 = tki.Button(self.frame_settings, text="Start classification", command=self.classif)
		btn3.grid(row=2, column=1, sticky='N', padx = 20, pady = 8)

		lbl1 = tki.Label(self.frame_settings, text="Best results for train")
		lbl1.grid(row=0, column=2, padx = 25, pady = 8, sticky='N')

		label_now_method = tki.Label(self.frame_settings, text="histogram, dft, dct, gradient, scale")
		label_now_method.grid(row=1, column=3, pady = 8, sticky='NW')

		lbl2 = tki.Label(self.frame_settings, text="Best parameters")
		lbl2.grid(row=1, column=2, padx = 25, pady = 30, sticky='N')
		self.label_parameter = tki.Label(self.frame_settings, text="")
		self.label_parameter.grid(row=1, column=3, pady = 30, sticky='NW')

		lbl3 = tki.Label(self.frame_settings, text="Best score")
		lbl3.grid(row=2, column=2, padx = 25, sticky='N')
		self.label_score = tki.Label(self.frame_settings, text="")
		self.label_score.grid(row=2, column=3, sticky='NW')

		lbl4 = tki.Label(self.frame_settings, text="Best faces per class")
		lbl4.grid(row=3, column=2, padx = 25, sticky='N')
		self.label_folds = tki.Label(self.frame_settings, text="")
		self.label_folds.grid(row=3, column=3, sticky='NW')

		lbl5 = tki.Label(self.frame_example, text="Classification example:")
		lbl5.grid(row=0, column=0, padx = 10, sticky='NW')

		lbl6 = tki.Label(self.frame_example, text="Original image:")
		lbl6.grid(row=0, column=1, padx = 10, sticky='NW')

		lbl7 = tki.Label(self.frame_example, text="histogram:")
		lbl7.grid(row=0, column=2, padx = 10, sticky='NW')

		lbl8 = tki.Label(self.frame_example, text="DFT:")
		lbl8.grid(row=0, column=3, padx = 10, sticky='NW')

		lbl9 = tki.Label(self.frame_example, text="DCT:")
		lbl9.grid(row=0, column=4, padx = 10, sticky='NW')

		lbl10 = tki.Label(self.frame_example, text="gradient:")
		lbl10.grid(row=0, column=5, padx = 10, sticky='NW')

		lbl11 = tki.Label(self.frame_example, text="scale:")
		lbl11.grid(row=0, column=6, padx = 10, sticky='NW')

		lbl12 = tki.Label(self.frame_example, text="another image from class:")
		lbl12.grid(row=0, column=7, padx = 10, sticky='NW')

		self.data = None
		self.images_1 = [tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example),
						tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example)]
		for i in range(len(self.images_1)):
			self.images_1[i].grid(row = 1, column = i+1, sticky='N', padx=10, pady=2)
		self.images_2 = [tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example),
						tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example)]
		for i in range(len(self.images_2)):
			self.images_2[i].grid(row = 2, column = i+1, sticky='N', padx=10, pady=2)		
		self.images_3 = [tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example),
						tki.Label(self.frame_example), tki.Label(self.frame_example), tki.Label(self.frame_example)]
		for i in range(len(self.images_2)):
			self.images_3[i].grid(row = 3, column = i+1, sticky='N', padx=10, pady=2)	

	def onClose(self):
		print("[INFO] closing...")
		# self.stopEvent.set()
		self.root.quit()

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

	def computing(self):
		if self.data is None:
			mb.showinfo("Attention", "Please, select database")
			return
		mb.showinfo("Attention", "Wait, while parameters will be computed")
		self.parameters, self.size_train, self.best_score = cross_validation(self.data)
		str = "[ "
		for method in self.methods:
			if method == get_scale:
				str += '{0:.2f}'.format(self.parameters[method.__name__])
			else:
				str += '{}'.format(self.parameters[method.__name__])
			str += ";  "
		str = str[:-3]
		str += " ]"
		self.label_parameter.configure(text = str)
		self.label_folds.configure(text = "{}".format(self.size_train))
		self.label_score.configure(text="{0:.4f}".format(self.best_score))

	def classif(self):
		if self.parameters is None or self.size_train is None:
			mb.showinfo("Attention", "Please, start training first")
			return
		x_train, x_test, y_train, y_test = split_data(self.data, self.size_train)
		train = mesh_data([x_train, y_train])
		mb.showinfo("Attention", "Wait, while classification will be computed")
		v = voting(train, x_test, self.parameters)

		indexes = rnd.sample(range(0, len(x_test)), 3)
		example1 = x_test[indexes[0]]*255
		example2 = x_test[indexes[1]]*255
		example3 = x_test[indexes[2]]*255
		
		print(v[indexes[0]], " ", y_test[indexes[0]])
		print(v[indexes[1]], " ", y_test[indexes[1]])
		print(v[indexes[2]], " ", y_test[indexes[2]])
		
		image = Image.fromarray(example1)
		image = ImageTk.PhotoImage(image)
		self.images_1[0].configure(image=image)
		self.images_1[0].image = image

		hist, bins = get_histogram(example1/255, self.parameters["get_histogram"])
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
		self.images_1[1].configure(image=image)
		self.images_1[1].image = image

		dft = get_dft(example1, self.parameters["get_dft"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.pcolormesh(range(dft.shape[0]),
							range(dft.shape[0]),
							np.flip(dft, 0), cmap="Greys")
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_1[2].configure(image=image)
		self.images_1[2].image = image

		dct = get_dft(example1, self.parameters["get_dct"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.pcolormesh(range(dct.shape[0]),
							range(dct.shape[0]),
							np.flip(dct, 0), cmap="Greys")
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_1[3].configure(image=image)
		self.images_1[3].image = image

		hist = get_gradient(example1, self.parameters["get_gradient"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.plot(range(0, len(hist)), hist)
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_1[4].configure(image = image)
		self.images_1[4].image = image

		image = Image.fromarray(cv2.resize(example1,
											(int(self.parameters["get_scale"]*example1.shape[0]), int(self.parameters["get_scale"]*example1.shape[1])), 
											interpolation = cv2.INTER_AREA))
		image = ImageTk.PhotoImage(image)
		self.images_1[5].configure(image=image)
		self.images_1[5].image = image

		image = Image.fromarray(self.data[0][10*v[indexes[0]]]*255)
		image = ImageTk.PhotoImage(image)
		self.images_1[6].configure(image=image)
		self.images_1[6].image = image
		
		image = Image.fromarray(example2)
		image = ImageTk.PhotoImage(image)
		self.images_2[0].configure(image=image)
		self.images_2[0].image = image

		hist, bins = get_histogram(example2/255, self.parameters["get_histogram"])
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
		self.images_2[1].configure(image=image)
		self.images_2[1].image = image

		dft = get_dft(example2, self.parameters["get_dft"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.pcolormesh(range(dft.shape[0]),
							range(dft.shape[0]),
							np.flip(dft, 0), cmap="Greys")
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_2[2].configure(image=image)
		self.images_2[2].image = image

		dct = get_dft(example2, self.parameters["get_dct"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.pcolormesh(range(dct.shape[0]),
							range(dct.shape[0]),
							np.flip(dct, 0), cmap="Greys")
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_2[3].configure(image=image)
		self.images_2[3].image = image

		hist = get_gradient(example2, self.parameters["get_gradient"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.plot(range(0, len(hist)), hist)
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_2[4].configure(image=image)
		self.images_2[4].image = image

		image = Image.fromarray(cv2.resize(example2, 
								(int(self.parameters["get_scale"]*example2.shape[0]), int(self.parameters["get_scale"]*example2.shape[1])), 
								interpolation = cv2.INTER_AREA))
		image = ImageTk.PhotoImage(image)
		self.images_2[5].configure(image=image)
		self.images_2[5].image = image

		image = Image.fromarray(self.data[0][10*v[indexes[1]]]*255)
		image = ImageTk.PhotoImage(image)
		self.images_2[6].configure(image=image)
		self.images_2[6].image = image

		image = Image.fromarray(example3)
		image = ImageTk.PhotoImage(image)
		self.images_3[0].configure(image=image)
		self.images_3[0].image = image

		hist, bins = get_histogram(example3/255, self.parameters["get_histogram"])
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
		self.images_3[1].configure(image=image)
		self.images_3[1].image = image

		dft = get_dft(example3, self.parameters["get_dft"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.pcolormesh(range(dft.shape[0]),
							range(dft.shape[0]),
							np.flip(dft, 0), cmap="Greys")
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_3[2].configure(image=image)
		self.images_3[2].image = image

		dct = get_dft(example3, self.parameters["get_dct"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.pcolormesh(range(dct.shape[0]),
							range(dct.shape[0]),
							np.flip(dct, 0), cmap="Greys")
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_3[3].configure(image=image)
		self.images_3[3].image = image

		hist = get_gradient(example3, self.parameters["get_gradient"])
		fig = plt.figure(figsize=(1.1,1.1))
		ax = fig.add_subplot(111)
		ax.plot(range(0, len(hist)), hist)
		plt.xticks(color='w')
		plt.yticks(color='w')
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		image = Image.open(buf)
		image = ImageTk.PhotoImage(image)
		self.images_3[4].configure(image=image)
		self.images_3[4].image = image

		image = Image.fromarray(cv2.resize(example3, 
								(int(self.parameters["get_scale"]*example3.shape[0]), int(self.parameters["get_scale"]*example3.shape[1])), 
								interpolation = cv2.INTER_AREA))
		image = ImageTk.PhotoImage(image)
		self.images_3[5].configure(image=image)
		self.images_3[5].image = image

		image = Image.fromarray(self.data[0][10*v[indexes[2]]]*255)
		image = ImageTk.PhotoImage(image)
		self.images_3[6].configure(image=image)
		self.images_3[6].image = image

pba = PhotoBoothApp()
pba.root.mainloop()
