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

class PhotoBoothApp:
	def __init__(self):
		self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
		self.thread = None
		self.stopEvent = None
		self.root = tki.Tk()
		self.root.title("Paractice 3. Face classificator")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.root.quit())
		self.root.geometry("1350x700+50+50")
		self.root.resizable(False, False)
		
		self.frame_settings = tki.Frame(self.root, bg='peach puff', height=100, width=1350)
		self.frame_settings.grid(row=0)
		self.frame_example = tki.Frame(self.root, height=200, width=1350)
		self.frame_example.grid(row=1, sticky = 'N')
		self.frame_stats = tki.Frame(self.root)
		self.frame_stats.grid(row=2)
		
		lbl1 = tki.Label(self.frame_settings, text="Step 1. Choose dataset")
		lbl1.grid(row=0, column=0, padx = 30, pady = 10)

		self.list = tki.Listbox(self.frame_settings, selectmode = "single", width=35, height=2)
		self.list.grid(row=1, column=0, sticky='N', padx = 30, pady=8)
		m = ["The ORL face database (64x64)", "The ORL face database (112x92)"]
		for each_item in m: 
			self.list.insert("end", each_item)  

		btn1 = tki.Button(self.frame_settings, text="Load data", command=self.load_data)
		btn1.grid(row=2, column=0, sticky='N', padx = 30, pady = 8)

		self.label_data = tki.Label(self.frame_settings)
		self.label_data.grid(row=3, column=0, sticky='N', padx = 30)

		lbl3 = tki.Label(self.frame_settings, text="Step 2. Choose numbers of every face in train")
		lbl3.grid(row=0, column=2, padx = 25, pady = 10)

		l1 = tki.Label(self.frame_settings, text="Low range (>=1)")
		l1.grid(row=1, column=2, padx = 25, pady = 8, sticky='NW')
		self.e1 = tki.Entry(self.frame_settings, width=10)
		self.e1.grid(row=1, column=2, padx = 25, pady = 8, sticky='NE')
		l2 = tki.Label(self.frame_settings, text="High range (<=8)")
		l2.grid(row=1, column=2, padx = 25, pady = 40, sticky='NW')
		self.e2 = tki.Entry(self.frame_settings, width=10)
		self.e2.grid(row=1, column=2, padx = 25, pady = 40, sticky='NE')

		lbl4 = tki.Label(self.frame_settings, text="Step 3. Choose method")
		lbl4.grid(row=0, column=4, padx = 25, pady = 10)

		self.list1 = tki.Listbox(self.frame_settings, selectmode = "single", width=35, height=5)
		self.list1.grid(row=1, column=4, sticky='N', padx = 25, pady=8)
		self.methods = ["histogram", "dft", "dct", "scale", "gradient"]
		for each_item in self.methods:
			self.list1.insert("end", each_item)

		btn4 = tki.Button(self.frame_settings, text="Confirm", command=self.set_feature)
		btn4.grid(row=2, column=4, sticky='N', padx = 25, pady = 8)

		self.label_method = tki.Label(self.frame_settings, text="")
		self.label_method.grid(row=3, column=4, padx = 25)

		lbl2 = tki.Label(self.frame_settings, text="Step 4.")
		lbl2.grid(row=0, column=5, padx = 25, pady = 10)

		btn2 = tki.Button(self.frame_settings, text="Show feature examples", command=self.feature_example)
		btn2.grid(row=1, column=5, sticky='N', padx = 25, pady = 10)

		lbl5 = tki.Label(self.frame_settings, text="Step 5.")
		lbl5.grid(row=0, column=6, padx = 25, pady = 10)

		btn3 = tki.Button(self.frame_settings, text="Start computing", command=self.start_computing)
		btn3.grid(row=1, column=6, sticky='N', padx = 25, pady = 10)

		lbl6 = tki.Label(self.frame_example, text="Feature example:")
		lbl6.grid(row=0, column=0, padx = 30, sticky='NW', ipadx = 0)	

		lbl7 = tki.Label(self.frame_stats, text="Test data computing stats:")
		lbl7.grid(row=0, column=0, padx = 30, sticky='NW', ipadx = 0)

		self.data = None
		self.number_face_train = [1, 8]
		self.control_faces = None
		self.method = None
	
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
		pass

	def start_computing(self):
		if self.data is None:
			mb.showinfo("Attention", "Please, select database")
			return
		try:
			self.number_face_train = [int(self.e1.get()), int(self.e2.get())]
		except Exception:
			mb.showinfo("Attention", "Please, input integer numbers in range [1, 8]")
			return
		if self.number_face_train[0] < 1:
			self.number_face_train[0] = 1
			self.e1.delete(0, "end")
			self.e1.insert(0, self.number_face_train[0])
		if self.number_face_train[1] > 8:
			self.number_face_train[1] = 8
			self.e2.delete(0, "end")
			self.e2.insert(0, self.number_face_train[1])
		if self.method is None:
			self.set_feature()
		pass
	
	def onClose(self):
		print("[INFO] closing...")
		# self.stopEvent.set()
		self.root.quit()


pba = PhotoBoothApp()
pba.root.mainloop()
