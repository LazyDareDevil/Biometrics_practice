from __future__ import print_function
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

class PhotoBoothApp:
	def __init__(self):
		self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
		self.frame = None
		self.thread = None
		self.stopEvent = None
		self.root = tki.Tk()
		self.root.title("Paractice 2. Face detectors")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.root.quit())
		self.root.geometry("1350x700+50+50")
		self.root.resizable(False, False)
		self.panel = tki.Label()
		self.panel.grid(row=1, column=0, sticky='N', padx=5, pady=8)
		self.template_panel = tki.Label()
		self.template_panel.grid(row=1, column=2, sticky='N', padx=20, pady=35)
		self.list = tki.Listbox(self.root, selectmode = "single", width=45, height=7)
		self.list.grid(row=1, column=1, sticky='N', padx=20, pady=8) 

		self.methods = ["cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED", "cv2.TM_CCORR", "cv2.TM_CCORR_NORMED", "cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "VJ"]
		m = ["Template Matching (TM_SQDIFF)", "Template Matching (TM_SQDIFF_NORMED)", "Template Matching (TM_CCORR)", "Template Matching (TM_CCORR_NORMED)", "Template Matching (TM_CCOEFF)", "Template Matching (TM_CCOEFF_NORMED)", "Viola & Jonson (Haar cascades)"]
		for each_item in m: 
			self.list.insert("end", each_item) 

		self.face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

		self.cap = cv2.VideoCapture(0)
		self.template = None
		self.temp_image = None
		self.method = None

		lbl1 = tki.Label(self.root, text="Camera output")
		lbl1.grid(row=0, column=0, padx = 5)

		lbl2 = tki.Label(self.root, text="Choose method")
		lbl2.grid(row=0, column=1, padx = 5)

		lbl2 = tki.Label(self.root, text="Add template (jpg, jpeg, png)")
		lbl2.grid(row=0, column=2, padx = 5)

		btn1 = tki.Button(self.root, text="Save a frame", command=self.takeSnapshot)
		btn1.grid(row=2, column=0, sticky='N', padx=5, pady=8)
		
		btn2 = tki.Button(self.root, text="Add template", command=self.addTemplate)
		btn2.grid(row=1, column = 2, sticky='N', padx=20, pady=8)
		
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

	def videoLoop(self):
		try:
			while not self.stopEvent.is_set():
				_, self.frame = self.cap.read()
				self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
				self.method = self.list.curselection()
				if len(self.method) > 0:
					if self.method[0] >=0 and self.method[0] <6:
						if self.temp_image is not None:
							met = eval(self.methods[self.method[0]])
							res = cv2.matchTemplate(self.image, self.temp_image, met)
							_, _, min_loc, max_loc = cv2.minMaxLoc(res)
							if met in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
								top_left = min_loc
							else:
								top_left = max_loc
							w, h = self.temp_image.shape
							bottom_right = (top_left[0] + w, top_left[1] + h)
							cv2.rectangle(self.frame, top_left, bottom_right, (255, 0, 0), 2)
					else:
						faces = self.face_cascade.detectMultiScale(self.image, 1.3, 5)
						for (x, y, w, h) in faces:
							self.frame = cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
							# roi_gray = self.image[y : y + h, x : x + w]
							eyes = self.eye_cascade.detectMultiScale(self.image)
							for (ex, ey, ew, eh) in eyes:
								cv2.rectangle(self.frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
				image = Image.fromarray(self.frame)
				image = ImageTk.PhotoImage(image)

				self.panel.configure(image=image)
				self.panel.image = image
		except RuntimeError:
			print("[INFO] caught a RuntimeError")

	def takeSnapshot(self):
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))
		cv2.imwrite(p, self.frame.copy())
		print("[INFO] saved {}".format(filename))

	def addTemplate(self):
		fl = filedialog.askopenfilename()
		if "jpg" in fl or "jpeg" in fl or "png" in fl:
			self.template = cv2.imread(fl)
			self.temp_image = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
			image = Image.fromarray(self.temp_image)
			image = ImageTk.PhotoImage(image)

			self.template_panel.configure(image=image)
			self.template_panel.image = image
		else:
			print("[INFO] didn't opened file")
	
	def onClose(self):
		print("[INFO] closing...")
		self.stopEvent.set()
		self.root.quit()

try:
	pba = PhotoBoothApp()
	pba.root.mainloop()
except Exception:
	pass