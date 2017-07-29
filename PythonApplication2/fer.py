from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tk
import threading
import datetime
import imutils
import cv2
import os
import dlib
import numpy as np
import glob,math,random
from sklearn.svm import SVC
from sklearn.externals import joblib
from modules import VideoStream
import modules


class FER:
    def __init__(self,vs,classifier,predictor,trained):
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.cascade = cv2.CascadeClassifier(classifier)
        self.predictor = dlib.shape_predictor(predictor)
        self.trained = trained
        self.clf = SVC(kernel='linear', probability=True, tol = 1e-3)

        self.root = tk.Tk()
        #self.panel  = None
        
        leftFrame = tk.Frame(self.root,width=800)
        leftFrame.pack(side="left")
        
        rightFrame = tk.Frame(self.root)
        rightFrame.pack(side="left")
        self.panel = tk.panel(leftFrame)
        self.panel.pack()

        start_btn = tk.Button(rightFrame,text="Start",command=self.start)
        start_btn.pack(side="bottom", padx=10,pady=10)
       
        capture_btn = tk.Button(rightFrame,text="Train",command=self.train)
        capture_btn.pack(side="left", padx=10,pady=10) 

        self.root.title("FER")
        self.emotions = ['Angry','Happy','Neutral','Sad','Shocked']

    def start(self):
        angry = 0
        happy = 0
        neutral = 0
        sad = 0
        shocked = 0

        

        expression = None

        self.clf = joblib.load(self.trained)
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop,args=())
        self.thread.start()

    def train(self):
        pass

    def videoLoop(self):
        try: 
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame,width=800)
                
                gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

                image = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)

                faces = self.cascade.detectMultiScale(gray,1.3,5)

                self.calculation(image,faces)

                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                if self.panel is None:
                    self.panel = tk.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side='left',padx=10,pady=10)
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError, e:
            print ("error")

    def calculation(self,image,faces):
        for x,y,w,h, in faces:
            roi = self.frame[y:y+h,x:x+w]

            #resizing the ROI
            roi = modules.resize(roi,width=500,inter=cv2.INTER_CUBIC)

            #getting the bottom and the right corner of the resized image
            bottom,right = roi.shape[:2]

            #converting  cv2 rectangle to dlib rectandle
            face_dlib = dlib.rectangle(x,y,x+w,y+h)
            rect = dlib.rectangle(0,0,bottom,right)

            #finding the facial landmark
            #predictor takes two argument
            #1) The image
            #2) The rectangle bounding the roi
            original_shape = self.predictor(self.frame,face_dlib)
            shape = self.predictor(roi,rect)

            xlist = []
            ylist = []
            landmarks = []
            pred = []
            for i in range(38,68):
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
                cv2.circle(image,(original_shape.part(i).x,original_shape.part(i).y),1,(255,0,0),-1)
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)

            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]

            for x,y in zip(xlist,ylist):
                landmarks.append(x)
                landmarks.append(y)
            pred.append(landmarks)

            prediction = self.clf.predict(pred)

            expression = self.emotions[int(prediction)]

            cv2.putText(image,str(expression),(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)



         