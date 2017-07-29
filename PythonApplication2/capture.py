from modules import VideoStream
from fer import FER
import time
import threading
import Tkinter as tk
from ttk import Frame,Button, Label, Style
from PIL import Image,ImageTk
import cv2
from face import Face
from emotion import Emotion
from dataCreator import dataCreator

class Capture:
    def __init__(self,label):
        self.thread1 = None
        self.thread2 = None
        self.stopEvent1 = None
        self.stopEvent2 = None
        self.started = None
        self.captured = None
        self.label = label
        self.vs = VideoStream().start()
        self.fa = Face()
        self.emotion =  Emotion()
        self.dataCreate = dataCreator(self.label,self.vs)


    def start(self):
        #setting started to true to set it is started in the train function.
        #Because None type cannot be set
        self.started = True


        #Erasing stopEvent value (if any) to loop in the function
        self.stopEvent1 = None

        #checking if train thred is started
        #if started close it
        if(self.captured):
            self.stopEvent2.set()

        #initialize the thread
        self.stopEvent1 = threading.Event()
        self.thread1 = threading.Thread(target=self.videoLoop,args=())
        self.thread1.start()

    def close(self):
   
        #initiallly checking if the threads have been started and then closing 
        #if started
        
        if(self.started):
            self.stopEvent1.set()  
        if(self.captured):
            self.stopEvent2.set() 
        self.vs.stop()

    def train(self):      
        #self.close()
        var = tk.StringVar()

        for i in range(1,10):
            old = var.get()
            var.set(old+"\n"+str(i))
            self.label.configure(textvariable = var)
            self.label.textvariable = var

    def capture(self):
        #setting the trained variable to true for same reason as started
        self.captured = True

        #Setting StopEvent to None to loop 
        self.stopEvent2 = None
        
        #stopping started thread if running
        if(self.started):
            self.stopEvent1.set()
         
        #initializing the train thred
        self.stopEvent2 = threading.Event()
        self.thread2 = threading.Thread(target=self.captureLoop,args=())
        self.thread2.start()


    def videoLoop(self):
        try:
            #loop until the stopEvent is not set
            while not self.stopEvent1.is_set():
                frame = self.vs.read()

                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                faces = self.fa.detect(gray)
                
                image = self.emotion.getLandmarks(image,faces)

                image = Image.fromarray(image)

                image = ImageTk.PhotoImage(image)

                self.label.configure(image = image)
                self.label.image = image

        except RuntimeError, e :
            print "Error"

    def captureLoop(self):
        try:
            while not self.stopEvent2.is_set():
                #frame = self.vs.read()

                #image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                self.dataCreate.create()

                #image = Image.fromarray(image)

                #image = ImageTk.PhotoImage(image)

                #self.label.configure(image = image)
                #self.label.image = image
               
                self.stopEvent2.set()
                return 

        except RuntimeError, e :
            print "Error"

    
    