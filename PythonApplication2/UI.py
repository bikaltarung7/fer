import Tkinter as Tk
from Tkinter import E,W,N,S
from PIL import Image,ImageTk
from ttk import Frame,Button, Label, Style
from capture import Capture
import threading
import tkMessageBox as mbox
from trainer import Trainer

class UI(Frame):
    def __init__(self,parent):

        Frame.__init__(self,parent)

        self.parent = parent

        self.initUI()



    def initUI(self):
        self.parent.title("FER")
        self.parent.wm_protocol("WM_DELETE_WINDOW",self.stop)
        self.pack(fill=Tk.BOTH,expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        label  = Label(self,text = "FER")
        label.grid()
        self.frameArea = Label(self,relief=Tk.RAISED, borderwidth=1)
        self.frameArea.grid(row =1, column=0, columnspan=2, rowspan=4,padx=5,pady=5,sticky = E+W+N+S)

        startButton = Button(self, text="Start",command=self.start)
        startButton.grid(row = 1,column=3)

        updateButton = Button(self,text = "Update Emotions",command=self.update)
        updateButton.grid(row = 2,column = 3)

        trainButton = Button(self,text = "train",command=self.train)
        trainButton.grid(row = 3,column = 3)
        
        self.cap = Capture(self.frameArea)
       
        self.trainer = Trainer()

       
    def start(self):
        print "Starting"
          
        self.cap.start()

    def update(self):
        print "Starting training"
        
        self.cap.capture()

        #mbox.showinfo("Completed", "Emotion capturing has been completed.")
    
    def train(self):
        self.trainer.train()

    def stop(self):
        
        self.cap.close()
        self.quit()
        print "stopping 2"