from __future__ import print_function
from modules import VideoStream
from fer import FER
import time
import Tkinter as tk
from UI import UI

#print ("starting camera")

classifier = "classifiers/haarcascade_frontalface_alt.xml"
trained = "trainedData.pkl"
predictor = "predictor.dat"

#vs = VideoStream().start()
#time.sleep(2.0)

#fer = FER(vs,classifier,predictor,trained)
root = tk.Tk()
root.geometry("720x520+300+300")
fer = UI(root)
fer.mainloop()