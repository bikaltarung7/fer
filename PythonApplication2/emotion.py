import cv2
import dlib
import modules
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

class Emotion:
    
    def __init__(self):
        self.predictor = dlib.shape_predictor("predictor.dat")
        self.clf = SVC(kernel='linear', probability=True, tol = 1e-3)
        self.clf = joblib.load('trainedSVM.pkl')
        self.emotions = ['Angry','Happy','Neutral','Sad','Shocked']

    def getLandmarks(self,image,faces):

        for x,y,w,h in faces:
            roi = image[y:y+h,x:x+w]

            #incresing the size of the ROI to give it to the shape predictor as image 
            ROI = image[(y-10):(y+h+10),(x-10):(x+w+10)]

            #resizing the roi
            resized_roi = modules.resize(roi,width=200,inter=cv2.INTER_CUBIC)

            #getting the bottom and the right corner of the resized image
            bottom,right = resized_roi.shape[:2]

            #converting cv2 rectangle to dlib rectangle
            rect = dlib.rectangle(x,y,x+w,y+h)
            resized_rect = dlib.rectangle(0,0,right,bottom)

            shape = self.predictor(image,rect)
            resized_shape = self.predictor(ROI,resized_rect)

            xlist = []
            ylist = []
            landmarks_vectorised = []
            pred = []


            for i in range(36,68):
                xlist.append(float(resized_shape.part(i).x))
                ylist.append(float(resized_shape.part(i).y))
                cv2.circle(image,(shape.part(i).x,shape.part(i).y),1,(255,0,0),-1)

            xmean = np.mean(xlist)
            ymean = np.mean(ylist)

            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]

            for x,y,w,z in zip(xcentral,ycentral,xlist,ylist):
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)

            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)

            pred.append(landmarks_vectorised)

            prediction = self.clf.predict(pred)
            expression = self.emotions[int(prediction)]

            cv2.putText(image,str(expression),(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        return image
