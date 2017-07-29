import cv2
import dlib
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import glob,math,itertools,random
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np 
from face import Face
import modules

class Trainer:
    def __init__(self):
        self.emotions = ["Angry","Happy","Neutral","Sad","Shocked"]
        self.face = Face()
        self.predictor = dlib.shape_predictor("predictor.dat")
        self.clf = SVC(kernel='linear', probability=True, tol = 1e-3)

    def  get_images(self,emotion):
        print emotion
        files = glob.glob("Dataset/%s/*" %emotion)
        print len(files)
        return files

    def get_landmarks(self,image):

        faces = self.face.detect(image)

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

             for i in range(36,68):
                 xlist.append(float(resized_shape.part(i).x))
                 ylist.append(float(resized_shape.part(i).y))

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
        if len(faces)<1:
            landmarks_vectorised = "error"
        return landmarks_vectorised

    def make_sets(self):
        training_data = []
        training_labels = []
        for emotion in self.emotions:
            training = self.get_images(emotion)
            for item in training:
                image = cv2.imread(item,0)
                landmarks_vectorised = self.get_landmarks(image)
                if landmarks_vectorised == "error":
                    pass
                else:
                    training_data.append(landmarks_vectorised)
                    training_labels.append(self.emotions.index(emotion))
        return training_data,training_labels

    def train(self):
        training_data,training_labels = self.make_sets()
        npar_train = np.array(training_data)
        print "Training linear SVM..."
        self.clf.fit(npar_train,training_labels)
        joblib.dump(self.clf,"trainedSVM.pkl")
        print "SVM trained successfully"