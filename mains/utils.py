import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import cv2

#Loading model_svm
# Load haar cascade
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
# Load pickle files
mean = pickle.load(open('./model/mean_preprocess.pickle','rb'))
model_svm = pickle.load(open('./model/model_svm.pickle','rb'))
model_pca = pickle.load(open('./model/pca_50.pickle','rb'))

#Settings
gender_pred = ['Male','Female']
font = cv2.FONT_HERSHEY_SIMPLEX

def pipeline_model(path,filename,color='bgr'):

    img = cv2.imread(path)
    if color == 'bgr':
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #step 3:Crop the face using haar cascade
    faces = haar.detectMultiScale(gray,1.5,3)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    #step 4:Normalization
        roi = gray[y:y+h,x:x+w]
        roi = roi/255.0
    #step 5:Resize to 100x100
        if roi.shape[1]> 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
    #step 6:flattening 1x10000
        roi_reshape = roi_resize.reshape(1,-1)
    #step 7: Substract with mean
        roi_mean = roi_reshape-mean
    #step 8:Get eigen image
        eigen_image = model_pca.transform(roi_mean)
    #step 9: pass to ml model (svm)
        results = model_svm.predict_proba(eigen_image)[0]
    #step 10:
        predict = results.argmax()
        score = results[predict]
    #step 11:
        text = "%s : %0.2f"%(gender_pred[predict],score)
        cv2.putText(img,text,(x,y),font,1,(255,255,0),2)

    cv2.imwrite('./static/predict/{}'.format(filename),img)
