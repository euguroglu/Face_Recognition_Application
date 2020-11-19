import cv2


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(-1)

    def __del__(self):
        self.video.release()


    def get_frame(self):

        ret,frame = self.video.read()
        frame = cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            break

        ret,jpeg = cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

    def get_skecth(self):

        ret,frame = self.video.read()
        #Convert image to grayscale
        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #Clean up image using Gaussian Blur
        img_gray_blur = cv2.GaussianBlur(img_gray,(5,5),0)
        #Extract edges from image
        canny_edges = cv2.Canny(img_gray_blur,10,70)
        #Do an invert binarizer the image
        ret,mask = cv2.threshold(canny_edges,70,255,cv2.THRESH_BINARY_INV)
        ret,jpeg = cv2.imencode('.jpg',mask)
        return jpeg.tobytes()
