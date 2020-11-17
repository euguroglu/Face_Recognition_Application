from flask import Flask, render_template,Response,request
from camera import VideoCamera
from flask import redirect,url_for
import os
from PIL import Image
from mains.utils import pipeline_model


UPLOAD_FOLDER ='static/upload'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/projects')
def projects():
    return render_template("projects.html")

@app.route('/project1')
def project1():
    return render_template("project1.html")

@app.route('/project2')
def project2():
    return render_template("project2.html")

def getwidth(path):
    img = Image.open(path)
    size = img.size #width and height
    aspect = size[0]/size[1] #witdh/height
    w = 300 * aspect

    return int(w)

@app.route('/project2/gender',methods=['GET','POST'])
def gender():
    if request.method == 'POST':
        f=request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        w = getwidth(path)
        #prediction pass to pipeline model
        img = pipeline_model(path,filename,color='bgr')
        return render_template("gender.html",fileupload=True,img_name=filename,w=w)
    return render_template("gender.html",fileupload=False,img_name="freeai.png",w='300')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    app.run(debug=True)
