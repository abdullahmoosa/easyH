from flask import Flask, render_template, request, send_file
from PIL import Image
# # Importing the OpenCV library.
import cv2
import tensorflow, keras
import numpy as np
import pandas as pd
from waitress import serve


app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")


    
@app.route("/submit", methods=["POST"])
def submit():

    # Getting the image from html to python
    if request.method == "POST":
        image = request.files["image_upload"]
        img = Image.open(image)
        img.save('image.jpg')
        img = img.resize((180,180))
        # pred_img = cv2.imread('image.jpg')
        # reshape_image = cv2.reshape(pred_img, (180,180))
        load_model = keras.models.load_model('models/easyH_cnn.h5', compile= False)
        pred = load_model.predict(np.expand_dims(img, axis = 0))[0][0]
        print(pred)


        # print(img.shape)
        return render_template('submit.html', pred = str(pred))

# print(img.sha)
img = cv2.imread('image.jpg')
# print(img.shape)
# print(img)
# def submit():
#     # Getting the image from html to python
#     if request.method == "POST":
#         image = request.files["image_upload"]
#         # img = Image.open(image)
#         # img = img.convert('RGB')
#         image.save('image.jpg')
#         # print(name.shape)
#         return send_file('image.jpg')



if __name__ == "__main__":
    
    serve(app=app,host='0.0.0.0', port=8080)
