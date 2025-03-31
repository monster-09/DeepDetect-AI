from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle
import sqlite3
import random
import smtplib 
from email.message import EmailMessage
import argparse
import io
import os
from flask import Flask, render_template, request, redirect, Response
from tensorflow.keras.utils import to_categorical
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import base64
import io
import os
import cv2
import numpy as np
# from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 


app = Flask(__name__)

path = "Dataset"
labels = []
X = []
Y = []
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())   
print("Dataset Class Labels : "+str(labels))
    
def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def GradCamImage(image_path, ext_model):
    grad_cam = Model(inputs = ext_model.inputs, outputs = ext_model.layers[0].output)
    image = cv2.imread(image_path)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = grad_cam.predict(img)[0]
    return preds

def getModel():
    extension_model = Sequential()
    extension_model.add(Convolution2D(32, (3 , 3), input_shape = (32, 32, 3), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (2, 2)))
    #for each CNN2d and Maxpool layer we have connected dropout layer to remove all irrelevant features
    extension_model.add(Dropout(0.3))
    extension_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (2, 2)))
    #adding another dropout layer for CNN and max layer
    extension_model.add(Dropout(0.3))
    extension_model.add(Flatten())
    extension_model.add(Dense(units = 256, activation = 'relu'))
    extension_model.add(Dense(units = 2, activation = 'softmax'))
    #compiling, training and loading model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    extension_model.load_weights("model/extension_weights.hdf5")   
    return extension_model    


   
@app.route("/about")
def about():
    return render_template("graph.html")




@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['files']
        img_bytes = file.read()
        if os.path.exists("static/test.jpg"):
            os.remove("static/test.jpg")
        with open('static/test.jpg', mode="wb") as jpg:
            jpg.write(img_bytes)
        jpg.close()
        extension_model = getModel()
        image = cv2.imread('static/test.jpg')#read test image   
        img = cv2.resize(image, (32, 32))#resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
        img = np.asarray(im2arr)
        img = img.astype('float32')#convert image features as float
        img = img/255 #normalized image
        predict = extension_model.predict(img)#now predict dog breed
        predict = np.argmax(predict)
        grad_cam = GradCamImage('static/test.jpg', extension_model)
        img = cv2.imread('static/test.jpg')
        img = cv2.resize(img, (500,300))#display image with predicted output
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        output = 'Predicted As : '+labels[predict]
        figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
        axis[0].set_title("Original Image")
        axis[1].set_title("Explainable Grad-Cam Image")
        axis[0].imshow(img, cmap='hot')
        axis[1].imshow(grad_cam[:,:,31], cmap='hot')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode() 
        return render_template('after.html', msg=output, img = img_b64)



@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("CIFAKE.html")





if __name__ == '__main__':
    app.run(debug=False)