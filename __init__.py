from flask import Flask, flash, json, redirect, sessions, request, jsonify
import os
from flask_restful import reqparse, Api, Resource
# from routes.routes import *
import cv2
import calendar
import time
import pathlib

app = Flask(__name__)
api = Api(app)


@app.route("/")
def hello_world():
  return "Hello, This is Python codebase!"

@app.post("/upload")
def upload_file():
    print('INSIDE UPLOAD FILE API')
    if request.method == 'POST':
        print('INSIDE UPLOAD FILE API- POST')
        if 'file' not in request.files:
            print('No File part')
            return jsonify('No File found')
        imgToUpload = request.files['file']
        if 'id' not in request.form:
          return jsonify('Please pass Id')
        if 'name' not in request.form:
          return jsonify('Please pass name')
        id = request.form['id']
        name = request.form['name']
        fileName = 'user-' + id + '-' + name + '.png'
        if imgToUpload:
            imgToUpload.save(os.path.join("/home/lalith/workspace/python workspace/faceRecognitionPrj/known-user-images", fileName))  
            return jsonify('Uploaded - '+ fileName)

@app.post("/recognize-img")
def recognizeImg():
  if request.method == 'POST':
    names = ['dummy', 'Lalith', 'shovanlal', 'bhavi']

    imgToUpload = request.files['file']
    
    filePrefix = calendar.timegm(time.gmtime())
    fileExtension = pathlib.Path(imgToUpload.filename).suffix
    fileName = str(filePrefix) + fileExtension

    imgToUpload.save(os.path.join("/home/lalith/workspace/python workspace/faceRecognitionPrj/unknown-user-images", fileName))
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    img = cv2.imread('unknown-user-images/'+ fileName)
    print(fileName)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5
      )
    resp = []
    for(x,y,w,h) in faces:
      id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
      resp.append((names[id], confidence))
    return jsonify(resp)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()
