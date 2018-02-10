from flask import Flask, render_template, request
import numpy as np
import keras.models
import re
import sys
import os
import base64
from scipy.misc import imread, imresize

# sys.path.append(os.path.abspath('./model'))
# from load import *
from model import load


app = Flask(__name__)

#from load.py
global model, graph
model, graph = load.init()

@app.route('/')
def index():
    return render_template('index.html')

def convertImage(imgData):
	with open('output.png','wb') as output:
		output.write(base64.decodebytes(imgData[22:]))

@app.route('/mnist')
def mnist():
    return render_template('mnist.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	x = imread('output.png',mode='L')
	x = np.invert(x)
	x = imresize(x,(28,28))
	x = x.reshape(1,28,28,1)
	with graph.as_default():
		out = model.predict(x)
		response = np.array_str(np.argmax(out,axis=1))
		return response	


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8004, debug=True)
 