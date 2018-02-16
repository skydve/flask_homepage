from flask import Flask, render_template, request
import numpy as np
import base64
from scipy.misc import imread, imresize
from models import load

app = Flask(__name__)

model, graph = load.init()

def convertImage(imgData):
	with open('output.png','wb') as output:
		output.write(base64.decodebytes(imgData[22:]))

@app.route('/')
def index():
    return render_template('index.html')

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
  app.run(host='0.0.0.0', port=8006, debug=True)
 