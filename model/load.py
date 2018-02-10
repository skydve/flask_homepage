import numpy as np
# import keras.models
from keras.models import model_from_json
import tensorflow as tf


def init(): 

	with open('model/model.json','r') as file:
		model = model_from_json(file.read())

	model.load_weights("model/model.h5")
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return model, graph