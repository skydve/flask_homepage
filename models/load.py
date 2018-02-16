import numpy as np
from keras.models import model_from_json
import tensorflow as tf

def init():
    with open('models/model.json','r') as file:
        model = model_from_json(file.read())
    model.load_weights("models/model.h5")
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = tf.get_default_graph()

    return model, graph