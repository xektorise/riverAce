import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from keras.src.models import Sequential
from keras.src.layers import Dense


class NeuralNetwork:
    def __init__(self):
        self.model = None
    
    def build_model(self, input, output):
        self.model = Sequential[(
            Dense(64, input = input, activation='relu'),
            Dense(64, activation='relu'),
            Dense(output, activation='softmax')
        )]
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_model(self, X_train, Y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return self.model.predict(X)
    