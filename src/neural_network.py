import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NeuralNetwork:
    def __init__(self):
        self.model = None
    
    def build_model(self, input_shape, output_units):
        """
        Build the neural network model.
        
        Parameters:
        input_shape (tuple): Shape of the input data (excluding batch size).
        output_units (int): Number of output units (classes for classification).
        """
        self.model = Sequential([
            Dense(64, input_shape=input_shape, activation='relu'),
            Dense(64, activation='relu'),
            Dense(output_units, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_model(self, X_train, Y_train, epochs=10, batch_size=32):
        """
        Train the model.
        
        Parameters:
        X_train (numpy.ndarray): Training features.
        Y_train (numpy.ndarray): Training labels (one-hot encoded).
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        """
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        """
        Predict using the trained model.
        
        Parameters:
        X (numpy.ndarray): Input data for prediction.
        
        Returns:
        numpy.ndarray: Model predictions.
        """
        return self.model.predict(X)