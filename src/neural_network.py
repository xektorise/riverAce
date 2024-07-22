import numpy as np
import shap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from typing import List, Dict
from data_processor import DataPreprocessor

class NeuralNetwork:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        self.preprocessor = DataPreprocessor()

    def _build_model(self, input_shape):
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(6, activation='softmax')  # 6 outputs for 6 possible actions
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, game_states: List[Dict], actions: List[int], epochs = 10, batch_size = 32):
        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.feature_columns()]
        y = keras.utils.to_categorical(df['action'], num_classes=6)
        x_scaled = self.preprocessor.fit_transform_data(x.values)
        early_stopping = EarlyStopping(monitor='val_loss', patience = 5)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only = True, monitor = 'val_loss')
        lr_scheduler = LearningRateScheduler(self.lr_scheudule)
        history = self.model.fit(x_scaled, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, checkpoint, lr_scheduler])
        
        return history

    def predict(self, game_state: Dict):
        preprocessed_state = self.preprocessor.preprocess_game_state(game_state)
        scaled_state = self.preprocessor.transform_data(preprocessed_state)
        return self.model.predict(scaled_state)
    
    def evaluate(self, game_states: List[Dict], actions: List[int]):
        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.get_feature_columns()]
        y = keras.utils.to_categorical(df['action'], num_classes = 6)
        x_scaled = self.preprocessor.transform_data(x.values)

        loss, accuracy = self.model.evaluate(x_scaled, y)
        return {'loss' : loss, 'accuracy' : accuracy}
    
    def feature_importance(self, game_states: List[Dict]):
        df = self.preprocessor.create_dataframe(game_states, [0]*len(game_states)) # dummy actions
        x = df[self.preprocessor.get_feature_columns()]
        x_scaled = self.preprocessor.transform_data(x.values)

        explainer = shap.DeepExplainer(self.model, x_scaled)
        shap_values = explainer.shap_values(x_scaled)

        feature_importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.preprocessor.get_feature_columns(), feature_importance))
    
    def lr_scheudule(self, epoch):
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lr = initial_lr * (drop ** (1 + epoch) // epochs_drop)
        
        return lr

    def save_model(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath):
        loaded_model = keras.models.load_model(filepath)
        instance = cls(loaded_model.input_shape[1:])
        instance.model = loaded_model
        return instance