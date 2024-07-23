import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import Hyperparameters
from sklearn.model_selection import KFold
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
    
    def _build_tunable_model(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                                   activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    
    def tune_hyperparameters(self, game_states: List[Dict], actions: List[int], epochs=10, max_trials=5):
        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.get_feature_columns()]
        y = keras.utils.to_categorical(df['action'], num_classes=6)
        x_scaled = self.preprocessor.fit_transform_data(x.values)

        tuner = RandomSearch(
            self._build_tunable_model,
            objective='val_accuracy',
            max_trials=max_trials,
            executions_per_trial=2,
            directory='my_dir',
            project_name='hparam_tuning'
        )

        tuner.search(x_scaled, y, epochs=epochs, validation_split=0.2)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        self.model = tuner.hypermodel.build(best_hps)

        return best_hps
            

    def train(self, game_states: List[Dict], actions: List[int], epochs = 10, batch_size = 32):
        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.feature_columns()]
        y = keras.utils.to_categorical(df['action'], num_classes=6)
        x_scaled = self.preprocessor.fit_transform_data(x.values)
        early_stopping = EarlyStopping(monitor='val_loss', patience = 5)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only = True, monitor = 'val_loss')
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        history = self.model.fit(x_scaled, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, checkpoint, lr_scheduler])
        
        return history

    def predict(self, game_state: Dict):
        try:
            
            preprocessed_state = self.preprocessor.preprocess_game_state(game_state)
            scaled_state = self.preprocessor.transform_data(preprocessed_state)
            return self.model.predict(scaled_state)
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
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
    
    def lr_schedule(self, epoch):
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lr = initial_lr * (drop ** ((1 + epoch) // epochs_drop))
        
        return lr
    
    def save(self, filepath):
        """Save the entire neural network object"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model' : self.model,
                'preprocessor' : self.preprocessor
            }, f)

    def save_model(self, filepath):
        self.model.save(filepath)
    
    
    @classmethod
    def load(cls, filepath):
        """Load the entire neural network object"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['model'].input_shape[1:])
        instance.model = data['model']
        instance.preprocessor = data['preprocessor']
        return instance

    @classmethod
    def load_model(cls, filepath):
        loaded_model = keras.models.load_model(filepath)
        instance = cls(loaded_model.input_shape[1:])
        instance.model = loaded_model
        return instance
    
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label = 'Training Loss')
        plt.plot(history.history['val_loss'], label = 'Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label = 'Training Accuracy')
        plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, game_states: List[Dict]):
        importance = self.feature_importance(game_states)
        plt.figure(figsize=(10, 6))
        plt.bar(importance.keys(), importance.values())
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def cross_validate(self, game_states: List[Dict], actions: List[int], n_splits = 5):
        df = self.preprocessor.create_dataframe(game_states, actions)
        x = df[self.preprocessor.get_feature_columns()]
        y = df['action']
        
        kf = KFold (n_splits=n_splits, shuffle=True, random_state=42)
        
        scores = []
        
        for train_index, val_index in kf.split(x):
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            x_train_scaled = self.preprocessor.fit_transform_data(x_train.values)
            x_val_scaled = self.preprocessor.fit_transform_data(x_val.values)
            
            y_train_cat = keras.utils.to_categorical(y_train, num_classes = 6)
            y_val_cat = keras.utils.to_categorical(y_val, num_classes = 6)
            
            self.model = self._build_model(x_train.shape[1])
            self.model.fit(x_train_scaled, y_train_cat, epochs = 10, batch_size = 32, verbose = 0)
            
            score = self.model.evaluate(x_val_scaled, y_val_cat, verbose = 0)
            scores.append(score[1]) # append accuracy
            
        return np.mean(scores), np.std(scores)
    
    def augmented_data(self, game_states: List[Dict], actions: List[int], num_augmentations = 1):
        augmented_states = []
        augmented_actions = []
        for state, action in zip(game_states, actions):
            augmented_states.append(state)
            augmented_actions.append(action)
            
            for _ in range(num_augmentations):
                new_state = state.copy()
                
                new_state['pot_size'] *= np.random.uniform(0.9, 1.1)
                new_state['current_bet'] *= np.random.uniform(0.9, 1.1)
                new_state['player_chips'] *= np.random.uniform(0.9, 1.1)
                
                augmented_states.append(new_state)
                augmented_actions.append(action)
        
        return augmented_states, augmented_actions
    
    def train_with_augmentation(self, game_states: List[Dict], actions: List[int], num_augmentations = 1, epochs = 10, batch_size = 32):
        augmented_states, augmented_actions = self.augmented_data(game_states, actions, num_augmentations)
        return self.train(augmented_states, augmented_actions, epochs, batch_size)
    
    def set_model_architecture(self, architecture = 'default'):
        if architecture == 'default':
            self.model = self._build_model(self.model.input_shape[1:])
        elif architecture == 'wide':
            self.model = keras.Sequential([
                layers.Input(shape=self.model.input_shape[1:]),
                layers.Dense(128, activation = 'relu'),
                layers.Dense(128, activation = 'relu'),
                layers.Dense(6, activation = 'softmax'),
            ])
            
        elif architecture == 'deep':
            self.model = keras.Sequential([
                layers.Input(shape = self.model.input_shape[1:]),
                layers.Dense(64, activation = 'relu'),
                layers.Dense(64, activation = 'relu'),
                layers.Dense(64, activation = 'relu'),
                layers.Dense(64, activation = 'relu'),
                layers.Dense(6, activation = 'softmax')
            ])
            
        self.model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def visualize_model(self, filepath = 'model.png'):
        plot_model(self.model, to_file=filepath, show_shapes = True, show_layer_names = True)
        
    def model_summary(self):
        self.model.summary()
        
    def get_current_lr(self):
        return self.model.optimizer.lr.numpy()