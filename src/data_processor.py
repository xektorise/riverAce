import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

class DataProcessor:
    
    @staticmethod
    def preprocess(data):
        """
        Preprocess the input data.
        
        Parameters:
        data (list or dict): The input data to preprocess.
        
        Returns:
        numpy.ndarray: The preprocessed data as a NumPy array.
        """
        df = pd.DataFrame(data)
        
        # handling missing values
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # encoding categorical data (example for a categorical column named 'category')
        if 'category' in df.columns:
            encoder = OneHotEncoder(sparse=False)
            encoded_cols = encoder.fit_transform(df[['category']])
            encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(['category']))
            df = df.drop('category', axis=1).join(encoded_df)
        
        # normalization/Standardization
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        # additional custom feature engineering steps can be added here
        
        return df.values
    
    @staticmethod
    def load_data(filepath):
        """
        Load data from a CSV file.
        
        Parameters:
        filepath (str): The path to the CSV file.
        
        Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
        """
        return pd.read_csv(filepath)
    
    @staticmethod
    def split_data(df, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        Parameters:
        df (pandas.DataFrame): The data to split.
        test_size (float): The proportion of the dataset to include in the test split.
        
        Returns:
        tuple: (X_train, X_test, y_train, y_test)
        """
        X = df.drop('target', axis=1)
        y = df['target']
        return train_test_split(X, y, test_size=test_size, random_state=42)
