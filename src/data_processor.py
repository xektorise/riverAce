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
        
        # handle missing values
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # encode categorical data
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            encoder = OneHotEncoder(sparse=False)
            encoded_cols = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
            df = df.drop(categorical_cols, axis=1).join(encoded_df)
        
        # normalize/Standardize
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
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
        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            raise Exception(f"The file at {filepath} was not found.")
        except pd.errors.ParserError:
            raise Exception(f"Error parsing the file at {filepath}.")
    
    @staticmethod
    def split_data(df, target_column='target', test_size=0.2):
        """
        Split the data into training and testing sets.
        
        Parameters:
        df (pandas.DataFrame): The data to split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        
        Returns:
        tuple: (X_train, X_test, y_train, y_test)
        """
        if target_column not in df.columns:
            raise Exception(f"Target column '{target_column}' not found in DataFrame.")
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)