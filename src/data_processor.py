import pandas as pd

class DataProcessor:
    
    def preprocess(data):
        
        df = pd.DataFrame(data)
        
        return df.values
    
    
    def load_data(filepath):
        return pd.read_csv(filepath)
    