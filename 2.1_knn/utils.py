import pandas as pd

def read_data(data_path, target_column):
    df = pd.read_csv(data_path)
    X = df.loc[:, df.columns != target_column]
    Y = df['target']
    return X, Y

if __name__ == "__main__":
    
    read_data('./data/heart.csv', 'target')