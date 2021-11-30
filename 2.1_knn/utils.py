import pandas as pd
import random as rnd

def read_data(data_path, target_column):
    df = pd.read_csv(data_path)
    X = df.loc[:, df.columns != target_column]
    Y = df['target']
    return X.values, Y.values

def shuffle(X, Y):
    num_data = X.shape[0]
    mask = list(range(num_data))
    rnd.shuffle(mask)
    new_X = X[mask, :]
    new_Y = Y[mask]
    return new_X, new_Y

if __name__ == "__main__":
    
    X, Y = read_data('./data/heart.csv', 'target')
    X, Y = shuffle(X, Y)
    print(X.shape, Y.shape)
