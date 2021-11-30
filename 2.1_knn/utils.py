import numpy as np
import pandas as pd
import random as rnd

def read_data(data_path, target_column):
    df = pd.read_csv(data_path)
    X = df.loc[:, df.columns != target_column]
    Y = df[target_column]
    return X.values, Y.values

def shuffle(X, Y):
    num_data = X.shape[0]
    mask = list(range(num_data))
    rnd.shuffle(mask)
    new_X = X[mask, :]
    new_Y = Y[mask]
    return new_X, new_Y

def split_data(X, Y, split_fraction):
    num_total = X.shape[0]
    num_train = int(num_total * split_fraction)
    X_train, Y_train = X[:num_train, :], Y[:num_train]
    X_val, Y_val = X[num_train:, :], Y[num_train:]
    return X_train, Y_train, X_val, Y_val

def compute_accuracy(targets, preds):
    num_data = targets.shape[0]
    accuracy = np.sum(targets == preds) / num_data
    return accuracy

def compute_confusion_matrix(targets, preds):
    TP = np.sum(targets * preds)
    FP = np.sum((1-targets) * preds)
    TN = np.sum((1-targets) * (1-targets))
    FN = np.sum(targets * (1-preds))
    return TP, FP, TN, FN

def report_classification(targets, preds):
    accuracy = compute_accuracy(targets, preds)
    TP, FP, TN, FN = compute_confusion_matrix(targets, preds)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    specificity = TN/(TN + FP)
    f1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, specificity, f1

if __name__ == "__main__":
    
    X, Y = read_data('./data/heart.csv', 'target')
    X, Y = shuffle(X, Y)
    X_train, Y_train, X_val, Y_val = split_data(X, Y, 0.8)
    print(X.shape, Y.shape)
