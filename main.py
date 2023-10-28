import pandas as pd
import numpy as np
from Perceptron_Class import PrecptronModel

def train_test_split(test_size: float, df: pd.DataFrame, random_state=None) -> (pd.DataFrame, pd.DataFrame):
    """
    spliting the dataset to train 80% and test 20% before fiting the model
    :param test_size:
    :param df: the function gets DataFrame and takes random X% for train and the rest gives for test
    :param random_state:
    :return: (df,df)
    """
    n = len(df)
    indices = np.arange(n)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)
    test_idx = int(n * test_size)
    train_idx = indices[test_idx:]
    test_idx = indices[:test_idx]
    return df.iloc[train_idx], df.iloc[test_idx]

def main_my_data_set():
    """
        the fucntion creates the DATASET from section A in the assigment and activate on it the perception algorithem
        :return: none
    """
    p = PrecptronModel()
    X = np.array([[-2, -1], [0, 0], [2, 1], [1, 2], [-2, 2], [-3, 0]])
    y = np.array([-1, 1, 1, 1, -1, -1])
    p.fit(X, y)

    print("score:", p.score(X, y))
    print("weight vector:", np.squeeze(p._weight))
    print("biase :", p.b)

def main_given_data_set():
    """
    the fucntion gets the DATASET we have been given in the assigment and activate on it the perception algorithem
    :return: none
    """
    df = pd.read_csv('dataset.csv')
    df['diagnosis'] = df['diagnosis'].replace(0,-1)

    train_set, test_set = train_test_split(0.2, df)
    train_y = train_set['diagnosis'].values
    train_x = train_set.drop(['diagnosis'], axis=1).values

    p = PrecptronModel()

    test_y = test_set['diagnosis'].values
    test_x = test_set.drop(['diagnosis'], axis=1).values
    p.fit(train_x, train_y)

    print(f'Train Error =', 1.0 - p.score(train_x, train_y))
    print(f'Test Error =', 1.0 - p.score(test_x, test_y))


if __name__ == '__main__':
    print("perceptron algorithm on the data set from section A:\n")
    main_my_data_set()
    print("\n")
    print("perceptron algorithm on the data set of Processed Wisconsin Diagnostic Breast Cancer - from section E:\n")
    main_given_data_set()
