import numpy as np
from typing import Union

class PrecptronModel:
    '''
    The Perceptron algorithm is a binary classification algorithm that is used to classify data into two classes.
    The algorithm learns a decision boundary that separates two classes by iteratively adjusting the weights and bias of the linear model.
    '''
    def __init__(self, max_epochs=1000):
        self.max_epochs = max_epochs
        self.b = 0
        self._weight: Union[np.array, None] = None

    def fit(self, X, y):
        '''
       The fit method is responsible for training the model on the given data.
       In the case of the perceptron algorithm, the fit method updates the weights of the model using the training data.
        '''
        num_features = X.shape[1]
        self._weight = np.zeros(num_features)
        self.b = 0
        error_found = True

        epoch =0
        while epoch in range(self.max_epochs) and error_found:
            error_found = False
            for i in range(X.shape[0]):
                x_i = X[i]
                y_i = y[i]

                if (
                        (np.dot(self._weight, x_i) + self.b) < 0 and y_i > 0
                ) or (
                        (np.dot(self._weight, x_i) + self.b) >= 0 and y_i < 0
                ):
                    self._weight += y_i * x_i
                    self.b += y_i
                    error_found = True
            epoch+=1

    def predict(self, X):
        ''' The predict method in the takes in an input feature matrix X and returns predicted labels using the perceptron algorithm.'''
        y = np.dot(X, self._weight) + self.b
        yhat = np.where(y>=0, 1, -1)
        return yhat

    def score(self, X, y):
        ''' The score method returns the accuracy of the model on the given input X and y.
        It compares the predicted outputs y_hat with the actual outputs y and returns the ratio of the number of correct
        predictions to the total number of predictions.  '''
        yhat = self.predict(X)
        errors = np.sum(yhat == y) / len(y)
        return errors


