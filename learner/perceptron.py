import numpy as np


class Perceptron:

    def __init__(self, epochs=5, shuffle=False):
        self.epochs = epochs
        self.shuffle = shuffle

        self.weights_ = []

    @staticmethod
    def epoch_error(y, predictions):
        return np.sum(y - predictions) / float(len(y))

    def train(self, X, y, alpha=0.1, verbose=False):
        """
        fit the model
        :param verbose: print epoch training information if set to true, default false
        :param X: features
        :param y: target labels
        :param alpha: learning rate
        :return: return an instance of the object
        """
        weights = np.zeros(X.shape[1])

        for i in range(self.epochs):
            predictions = []
            for j, entry in enumerate(X):
                y_hat = np.dot(weights, entry)
                predictions.append(y_hat)
                error = y[j] - y_hat  # find error
                weights = [coef + alpha * error * entry for coef in weights]  # update weights
            if verbose:
                print('=============================================================')
                print('                         Epoch{}                            '.format(i))
                print('=============================================================')
                print('Weights: {}').format(weights)
                print('Epoch error: {}').format(self.epoch_error(y, predictions).round(3))

        self.weights_ = weights

        return self

    def predict(self, X):
        """
        predict using a trained model
        :param X: array like, (n_samples, n_features)
        :return: returns an array of the predictions
        """
        return [np.dot(self.weights_, entry) for entry in X]
