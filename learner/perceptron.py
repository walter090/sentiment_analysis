import numpy as np
from learner.scorer import Scorer


class Perceptron(object):

    def __init__(self, epochs=3):
        self.epochs = epochs
        self.weights_ = []

    @staticmethod
    def epoch_error(errors, y):
        return np.sum(errors) / len(y)

    def train(self, X, y, alpha=0.1, verbose=False):
        """
        fit the model
        :param verbose: print epoch training information if set to true, default false
        :param X: features in dicts
        :param y: target labels
        :param alpha: learning rate
        :return: return an instance of the object
        """
        self.weights_ = np.zeros(len(X[0]))

        for i in range(self.epochs):
            predictions = []
            errors = []
            for j, entry in enumerate(X):
                y_hat = 1 if np.dot(self.weights_, entry) > 0 else -1
                predictions.append(y_hat)
                error = y[j] - y_hat  # find error
                errors.append(error)

                for k in range(0, len(self.weights_)):
                    self.weights_[k] += alpha * error * entry[k]

                # update weights
            if verbose:
                print('=============================================================')
                print('                         Epoch{}                            '.format(i))
                print('=============================================================')
                print('Epoch error (training): {}'.format(self.epoch_error(errors, y).round(3)))

        return self

    def predict(self, X):
        """
        predict using a trained model
        :param X: array like, (n_samples, n_features)
        :return: returns an array of the predictions
        """
        return [1 if np.dot(self.weights_, entry) > 0 else -1 for entry in X]

    def score(self, X, y, metric='cv', beta=1.0, n=5):
        """
        :param n:
        :param X: feature values
        :param y: target label
        :param metric: the metric used to score the model, default accuracy
                       if 'dump' is selected return a dictionary of all metrics
        :param * beta: set beta for f-beta score if fbeta score is selected
                       beta < 1 favors precision, beta > 0 favors recall
        :return: return the score for the model
        """
        validator = Scorer()

        if metric == 'cv':
            return validator.cross_validate(X, y, self.train, n, beta)

        predictions = self.predict(X)

        metric_options = {
            'accuracy': validator.get_accuracy(y, predictions),
            'precision': validator.get_precision(y, predictions),
            'recall': validator.get_recall(y, predictions),
            'f-beta': validator.get_fbeta(y, predictions, beta),
        }

        if metric == 'dump':
            return metric_options

        return metric_options[metric]
