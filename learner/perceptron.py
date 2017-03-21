import numpy as np


class Perceptron(object):

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
        return [1 if np.dot(self.weights_, entry) > 0 else 0 for entry in X]

    @staticmethod
    def get_accuracy(y, predictions):
        correct = [predictions[i] for i in range(len(predictions))
                   if predictions[i] == y[i]]

        return len(correct) / float(len(predictions))

    @staticmethod
    def get_precision(y, predictions):
        true_positive = [predictions[i] for i in range(len(predictions))
                         if predictions[i] == y[i] == 1]
        positive = [predictions[i] for i in range(len(predictions)) if predictions[i] == 1]

        return len(true_positive) / float(len(positive))

    @staticmethod
    def get_recall(y, predictions):
        true_positive = [predictions[i] for i in range(len(predictions))
                         if predictions[i] == y[i] == 1]
        false_negative = [predictions[i] for i in range(len(predictions))
                          if predictions[i] == 0 != y[i]]

        return len(true_positive) / float(len(true_positive) + len(false_negative))

    def get_fbeta(self, y, predictions, beta):
        precision = self.get_precision(y, predictions)
        recall = self.get_recall(y, predictions)
        f_score = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

        return f_score

    def score(self, X, y, metric='accuracy', beta=1.0):
        """
        :param X: feature values
        :param y: target label
        :param metric: the metric used to score the model, default accuracy
        :param * beta: set beta for f-beta score if fbeta score is selected
                       beta < 1 favors precision, beta > 0 favors recall
        :return: return the score for the model
        """
        predictions = self.predict(X)
        metric_options = {
            'accuracy': self.get_accuracy(y, predictions),
            'precision': self.get_precision(y, predictions),
            'recall': self.get_recall(y, predictions),
            'fbeta': self.get_fbeta(y, predictions, beta),
        }

        return metric_options[metric]
