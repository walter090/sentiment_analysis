import numpy as np
import time
import math
from collections import defaultdict
from learner.scorer import Scorer


class NaiveBayesClassifier(object):
    def __init__(self, prior=None):
        """

        :param prior: domain knowledge if prior is set,
         it will not be updated during training
        """
        self.count = defaultdict(dict)
        self.prior = prior
        self.all = set()  # all unique words, type: string

    def train(self, X, y, verbose=False):
        """

        :param X: array of dictionaries of {string: int}
        :param y: class label
        :param verbose:
        :return: a trained instance of self
        """
        start = time.time()
        prior_count = {}
        for i, label in enumerate(y):
            if label in prior_count.keys():
                prior_count[label] += 1
            else:
                prior_count[label] = 1

            for key, value in X[i].items():
                self.all.add(key)
                if key in self.count[label]:
                    # if the key has already appeared, add the count
                    self.count[label][key] += value
                else:
                    # if the key is new, add the key and its count
                    self.count[label][key] = value

        if self.prior is None:
            self.prior = {}
            for key, value in prior_count.items():
                self.prior[key] = value / len(y)

        end = time.time()
        if verbose:
            print('Training time: {} sec'.format(start - end))

        return self

    def predict(self, X):
        """

        :param X: a dictionary of {string: int}
        :return:
        """
        predictions = []
        for entry in X:
            posterior = {}
            for label in self.prior.keys():
                p_given_label = 0.
                for key in entry:
                    if key in self.count[label]:
                        p_given_label += math.log2(
                            (self.count[label][key] + 1) / (len(self.count[label]) + 1 + len(self.all)))
                    else:
                        p_given_label += math.log2(1 / (len(self.count[label]) + len(self.all)))

                posterior[label] = math.log2(self.prior[label]) + p_given_label
            predictions.append(max(posterior, key=posterior.get))
        return predictions

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
