class Scorer(object):
    @staticmethod
    def get_accuracy(y, predictions):
        correct = [predictions[i] for i in range(len(predictions))
                   if predictions[i] == y[i]]

        return len(correct) / len(predictions)

    @staticmethod
    def get_precision(y, predictions):
        true_positive = [predictions[i] for i in range(len(predictions))
                         if predictions[i] == y[i] == 1]
        positive = [predictions[i] for i in range(len(predictions)) if predictions[i] == 1]

        return len(true_positive) / len(positive)

    @staticmethod
    def get_recall(y, predictions):
        true_positive = [predictions[i] for i in range(len(predictions))
                         if predictions[i] == y[i] == 1]
        false_negative = [predictions[i] for i in range(len(predictions))
                          if predictions[i] == -1 != y[i]]

        return len(true_positive) / (len(true_positive) + len(false_negative))

    def get_fbeta(self, y, predictions, beta):
        precision = self.get_precision(y, predictions)
        recall = self.get_recall(y, predictions)
        f_score = (1 + beta ** 2.) * (precision * recall) / (beta ** 2 * precision + recall)

        return f_score

    def cross_validate(self, X, y, train, n=5, beta=1.0, verbose=False):
        """
        cross validation from the model
        :param verbose:
        :param beta: parameter for calculating f beta score,
               beta < 1 favors precision, beta > 0 favors recall
        :param train: training function
        :param X:
        :param y:
        :param n: n folds, type: int
        :return: an array of dictionaries of cross validation information
        """
        import numpy as np

        size = len(y) // n
        average_score = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f-beta': [],
        }
        for i in range(0, len(y), size):
            if verbose:
                print('Completed part {}'.format(i + 1))
            train_x = X[: i] + X[i + size:]
            train_y = y[: i] + y[i + size:]
            test_x = X[i: i + size]
            test_y = y[i: i + size]
            clf = train(train_x, train_y)
            predictions = clf.predict(test_x)

            average_score['accuracy'].append(self.get_accuracy(test_y, predictions))
            average_score['precision'].append(self.get_precision(test_y, predictions))
            average_score['recall'].append(self.get_recall(test_y, predictions))
            average_score['f-beta'].append(self.get_fbeta(test_y, predictions, beta))

        for key, value in average_score.items():
            average_score[key] = np.mean(value)

        return average_score
