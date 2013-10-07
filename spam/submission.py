"""
CS221 2013
AssignmentID: spam
"""

import util
import operator
from collections import Counter

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.blacklist = set(blacklist[:k])
        self.n = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        num_spam_words = 0
        words = text.split()
        for word in words:
            if word in self.blacklist:
                num_spam_words += 1
        return 1.0 if num_spam_words >= self.n else -1.0
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    feature = dict()
    for word in x.split():
        if word in feature:
            feature[word] += 1.0
        else:
            feature[word] = 1.0
    return feature
    # END_YOUR_CODE

# Taken from warmup assignment
def dotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as dicts, 
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    """
    # v1 <= v2 in all cases
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum([v1[k] * v2[k] for k in v1 if k in v2])


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        feature = self.featureFunction(x)
        return dotProduct(feature, self.params)
        # END_YOUR_CODE

# From warmup assignment
def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    """
    for k in v2:
        if k in v1:
            v1[k] += scale * v2[k]
        else:
            v1[k] = scale * v2[k]

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('positive', 'negative'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    weights = dict()
    for i in range(0, iters):
        for train in trainExamples:
            features = featureExtractor(train[0])
            score = dotProduct(weights, features)
            actual_result = 1.0 if train[1] == labels[0] else -1.0
            margin = score * actual_result
            if margin <= 0:
                incrementSparseVector(weights, actual_result, features)
    return weights
    # END_YOUR_CODE

def isEndPunctuation(c):
    if (c == "!" or c == "?" or c == "."):
        return True
    return False

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    words = x.split()
    feature = dict()
    prev_word = "-BEGIN-"
    for word in words:
        if isEndPunctuation(prev_word):
            prev_word = "-BEGIN-"

        if word in feature:
            feature[word] += 1.0
        else:
            feature[word] = 1.0

        word_pair = prev_word + " " + word
        if word_pair in feature:
            feature[word_pair] += 1.0
        else:
            feature[word_pair] = 1.0

        prev_word = word
    return feature
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); 
        each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.labels = labels
        self.classifiers = classifiers
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        scores = self.classify(x)
        return max(scores, key=lambda score:score[1])[0]
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); 
        the classifier is the one-vs-all classifier
        """
        print "create one all classifier"
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        scores = list()
        for classifier in self.classifiers:
            score = classifier[1].classify(x)
            scores.append((classifier[0], score))
        return scores
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    print "learn classifiers"
    classifiers = list()
    for label in labels:
        weights = learnWeightsFromPerceptron(trainExamples, featureFunction, (label, ""), perClassifierIters)
        classifier = WeightedClassifier((label, ""), featureFunction, weights)
        classifiers.append((label, classifier))
    return classifiers
    # END_YOUR_CODE
