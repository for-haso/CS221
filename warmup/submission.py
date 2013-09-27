import collections

############################################################
# Nisha Masharani (nisham)
# CS221 warmup assignment

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet.  You might find
    max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return max(text.split(' '), key=len)
    # END_YOUR_CODE

############################################################
# Problem 3b

def createExistsFunction(text):
    """
    Given a text, return a function f, where f(word) returns whether |word|
    occurs in |text| or not.  f should run in O(1) time.  You might find it
    useful to use set().
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    word_set = set(text.split(' '))
    def f(word):
        return word in word_set
    return f
    # END_YOUR_CODE

############################################################
# Problem 3c

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where locations are
    pairs (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return sum([abs(loc1[x] - loc2[x]) for x in range(len(loc1))])
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as Counters, return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    item_list = v1.most_common() if len(v1) > len(v2) else v2.most_common()
    counter = v2 if len(v1) > len(v2) else v1
    return sum([x[1] * counter[x[0]] for x in item_list])
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    v2_list = v2.most_common()
    for x in v2_list:
        v1[x[0]] += x[1] * scale
    # END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
        the set of words that occur the maximum number of times, and
	their count, i.e.
	(set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.Counter().
    """
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    counter = collections.Counter(text.split())
    most_common = counter.most_common()
    if len(most_common) < 1:
        return set([]), 0
    max_count = most_common[0][1]
    most_common_set = set([x[0] for x in most_common if x[1] == max_count])
    return most_common_set, max_count
    # END_YOUR_CODE
