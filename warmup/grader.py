#!/usr/bin/env python

import graderUtil, collections, random

grader = graderUtil.Grader()
submission = grader.load('submission')

############################################################
# Problems 1 and 2

grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf'))


############################################################
# Problem 3a: computeMaxWordLength

grader.addBasicPart('3a-0', lambda :
        grader.requireIsEqual('longest', submission.computeMaxWordLength('which is the longest word')))


############################################################
# Problem 3b: createExistsFunction

def test():
    func = submission.createExistsFunction('the quick brown fox jumps over the lazy fox')
    grader.requireIsEqual(True, func('lazy'))
    grader.requireIsEqual(False, func('laz'))
grader.addBasicPart('3b-0', test)


############################################################
# Problem 3c: manhattanDistance

grader.addBasicPart('3c-0', lambda : grader.requireIsEqual(6, submission.manhattanDistance((3, 5), (1, 9))))


############################################################
# Problem 3d: dotProduct

grader.addBasicPart('3d-0', lambda : grader.requireIsEqual(15, submission.sparseVectorDotProduct(collections.Counter({'a': 5}), collections.Counter({'b': 2, 'a': 3}))))


############################################################
# Problem 3e: incrementSparseVector

def test():
    v = collections.Counter({'a': 5})
    submission.incrementSparseVector(v, 2, collections.Counter({'b': 2, 'a': 3}))
    grader.requireIsEqual(collections.Counter({'a': 11, 'b': 4}), v)
grader.addBasicPart('3e-0', test)


############################################################
# Problem 3f

def test():
    grader.requireIsEqual((set(['the', 'fox']), 2), submission.computeMostFrequentWord('the quick brown fox jumps over the lazy fox'))
grader.addBasicPart('3f-0', test)

def test():
	f = open('3f_test', 'r')
	text = f.read()
	grader.requireIsEqual((set(['the', 'of']), 15), submission.computeMostFrequentWord(text))
grader.addBasicPart('3f-1', test)

def test():
	grader.requireIsEqual((set([]), 0), submission.computeMostFrequentWord(""))
grader.addBasicPart('3f-2', test)

def test():
	f = open('3f_test2', 'r')
	text = f.read()
	grader.requireIsEqual((set(['hello', 'goodbye']), 15), submission.computeMostFrequentWord(text))
grader.addBasicPart('3f-3', test)

grader.grade()
