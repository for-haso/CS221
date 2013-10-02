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
grader.addBasicPart('3a-1', lambda :
        grader.requireIsEqual('', submission.computeMaxWordLength(' ')))
grader.addBasicPart('3a-2', lambda :
        grader.requireIsEqual('', submission.computeMaxWordLength('')))
grader.addBasicPart('3a-3', lambda :
        grader.requireIsEqual('zzz', submission.computeMaxWordLength('aaa Zzz zzz bbb')))


############################################################
# Problem 3b: createExistsFunction

def test():
    func = submission.createExistsFunction('the quick brown fox jumps over the lazy fox')
    grader.requireIsEqual(True, func('lazy'))
    grader.requireIsEqual(False, func('laz'))
grader.addBasicPart('3b-0', test)
def test2():
    f = open('3f_test', 'r')
    text = f.read()
    func = submission.createExistsFunction(text)
    grader.requireIsEqual(True, func('calamity'))
    grader.requireIsEqual(True, func('fardels'))
    grader.requireIsEqual(False, func('homie'))
    grader.requireIsEqual(False, func('calamity '))
    grader.requireIsEqual(False, func(''))
    grader.requireIsEqual(False, func(' '))
grader.addBasicPart('3b-1', test2)


############################################################
# Problem 3c: manhattanDistance

grader.addBasicPart('3c-0', lambda : grader.requireIsEqual(6, submission.manhattanDistance((3, 5), (1, 9))))
grader.addBasicPart('3c-1', lambda : grader.requireIsEqual(8, submission.manhattanDistance((1, 4, 3), (3, 2, 7))))
grader.addBasicPart('3c-2', lambda : grader.requireIsEqual(0, submission.manhattanDistance((1, 1), (1, 1))))

############################################################
# Problem 3d: dotProduct

grader.addBasicPart('3d-0', lambda : grader.requireIsEqual(15, submission.sparseVectorDotProduct(collections.Counter({'a': 5}), collections.Counter({'b': 2, 'a': 3}))))
grader.addBasicPart('3d-1', lambda : grader.requireIsEqual(20, submission.sparseVectorDotProduct(collections.Counter({'a': 5, 'c': 5}), collections.Counter({'b': 2, 'a': 3, 'c':1}))))
grader.addBasicPart('3d-2', lambda : grader.requireIsEqual(25, submission.sparseVectorDotProduct(collections.Counter({'a': 5, 'b': 5}), collections.Counter({'b': 2, 'a': 3}))))
grader.addBasicPart('3d-3', lambda : grader.requireIsEqual(0, submission.sparseVectorDotProduct(collections.Counter({}), collections.Counter({}))))

############################################################
# Problem 3e: incrementSparseVector

def test():
    v = collections.Counter({'a': 5})
    submission.incrementSparseVector(v, 2, collections.Counter({'b': 2, 'a': 3}))
    grader.requireIsEqual(collections.Counter({'a': 11, 'b': 4}), v)
grader.addBasicPart('3e-0', test)
def test():
    v = collections.Counter({'a': 5, 'b': 10, 'c': 3})
    submission.incrementSparseVector(v, 2, collections.Counter({'b': 2, 'a': 3}))
    grader.requireIsEqual(collections.Counter({'a': 11, 'b': 14, 'c': 3}), v)
grader.addBasicPart('3e-1', test)
def test():
    v = collections.Counter({})
    submission.incrementSparseVector(v, 2, collections.Counter({'b': 2, 'a': 3}))
    grader.requireIsEqual(collections.Counter({'a': 6, 'b': 4}), v)
grader.addBasicPart('3e-2', test)
def test():
    v = collections.Counter({})
    submission.incrementSparseVector(v, 2, collections.Counter({}))
    grader.requireIsEqual(collections.Counter({}), v)
grader.addBasicPart('3e-3', test)


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
