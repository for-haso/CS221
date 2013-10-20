#!/usr/bin/env python
import random, util, collections

import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Manual problems

grader.addBasicPart('writeupValid', lambda : grader.requireIsValidPdf('writeup.pdf')) 

def testQ(f, V):
    mdp = util.NumberLineMDP()
    goldQ = {}
    values = [l.split() for l in open(f)]
    for state, action, value in values:
        goldQ[(int(state), int(action))] = float(value)
    for state in range(-5,6):
        for action in [-1,1]:
            if not grader.requireIsEqual(goldQ[(state, action)],
                                         submission.computeQ(mdp, V, state,
                                             action)):
                print '   state: {}, action: {}'.format(state, action)

def test1a_0():
    V = collections.defaultdict(lambda: 1)
    testQ('1a_0.gold', V)
grader.addBasicPart('1a-0', test1a_0)

def test1a_1():
    V = collections.Counter()  # state -> value of state
    for state in range(-5,6):
        V[state] = state
    testQ('1a_1.gold', V)
grader.addBasicPart('1a-1', test1a_1)

def test1b():
    V = collections.defaultdict(int)
    pi = collections.defaultdict(lambda: -1)
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    goldV = {}
    values = [l.split() for l in open('1b.gold')]
    for state, value in values:
        goldV[int(state)] = float(value)
    V = submission.policyEvaluation(mdp, V, pi, .0001)
    for state in range(-5,6):
        # print '{}\t{}'.format(state, V[state])
        if not grader.requireIsLessThan(.001, abs(goldV[state] - V[state])):
            print '   state: {}'.format(state)
grader.addBasicPart('1b', test1b)

def test1c():
    V = collections.Counter()  # state -> value of state
    for state in range(-5,6):
        V[state] = state
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    goldPi = collections.defaultdict(lambda: 1)
    pi = submission.computeOptimalPolicy(mdp, V)
    for state in range(-5,6):
        if not grader.requireIsEqual(goldPi[state], pi[state]):
            print '   state: {}'.format(state)
grader.addBasicPart('1c', test1c)

def testIteration(algorithm):
    mdp = util.NumberLineMDP()
    goldPi = collections.defaultdict(lambda: 1)
    goldV = {}
    values = [l.split() for l in open('1d.gold')]
    for state, value in values:
        goldV[int(state)] = float(value)
    algorithm.solve(mdp, .0001)
    for state in range(-5,6):
        # print '{}\t{}'.format(state, algorithm.V[state])
        if not grader.requireIsEqual(goldPi[state], algorithm.pi[state]):
            print '   action for state: {}'.format(state)
        if not grader.requireIsLessThan(.001, abs(goldV[state] - algorithm.V[state])):
            print '   value for state: {}'.format(state)

def test1d():
    testIteration(submission.PolicyIteration())
grader.addBasicPart('1d', test1d)

def test1e():
    testIteration(submission.ValueIteration())
grader.addBasicPart('1e', test1e)

def test2a():
    mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    startState = mdp.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, (0,))
    tests = [([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)],
              startState, 'Take'),
             ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)],
              startState, 'Peek'),
             ([((0, None, (0,)), 1, 0)], startState, 'Quit'),
             ([((7, None, (0, 1)), 0.5, 0), ((11, None, (0,)), 0.5, 0)],
              preBustState, 'Take'),
             ([], postBustState, 'Take'),
             ([], postBustState, 'Peek'),
             ([], postBustState, 'Quit')]
    for gold, state, action in tests:
        if not grader.requireIsEqual(gold,
                                     mdp.succAndProbReward(state, action)):
            print '   state: {}, action: {}'.format(state, action)
grader.addBasicPart('2a', test2a)

def test2b():
    mdp = submission.peekingMDP()
    vi = submission.ValueIteration()
    vi.solve(mdp)
    grader.requireIsEqual(mdp.threshold, 20)
    grader.requireIsEqual(mdp.peekCost, 1)
    f = len([a for a in vi.pi.values() if a == 'Peek']) / float(len(vi.pi.values()))
    grader.requireIsGreaterThan(.1, f)
grader.addBasicPart('2b', test2b)

def testQLearning(featureExtractor, goldFile):
    random.seed(3)
    mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       featureExtractor,
                                       0.2)
    util.simulate(mdp, rl, 1, sort=True)
    values = [l.strip().split('\t') for l in open(goldFile)]
    i = 0
    for state in sorted(mdp.states):
        for action in sorted(rl.actions(state)):
            _, _, goldValue = values[i]
            if not grader.requireIsEqual(float(goldValue),
                                         rl.getQ(state, action)):
                print '   state: {}, action: {}'.format(state, action)
            i += 1

def test3a():
    testQLearning(submission.identityFeatureExtractor, '3a.gold')
grader.addBasicPart('3a', test3a, maxSeconds=10)

grader.addManualPart('3b', 1)

def test3c():
    testQLearning(submission.blackjackFeatureExtractor, '3c.gold')
grader.addBasicPart('3c', test3c, maxSeconds=10)

grader.addManualPart('3d', 1)

grader.grade()
