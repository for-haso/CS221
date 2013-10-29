#!/usr/bin/env python
"""
Grader for backgammon assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""


import random, math
import game, agent, submission
import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')


def makeTestGame1():
    layout = "0-2-x,2-2-o,4-2-o,8-2-x,7-2-o,9-1-x,12-2-o,15-3-x"
    g = game.Game(layout)
    g.new_game()
    return g

def makeTestGame2():
    layout = "0-2-x,2-2-o,4-2-o,8-2-x,7-2-o,9-1-x,12-2-o,15-3-x"
    g = game.Game(layout)
    g.new_game()
    g.grid[12].pop()
    g.grid[15].pop()
    g.offPieces['o'].append('o')
    g.barPieces['x'].append('x')
    return g


############################################################
# Problem 1a: simpleEvaluation
g1 = makeTestGame1()
g2 = makeTestGame2()

g1r = g1.clone()
g1r.reverse()
g2r = g2.clone()
g2r.reverse()

grader.addBasicPart('1a-0',lambda : grader.requireIsEqual(2.0,submission.simpleEvaluation((g1r,'o'))))
grader.addBasicPart('1a-1',lambda : grader.requireIsEqual(2.0,submission.simpleEvaluation((g1,'o'))))
grader.addBasicPart('1a-2',lambda : grader.requireIsEqual(1.9,submission.simpleEvaluation((g2r,'o'))))
grader.addBasicPart('1a-3',lambda : grader.requireIsEqual(11.0,submission.simpleEvaluation((g2,'o'))))


def evalFn(state,evalArgs=None):
    g,player = state
    return len(g.grid[5])

############################################################
# Problem 1b: ReflexAgent
reflexAgent = submission.ReflexAgent(game.Game.TOKENS[0],evalFn,None)
grader.addBasicPart('1b-0',lambda : grader.requireIsEqual(((2, 5), (2, 5)),reflexAgent.getAction(g1.getActions((3,3),'o'),g1)),3)


############################################################
# Problem 1c: ExpectimaxAgent
expectimaxAgent = submission.ExpectimaxAgent(game.Game.TOKENS[0],evalFn,None)
grader.addBasicPart('1c-0',lambda : grader.requireIsEqual(((7, 12), (7, 11)),expectimaxAgent.getAction(g1.getActions((5,4),'o'),g1)),4,5)


############################################################
# Problem 2a: extractFeatures
feats1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
feats2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
feats3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
feats4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
grader.addBasicPart('2a-0',lambda : grader.requireIsEqual(feats1,submission.extractFeatures((g1,'o'))))
grader.addBasicPart('2a-1',lambda : grader.requireIsEqual(feats2,submission.extractFeatures((g1,'x'))))
grader.addBasicPart('2a-2',lambda : grader.requireIsEqual(feats3,submission.extractFeatures((g2,'o'))))
grader.addBasicPart('2a-3',lambda : grader.requireIsEqual(feats4,submission.extractFeatures((g2,'x'))))


############################################################
# Problem 2b: logLinearEvaluation
weights = [math.exp(-math.sqrt(i)) for i in range(103)]
grader.addBasicPart('2b-0',lambda : grader.requireIsEqual(0.82160499,submission.logLinearEvaluation((g1,'o'),list(reversed(weights)))))
grader.addBasicPart('2b-1',lambda : grader.requireIsEqual(0.839164,submission.logLinearEvaluation((g1,'x'),list(reversed(weights)))))
grader.addBasicPart('2b-2',lambda : grader.requireIsEqual(0.559755,submission.logLinearEvaluation((g2,'o'),weights)))
grader.addBasicPart('2b-3',lambda : grader.requireIsEqual(0.84293,submission.logLinearEvaluation((g2,'x'),list(reversed(weights)))))



############################################################
# Problem 2c: TDUpdate
tdupdates = [0.082037,0.073124,0.047990,0.054536,0.017489,0.054536,0.048155,0.073124,0.075518,0.053519,0.055350,0.048155,0.073124,0.075518,0.048155,0.064538,0.074391,0.017489,0.053519,0.080386,0.055350,0.082844,0.082844,0.074391,0.064538,0.047990,0.081973,0.080386,1.16962,-1.29402]
def basicTDTest():
    it = 0
    for a in g1.getActions((2,3),'o'):
        weights = [math.exp(-math.sqrt(i)) for i in range(103)]
        tmpg = g1.clone()
        tmpg.takeAction(a,'o')
        grader.requireIsEqual(tdupdates[it],submission.TDUpdate((g1,'o'),(tmpg,'x'),0,weights,10)[6])
        it += 1
    weights = [math.exp(-math.sqrt(i)) for i in range(103)]
    grader.requireIsEqual(tdupdates[it],submission.TDUpdate((g1,'o'),None,1,weights,10)[6])
    weights = [math.exp(-math.sqrt(i)) for i in range(103)]
    grader.requireIsEqual(tdupdates[it+1],submission.TDUpdate((g1,'o'),None,0,weights,10)[6])
grader.addBasicPart('2c-0',basicTDTest,4)



grader.grade()
