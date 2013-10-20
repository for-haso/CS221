import graderUtil, util

grader = graderUtil.Grader()
submission = grader.load('submission')


############################################################
# Problem 1a: ucsTestCase

def testUCSTestCase(n):
    # Just make sure the test case is valid
    ucs = util.UniformCostSearch()
    ucs.solve(submission.createUCSTestCase(n))
    if ucs.actions and len(ucs.actions) <= 2 and ucs.numStatesExplored >= n:
        grader.assignFullCredit()
    else:
        grader.fail("Your test case did not meet the specifications")

grader.addBasicPart('1a-0', lambda : testUCSTestCase(3))


############################################################
# Problem 1b: astarReduction

def testZeroHeuristic():
    # Make sure putting the zero heuristic in doesn't change the problem.
    problem1 = util.trivialProblem
    problem2 = submission.astarReduction(problem1, lambda state : 0)
    grader.requireIsEqual(problem1.startState(), problem2.startState())
    for state in ['A', 'B', 'C']:
        if not grader.requireIsEqual(problem1.isGoal(state), problem2.isGoal(state)): return
        if not grader.requireIsEqual(problem1.succAndCost(state), problem2.succAndCost(state)): return
grader.addBasicPart('1b-0', testZeroHeuristic)



############################################################
# Problem 2b

def testDelivery():
    problem = submission.DeliveryProblem(util.deliveryScenario2)
    algorithm = util.UniformCostSearch()
    algorithm.solve(problem)
    grader.requireIsEqual(25, algorithm.totalCost)
grader.addBasicPart('2b-0', testDelivery)


############################################################
# Problem 2c

def testHeuristic1():
    scenario = util.deliveryScenario2
    problem = submission.DeliveryProblem(scenario)
    algorithm = submission.AStarSearch(submission.createHeuristic1(scenario))
    algorithm.solve(problem)
    if algorithm.totalCost != 25:
        grader.fail("heuristic1 produces wrong total cost")
        return
    # This is a coarse check, report your exact number of explored nodes in writeup
    if algorithm.numStatesExplored >= 61:
        grader.fail("heuristic1 explores too many states")
        return
    grader.assignFullCredit()
    print "numStatesExplored: ", algorithm.numStatesExplored
    print "totalCost: ", algorithm.totalCost
grader.addBasicPart('2c-0', testHeuristic1)

############################################################
# Problem 2d

def testHeuristic2():
    scenario = util.deliveryScenario2
    problem = submission.DeliveryProblem(scenario)
    algorithm = submission.AStarSearch(submission.createHeuristic2(scenario, 0))
    algorithm.solve(problem)
    if algorithm.totalCost != 25:
        grader.fail("heuristic2 produces wrong total cost")
        return
    # This is a coarse check, report your exact number of explored nodes in writeup
    if algorithm.numStatesExplored >= 40:
        grader.fail("heuristic2 explores too many states")
        return
    grader.assignFullCredit()
    print "numStatesExplored: ", algorithm.numStatesExplored
    print "totalCost: ", algorithm.totalCost
grader.addBasicPart('2d-0', testHeuristic2)


############################################################
# Problem 2e

def testHeuristic3():
    scenario = util.deliveryScenario2
    problem = submission.DeliveryProblem(scenario)
    algorithm = submission.AStarSearch(submission.createHeuristic3(scenario))
    algorithm.solve(problem)
    if algorithm.totalCost != 25:
        grader.fail("heuristic3 produces wrong total cost")
        return
    # This is a coarse check, report your exact number of explored nodes in writeup
    if algorithm.numStatesExplored >= 60:
        grader.fail("heuristic3 explores too many states")
        return
    grader.assignFullCredit()
    print "numStatesExplored: ", algorithm.numStatesExplored
    print "totalCost: ", algorithm.totalCost
grader.addBasicPart('2e-0', testHeuristic3, 1)

grader.grade()
