
"""# TODO: remove
class BitProblem(util.SearchProblem):
    # |scenario|: delivery specification.
    def __init__(self):
        return
    # state: tuple containing (location, packages delivered, packages held)
    # Return the start state.
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return "0000000"
        # END_YOUR_CODE

    # Return whether |state| is a goal state or not.
    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        if state == "1111111":
            return True
        return False
        # END_YOUR_CODE

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        print state
        # Hint: Call self.scenario.getNeighbors((x,y)) to get the valid neighbors
        # at that location. In order for the simulation code to work, please use
        # the exact strings 'Pickup' and 'Dropoff' for those two actions.
        # BEGIN_YOUR_CODE (around 18 lines of code expected)
        # check cur location for pickup/dropoff -> cost = 0
        succs = list()
        for i in range(len(state)):
            bit = state[i]
            if bit == 1:
                bit = 0
            else:
                bit = 1
            succs.append((i, state[:i] + str(bit) + state[i+1:], 1))
        return succs
        # END_YOUR_CODE

def createHeuristic4():
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    def heuristic(state):
        count = 1
        for i in range(len(state)):
            if state[i] == '0':
                count += 1
        return count ** 2
    return heuristic

problem = BitProblem()
algorithm = AStarSearch(createHeuristic4())
algorithm.solve(problem)
print algorithm.numStatesExplored"""
