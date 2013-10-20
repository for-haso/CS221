import re, util, math

############################################################
# Problem 1a: UCS test case

# Return an instance of util.SearchProblem.
# You might find it convenient to use
# util.createSearchProblemFromString.
def createUCSTestCase(n):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    problem_string = str()
    for i in range(2, n):
        problem_string += "1 " + str(i) + " 3\n"
        if i != n-1:
            problem_string += str(i) + " " + str(n) + " 3\n"    
    problem_string += str(n-1) + " " + str(n) + " 1\n"
    return util.createSearchProblemFromString("1", str(n), problem_string)
    # END_YOUR_CODE

############################################################
# Problem 1b: A-star search

# Takes the SearchProblem |problem| you're trying to solve and a |heuristic|
# (which is a function that maps a state to an estimate of the cost to the
# goal).  Returns another search problem |newProblem| such that running uniform
# cost search on |newProblem| is equivalent to running A* on |problem| with
# |heuristic|.
def astarReduction(problem, heuristic):
    class NewSearchProblem(util.SearchProblem):
        # Please refer to util.SearchProblem to see the functions you need to
        # overried.
        # BEGIN_YOUR_CODE (around 9 lines of code expected)
        def startState(self):
            return problem.startState()
        def isGoal(self, state):
            return problem.isGoal(state)
        # Return a list of (action, newState, cost) tuples corresponding to edges
        # coming out of |state|.
        def succAndCost(self, state):
            succAndCosts = problem.succAndCost(state)
            astar = list()
            for succ in succAndCosts:
                cost = succ[2] + heuristic(succ[1]) - heuristic(state)
                newSuccAndCost = (succ[0], succ[1], cost)
                astar.append(newSuccAndCost)
            return astar
        # END_YOUR_CODE
    newProblem = NewSearchProblem()
    return newProblem

# Implements A-star search by doing a reduction.
class AStarSearch(util.SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # Reduce the |problem| to |newProblem|, which is solved by UCS.
        newProblem = astarReduction(problem, self.heuristic)
        algorithm = util.UniformCostSearch()
        algorithm.solve(newProblem)

        # Copy solution back
        self.actions = algorithm.actions
        if algorithm.totalCost != None:
            self.totalCost = algorithm.totalCost + self.heuristic(problem.startState())
        else:
            self.totalCost = None
        self.numStatesExplored = algorithm.numStatesExplored

############################################################
# Problem 2b: Delivery

############################################################
# Supporting code for Problem 2.

# A |DeliveryScenario| contains the following information:
# - Starting and ending location of the truck.
# - Pickup and dropoff locations for each package.
# Note: all locations are represented as (row, col) pairs.
# Scenarios are used to construct DeliveryProblem (your job).
# Member variables:
# - numRows, numCols: dimensions of the grid
# - numPackages: number of packages to deliver
# - pickupLocations, dropoffLocations: array of locations for each of the packages
# - truckLocation: where you must start and end
############################################################

class DeliveryProblem(util.SearchProblem):
    # |scenario|: delivery specification.
    def __init__(self, scenario):
        self.scenario = scenario
    # state: tuple containing (location, packages delivered, packages held)
    # Return the start state.
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return (self.scenario.truckLocation, (), ())
        # END_YOUR_CODE

    # Return whether |state| is a goal state or not.
    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        if state == (self.scenario.truckLocation, tuple(range(self.scenario.numPackages)), ()):
            return True
        return False
        # END_YOUR_CODE

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        # Hint: Call self.scenario.getNeighbors((x,y)) to get the valid neighbors
        # at that location. In order for the simulation code to work, please use
        # the exact strings 'Pickup' and 'Dropoff' for those two actions.
        # BEGIN_YOUR_CODE (around 18 lines of code expected)
        # check cur location for pickup/dropoff -> cost = 0
        succs = list()
        # Return a list of valid (action, newLoc) pairs that we can take from loc.
        neighbors = self.scenario.getNeighbors(state[0])
        if state[0] in self.scenario.pickupLocations:
            package = self.scenario.pickupLocations.index(state[0])
            if package not in state[2] and package not in state[1]:
                packages = list(state[2])
                packages.append(package)
                succ = ("Pickup", (state[0], state[1], tuple(packages)), 0)
                succs.append(succ)

        if state[0] in self.scenario.dropoffLocations:
            package = self.scenario.dropoffLocations.index(state[0])
            if package in state[2]:
                packages = list(state[2])
                packages.remove(package)
                dropped = list(state[1])
                dropped.append(package)
                dropped.sort()
                succ = ("Dropoff", (state[0], tuple(dropped), tuple(packages)), 0)
                succs.append(succ)
        # check neighbors for moves -> cost = 1 + size(package_set)
        for neighbor in neighbors:
            succ = (neighbor[0], (neighbor[1], state[1], state[2]), 1 + len(state[2]))
            succs.append(succ)
        return succs
        # END_YOUR_CODE


def distance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
############################################################
# Problem 2c: heuristic 1


# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers and not do any deliveries,
# you just need to go home
def createHeuristic1(scenario):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return distance(scenario.truckLocation, state[0])
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2d: heuristic 2

# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers, but
# you'll need to deliver the given |package|, and then go home

def createHeuristic2(scenario, package):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        cost = 0.0
        loc = state[0]
        home = scenario.truckLocation
        num_packages = len(state[2])
        # case 3: dropped off target package
        if package in state[1]:
            # distance from loc to home
            return distance(loc, home)
        if package not in state[2]:
            # go pickup package
            pickup_loc = scenario.pickupLocations[package]
            cost += distance(loc, pickup_loc)
            num_packages += 1
            loc = pickup_loc
        # case 2: haven't dropped off target package
        # go drop off package
        delivery_loc = scenario.dropoffLocations[package]
        cost += distance(loc, delivery_loc) * 2
        # go home
        cost += distance(delivery_loc, home)
        return cost
        # END_YOUR_CODE
    return heuristic

############################################################
# Problem 2e: heuristic 3

# Return a heuristic corresponding to solving a relaxed problem
# where you will delivery the worst(i.e. most costly) |package|,
# you can ignore all barriers.
# Hint: you might find it useful to call
# createHeuristic2.
def createHeuristic3(scenario):
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    def heuristic(state):
        max_heuristic = 0.0
        for package in range(scenario.numPackages):
            heuristic2 = createHeuristic2(scenario, package)(state)
            if heuristic2 > max_heuristic:
                max_heuristic = heuristic2
        return max_heuristic
    return heuristic
    # END_YOUR_CODE
