import collections, util, math, random

############################################################
# Problem 1a

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    #(newState, prob, reward)
    succs = mdp.succAndProbReward(state, action)
    discount = mdp.discount()
    return sum([succ[1]*(succ[2] + (discount * V[succ[0]])) for succ in succs])
    # END_YOUR_CODE

############################################################
# Problem 1b

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    mdp.computeStates()
    oldV = V.copy()
    newV = dict()
    counter = set()
    while len(counter) != len(mdp.states):
        for s in mdp.states:
            newV[s] = computeQ(mdp, oldV, s, pi[s])
            if abs(oldV[s] - newV[s]) < epsilon:
                counter.add(s)
            oldV[s] = newV[s]
    return newV
    # END_YOUR_CODE

############################################################
# Problem 1c

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    pi = dict()
    for state in mdp.states:
        max_a = None
        max_Q = None
        actions = mdp.actions(state)
        for action in actions:
            Q = computeQ(mdp, V, state, action)
            if max_Q == None or Q > max_Q:
                max_Q = Q
                max_a = action
        pi[state] = max_a
    return pi
    # END_YOUR_CODE

############################################################
# Problem 1d

def IsChanging(states, prev_pi, pi):
    for state in states:
        if state in prev_pi and state in pi:
            if prev_pi[state] != pi[state]:
                return True
        else:
            return True
    return False

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute V and pi
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        V = dict()
        for state in mdp.states:
            V[state] = 0.0
        prev_pi = dict()
        pi = dict()
        while IsChanging(mdp.states, prev_pi, pi):
            prev_pi = pi.copy()
            pi = computeOptimalPolicy(mdp, V)
            V = policyEvaluation(mdp, V, pi, epsilon)
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 1e

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 12 lines of code expected)
        oldV = dict()
        for state in mdp.states:
            oldV[state] = 0.0
        newV = oldV.copy()
        counter = set()
        pi = dict()
        while len(counter) != len(mdp.states):
            pi = computeOptimalPolicy(mdp, newV)
            for s in mdp.states:
                newV[s] = computeQ(mdp, oldV, s, pi[s])
                if abs(oldV[s] - newV[s]) < epsilon:
                    counter.add(s)
                oldV[s] = newV[s]
        # END_YOUR_CODE
        self.pi = pi
        self.V = newV

############################################################
# Problem 1f

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().

# TODO
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 2a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 50 lines of code expected)
        actions = self.actions(state)
        succAndRewards = list()
        if sum(state[2]) == 0 or state[0] > self.threshold:
            return succAndRewards
        #Quit   
        if action == actions[2]:
            succAndRewards.append(((state[0], None, (0,)), 1.0, state[0]))

        #Take
        elif action == actions[0]:
            # Peeked previously
            if state[1] != None:
                newDeck = list(state[2])
                newDeck[state[1]] = newDeck[state[1]] - 1
                newValue = state[0] + self.cardValues[state[1]]
                if newValue > self.threshold:
                    newDeck = [0]
                newState = (newValue, None, 
                            tuple(newDeck))
                succAndRewards.append((newState, 1.0, 0))
            else:
                deck = state[2]
                totalCards = sum(deck)
                for cardIndex, count in enumerate(deck):
                    if count <= 0: 
                        continue
                    prob = count * 1.0 / totalCards
                    newDeck = list(deck)
                    newDeck[cardIndex] = newDeck[cardIndex] - 1
                    newCardCount = state[0] + self.cardValues[cardIndex]
                    reward = 0
                    if newCardCount > self.threshold:
                        newState = (newCardCount, state[1], (0,))
                        succAndRewards.append((newState, prob, reward))
                    else:
                        if sum(newDeck) == 0:
                            reward = newCardCount
                            newState = (newCardCount, state[1], (0,))
                        else:
                            newState = (newCardCount, state[1], tuple(newDeck))
                        succAndRewards.append((newState, prob, reward))
        #Peek
        elif action == actions[1]:
            if state[1] == None:
                deck = state[2]
                totalCards = sum(deck)
                for card, count in enumerate(deck):
                    if count <= 0: continue
                    prob = count * 1.0 / totalCards
                    newState = (state[0], card, state[2])
                    succAndRewards.append((newState, prob, -1))

        return succAndRewards
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 2b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    #TODO
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # Your algorithm will be asked to produce an action given a state.
    # You should use an epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        actions = self.actions(state)
        max_a = None
        if random.random() < self.explorationProb:
            index = random.randrange(len(actions))
            max_a = actions[index]
        else:
            
            max_Q = None
            for action in actions:
                Q = self.getQ(state, action)
                if max_Q == None or Q > max_Q:
                    max_Q = Q
                    max_a = action
        return max_a 
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (around 12 lines of code expected)
        r = None
        if newState == None:
            r = reward - self.getQ(state, action)
        else:
            r = (reward + self.discount * max([self.getQ(newState, a) for a in self.actions(newState)]) 
                 - self.getQ(state, action))
        featureVector = self.featureExtractor(state, action)
        for pair in featureVector:
            self.weights[pair[0]] = self.weights[pair[0]] + self.getStepSize() * r * pair[1]
        # END_YOUR_CODE

############################################################
# Problem 3b: convergence of Q-learning

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

############################################################
# Problem 3c: features for Q-learning.

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card (1 feature).
# - indicator on the number of cards for each card type and the action (len(counts) features).
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    features = list()
    features.append(((total, action), 1))
    cardPresence = [1 if count > 0 else 0 for count in counts]
    features.append(((tuple(cardPresence), action), 1))
    for i, count in enumerate(counts):
        features.append(((i, count, action), 1))
    return features
    # END_YOUR_CODE

############################################################
# Problem 3d: changing mdp

#TODO redownload code
# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=9, peekCost=1)
