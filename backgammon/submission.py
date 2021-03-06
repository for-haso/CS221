import agent, math


############################################################
# Problem 1a

def simpleEvaluation(state, evalArgs=None):
    """
    Evaluates the current game state with a simple heuristic.

    @param state : Tuple of (game,player), the game is
    a game object (see game.py for details, and player in
    {'o','x'} designates whose turn it is.

    @returns V : (scalar) evaluation of current game state
    """
    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    game = state[0]
    player = game.players[0]
    numOffPieces = len(game.offPieces[player])
    numBarPieces = len(game.barPieces[player])
    numHomePieces = 0
    for i in range(12, 16):
        for j in range(len(game.grid[i])):
            if game.grid[i][j] == player:
                numHomePieces+=1
    V = 10.0 * numOffPieces + 1.0 * numHomePieces - numBarPieces * 1.0/10.0
    # END_YOUR_CODE
    return V

############################################################
# Problem 1b

# Reflex Agent evaluates only the current game state and returns
# the best action.
class ReflexAgent(agent.Agent, object):

    def __init__(self, player, evalFunction, evalArgs=None):
        super(self.__class__, self).__init__(player)
        self.evaluationFunction = evalFunction
        self.evaluationArgs = evalArgs

    def getAction(self, actions, game):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.

        @param actions : A set() of possible legal actions for a given roll,
        player and game state.
        @param game : game object (see game.py for details).

        Methods and attributes that may come in handy include:

        self.player - the player this agent represents

        game.clone() - returns a copy of the current game

        game.takeAction(action, player) - takes the action for the
        player with the given player.

        @returns action : Best action according to
        self.evaluationFunction from set of actions.  If there are
        several best, pick the one with the lexicographically largest
        action.
        """
        # - Call the evaluation function using the instance attribute
        #
        #     self.evaluationFunction(state, self.evaluationArgs)
        #
        # - state is a pair: (game, player)
        # - self.evaluationArgs will be the weights when we are using
        #   a linear evaluation function.  For the simple evaluation
        #   this will be None.
        # BEGIN_YOUR_CODE (around 6 lines of code expected)
        max_v = None
        max_a = None
        for action in actions:
            newGame = game.clone()
            newGame.takeAction(action, self.player)
            v = self.evaluationFunction((newGame, self.player), self.evaluationArgs)
            if max_v != None:
                if v > max_v:
                    max_a = action
                    max_v = v
                elif v == max_v:
                    if action > max_a:
                        max_a = action
            else:
                max_v = v
                max_a = action

        # END_YOUR_CODE
        return max_a

    def setWeights(self, w):
        """
        Updates weights of reflex agent.  Used for training.
        """
        self.evaluationArgs = w

############################################################
# Problem 1c

class ExpectimaxAgent(agent.Agent, object):

    def getValue(self, game, player):
        rolls = [(i, j) for i in range(1, game.die + 1) for j in range(1, game.die + 1)]
        expectation = 0.0
        for roll in rolls:
            actions = game.getActions(roll, player)
            v = 0.0
            for action in actions:
                newGame = game.clone()
                newGame.takeAction(action, player)
                v += self.evaluationFunction((newGame, game.opponent(player)), self.evaluationArgs)
            if len(actions) != 0:
                v *= 1.0/len(actions)
            expectation += v * 1.0/len(rolls)
        return expectation


    def getAction(self, actions, game):
        """
        Return best action according to self.evaluationFunction,
        with 2-ply lookahead.

        @param actions : A set() of possible legal actions for a given roll,
        player and game state.
        @param game : game object (see game.py for details).

        Methods and instance variables that may come in handy include:

        game.getActions(roll, player) - returns the set of legal actions for
        a given roll and player.

        game.clone() - returns a copy of the current game

        game.takeAction(action, player) - takes the action for the
        player and CHANGES the game state. You probably want to use
        game.clone() to copy the game first.

        game.die - the number of sides on the die

        game.opponent(player) - returns the opponent of the given player

        @returns action : Best action according to
        self.evaluationFunction from set of actions.  If there are
        several best, pick the one with the lexicographically largest
        action.

        """
        # - Call the evaluation function using the instance attribute
        #
        #     self.evaluationFunction(state, self.evaluationArgs)
        #
        # - state is a pair: (game, player)
        # - self.evaluationArgs will be the weights when we are using
        #   a linear evaluation function.  For the simple evaluation
        #   this will be None.

        def allDiceRolls(game):
            # Helper function to return all possible dice rolls for a game object
            return [(i, j) for i in range(1, game.die + 1) for j in range(1, game.die + 1)]

        # BEGIN_YOUR_CODE (around 20 lines of code expected)
        rolls = allDiceRolls(game)
        numRolls = len(rolls)
        max_v = None
        max_a = None
        for action in actions:
            newGame = game.clone()
            newGame.takeAction(action, self.player)
            v = self.getValue(newGame, game.opponent(self.player))
            if max_v != None:
                if v > max_v:
                    max_a = action
                    max_v = v
                elif v == max_v:
                    if action > max_a:
                        max_a = action
            else:
                max_v = v
                max_a = action

        # END_YOUR_CODE
        return max_a


    def __init__(self, player, evalFn, evalArgs=None):
        super(self.__class__, self).__init__(player)
        self.evaluationFunction = evalFn
        self.evaluationArgs = evalArgs

############################################################
# Problem 2a


def extractFeatures(state):
    """
    @param state : Tuple of (game, player), the game is
    a game object (see game.py for details, and player in
    {'o', 'x'} designates whose turn it is.

    @returns features : List of real valued features for given state.

    Methods and instance variables that may come in handy include:

    game.getActions(roll, player) - returns the set of legal actions for
    a given roll and player.

    game.clone() - returns a copy of the current game

    game.grid - 2-D array (list of lists) with current piece placement on
    board. For example game.grid[0][3] = 'x'

    game.barPieces - dictionary with key as player and value a list of
    pieces on the bar for that player. Recall on the bar means the piece was
    "clobbered" by the opponent. In our simplified backgammon these pieces
    can't return to play.

    game.offPieces - dictionary with key as player and value a list
    of pieces successfully taken of the board by the player.

    game.numPieces - dictionary with key as player and value number
    of total pieces for that player.

    game.players - list of players 1 and 2 in order
    """
    # BEGIN_YOUR_CODE (around 20 lines of code expected)
    game = state[0]
    features = []
    for player in game.players:
        for i in range(len(game.grid)):
            count = 0
            for j in range(len(game.grid[i])):
                if game.grid[i][j] == player:
                    count += 1
            if count >= 1:
                features.append(1)
            else:
                features.append(0)
            if count >= 2:
                features.append(1)
            else:
                features.append(0)
            if count >= 3:
                features.append(count - 2)
            else:
                features.append(0)
        features.append(len(game.barPieces[player]))
        features.append(len(game.offPieces[player]) * 1.0/game.numPieces[player])
    if state[1] == game.players[0]:
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)
    features.append(1)
    return features


############################################################
# Problem 2b

def logLinearEvaluation(state, w):
    """
    Evaluate the current state using the log-linear evaluation
    function.

    @param state : Tuple of (game, player), the game is
    a game object (see game.py for details, and player in
    {'o', 'x'} designates whose turn it is.

    @param w : List of feature weights.

    @returns V : Evaluation of current game state.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    if state == None:
        return 0.0
    phi = extractFeatures(state)
    z = sum([phi[i] * weight for i, weight in enumerate(w)])
    V = 1 / (1 + math.exp(-1 * z))
    # END_YOUR_CODE
    return V

############################################################
# Problem 2c

def TDUpdate(state, nextState, reward, w, eta):
    """
    Given two sequential game states, updates the weights
    with a step size of eta, using the Temporal Difference learning
    algorithm.

    @param state : Tuple of game state (game object, player).
    @param nextState : Tuple of game state (game object, player),
    note if the game is over this will be None. In this case, 
    the next value for the TD update will be 0.
    @param reward : The reward is 1 if the game is over and your
    player won, 0 otherwise.
    @param w : List of feature weights.
    @param eta : Step size for learning.

    @returns w : Updated weights.
    """
    # BEGIN_YOUR_CODE (around 13 lines of code expected)
    r = reward + logLinearEvaluation(nextState, w) - logLinearEvaluation(state, w)
    phi = extractFeatures(state)
    z = sum([phi[i] * weight for i, weight in enumerate(w)])
    gradient = [phi[i] * math.exp(z)/((1 + math.exp(z)) ** 2) for i, weight in enumerate(w)]
    print gradient
    for i in range(len(w)):
        w[i] = w[i] + eta * r * gradient[i]
    # END_YOUR_CODE
    return w
