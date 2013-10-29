from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.	You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""
	def __init__(self):
		self.lastPositions = []
		self.dc = None


	def getAction(self, gameState):
		"""
		getAction chooses among the best options according to the evaluation function.

		getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
		------------------------------------------------------------------------------
		Description of GameState and helper functions:

		A GameState specifies the full game state, including the food, capsules,
		agent configurations and score changes. In this function, the |gameState| argument 
		is an object of GameState class. Following are a few of the helper methods that you 
		can use to query a GameState object to gather information about the present state 
		of Pac-Man, the ghosts and the maze.
		
		gameState.getLegalActions(): 
			Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

		gameState.generateSuccessor(agentIndex, action): 
			Returns the successor state after the specified agent takes the action. 
			Pac-Man is always agent 0.

		gameState.getPacmanState():
			Returns an AgentState object for pacman (in game.py)
			state.pos gives the current position
			state.direction gives the travel vector

		gameState.getGhostStates():
			Returns list of AgentState objects for the ghosts

		gameState.getNumAgents():
			Returns the total number of agents in the game

		
		The GameState class is defined in pacman.py and you might want to look into that for 
		other helper methods, though you don't need to.
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (oldFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		oldFood = currentGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.	Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.	Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.	It's
	only partially specified, and designed to be extended.	Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent (problem 1)
	"""

	def getValue(self, gameState, depth, player):
		# base cases: terminal states
		if depth == 0 and player == 0:
			return self.evaluationFunction(gameState)
		actions = gameState.getLegalActions(player)
		if gameState.isWin() or gameState.isLose() or len(actions) == 0:
			return self.evaluationFunction(gameState)
		# recursive cases: still more states to go
		# it's pacman's turn
		if player == 0:
			max_v = None
			for action in actions:
				nextState = gameState.generateSuccessor(player, action)
				v = self.getValue(nextState, depth - 1, 1)
				if max_v != None:
					if v > max_v:
						max_v = v
				else:
					max_v = v
			return max_v
		# else, if the current player is an opponent (ghost)
		min_v = None
		numAgents = gameState.getNumAgents()
		for action in actions:
			nextState = gameState.generateSuccessor(player, action)
			v = self.getValue(nextState, depth, (player + 1) % numAgents)
			if min_v != None:
				if v < min_v:
					min_v = v
			else:
				min_v = v
		return min_v

	def getAction(self, gameState):
		"""
			Returns the minimax action from the current gameState using self.depth
			and self.evaluationFunction. 
			Terminal states can be found by one of the following: 
			pacman won, pacman lost or there are no legal moves. 
		"""

		# BEGIN_YOUR_CODE (around 68 lines of code expected)
		numAgents = gameState.getNumAgents()
		max_v = None
		max_a = None
		actions = gameState.getLegalActions(self.index)
		if len(actions) == 0:
			return None
		
		depth = self.depth
		if self.index == 0:
			depth -= 1
		for action in actions:
			nextState = gameState.generateSuccessor(self.index, action)
			v = self.getValue(nextState, depth, (self.index + 1) % numAgents)
			if max_v != None:
				if v > max_v:
					max_v = v
					max_a = action
			else:
				max_v = v
				max_a = action
		print max_v
		return max_a
		# END_YOUR_CODE

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (problem 2)
	"""

	def getValue(self, gameState, depth, player, alpha, beta):
		# base cases: terminal states
		if depth == 0 and player == 0:
			return self.evaluationFunction(gameState)
		actions = gameState.getLegalActions(player)
		if gameState.isWin() or gameState.isLose() or len(actions) == 0:
			return self.evaluationFunction(gameState)
		# recursive cases: still more states to go
		# it's pacman's turn
		if player == 0:
			max_v = None
			for action in actions:
				nextState = gameState.generateSuccessor(player, action)
				v = self.getValue(nextState, depth - 1, 1, alpha, beta)
				if max_v != None:
					if v > max_v:
						max_v = v
				else:
					max_v = v
				if beta != None:
					if max_v >= beta:
						return max_v
				if alpha != None:
					alpha = max([alpha, max_v])
				else:
					alpha = max_v
			return max_v
		# else, if the current player is an opponent (ghost)
		min_v = None
		numAgents = gameState.getNumAgents()
		for action in actions:
			nextState = gameState.generateSuccessor(player, action)
			v = self.getValue(nextState, depth, (player + 1) % numAgents, alpha, beta)
			if min_v != None:
				if v < min_v:
					min_v = v
			else:
				min_v = v
			if alpha != None:
				if min_v <= alpha:
					return min_v
			if beta != None:
				beta = min([beta, min_v])
			else:
				beta = min_v
		return min_v

	def getAction(self, gameState):
		"""
			Returns the minimax action using self.depth and self.evaluationFunction
		"""

		# BEGIN_YOUR_CODE (around 69 lines of code expected)
		numAgents = gameState.getNumAgents()
		max_v = None
		max_a = None
		actions = gameState.getLegalActions(self.index)
		if len(actions) == 0:
			return None
		
		depth = self.depth
		if self.index == 0:
			depth -= 1
		for action in actions:
			nextState = gameState.generateSuccessor(self.index, action)
			v = self.getValue(nextState, depth, (self.index + 1) % numAgents, None, None)
			if max_v != None:
				if v > max_v:
					max_v = v
					max_a = action
			else:
				max_v = v
				max_a = action
		return max_a
		# END_YOUR_CODE

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	Your expectimax agent (problem 3)
	"""

	def getAction(self, gameState):
		"""
			Returns the expectimax action using self.depth and self.evaluationFunction

			All ghosts should be modeled as choosing uniformly at random from their
			legal moves.
		"""

		# BEGIN_YOUR_CODE (around 70 lines of code expected)
		raise Exception("Not implemented yet")
		# END_YOUR_CODE

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (problem 4).

	DESCRIPTION: <write something here so we know what you did>
	"""

	# BEGIN_YOUR_CODE (around 71 lines of code expected)
	raise Exception("Not implemented yet")
	# END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction


