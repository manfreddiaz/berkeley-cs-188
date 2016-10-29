# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from math import log

import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        next_pacman_position = successor_game_state.getPacmanPosition()
        next_food = [food for food in successor_game_state.getFood().asList() if food]
        next_ghosts_states = successor_game_state.getGhostStates()
        next_ghosts_scared_timers = [ghostState.scaredTimer for ghostState in next_ghosts_states]
        "*** YOUR CODE HERE ***"
        # Calculations
        ghost_distance = min(manhattan_distance(next_pacman_position, ghost.configuration.pos) for ghost in next_ghosts_states)
        closest_food_distance = min(manhattan_distance(next_pacman_position, nextFood) for nextFood in next_food) if next_food else 0
        scared_time = min(next_ghosts_scared_timers)

        # It is bad to have remaining pellets, very bad!
        remaining_food_feature = -len(next_food)
        # It is bad to be closed to ghost, but if they are harmless, is not so bad (better not to eat them)
        ghost_distance_feature = -2 / (ghost_distance + 1) if scared_time == 0 else 0.5 / (ghost_distance + 1)
        # It is bad to be fat from food
        closest_food_feature = 0.5 / (closest_food_distance + 1)
        # Power pellets are good, but not that good
        power_pellets_feature = scared_time * 0.5

        return remaining_food_feature + ghost_distance_feature + closest_food_feature + power_pellets_feature

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

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
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.current_depth = 0


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        max_value, next_action = self.minimax_value(gameState, self.index, 0)
        return next_action

    def minimax_value(self, state, agent, depth):
        num_agents = state.getNumAgents()

        if depth == self.depth and agent % num_agents == 0:
            return self.evaluationFunction(state), None

        if agent % num_agents == 0:
            return self.maximize_value(state, agent % num_agents, depth)

        return self.minimize_value(state, agent % num_agents, depth)

    def minimize_value(self, state, agent, depth):
        successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

        if len(successor_states) == 0:
            return self.evaluationFunction(state), None

        value = sys.maxint
        value_action = None

        next_agent = agent + 1
        for successor_state, action in successor_states:
            next_value, next_action = self.minimax_value(successor_state, next_agent, depth)
            if next_value < value:
                value = next_value
                value_action = action

        return value, value_action

    def maximize_value(self, state, agent, depth):
        successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

        if len(successor_states) == 0:
            return self.evaluationFunction(state), None

        value = -sys.maxint
        value_action = None

        next_agent = agent + 1
        next_depth = depth + 1
        for successor_state, action in successor_states:
            next_value, next_action = self.minimax_value(successor_state, next_agent, next_depth)
            if next_value > value:
                value = next_value
                value_action = action

        return value, value_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

