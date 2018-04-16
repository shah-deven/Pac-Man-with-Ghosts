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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print "temp : ",
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        import sys
        MAX = sys.maxint
        MIN = -MAX

        # No Food left, return MAX
        if len(newFood) == 0:
            return MAX

        newPacmanPosition = newPos
        newGhostPositions = successorGameState.getGhostPositions()
        currentFood = currentGameState.getFood().asList()

        # To store manhattan distances between pacman's new position and ghost's new position
        nonscaryGhostDistances = []
        scaryGhostDistances = []

        # To store the new positions of ghosts
        nonscaryGhosts = []
        scaryGhosts = []

        # Calculate the manhattan distances between Pacman's new position and each new ghost position
        for i in range(0, len(newGhostPositions)):
            if newScaredTimes[i] > 0:
                nonscaryGhostDistances.append(manhattanDistance(newPacmanPosition, newGhostPositions[i]))
                nonscaryGhosts.append(newGhostPositions[i])
            else:
                scaryGhostDistances.append(manhattanDistance(newPacmanPosition, newGhostPositions[i]))
                scaryGhosts.append(newGhostPositions[i])

        # To store the manhattan distances between Pacman's new position and new food positions
        foodDistances = []
        for elem in newFood:
            foodDistances.append(manhattanDistance(newPacmanPosition, elem))


        closestFoodDistance = MAX
        closestScaryGhostDist = MAX
        closestNonScaryGhostDist = MAX

        # Find the manhattan distance to the closest non scary ghost
        if len(nonscaryGhostDistances) > 0:
            nonscaryGhostDistances = sorted(nonscaryGhostDistances)
            closestNonScaryGhostDist = nonscaryGhostDistances[0]

        # Find the manhattan distance to the closest scary ghost
        if len(scaryGhostDistances) > 0:
            scaryGhostDistances = sorted(scaryGhostDistances)
            closestScaryGhostDist = scaryGhostDistances[0]

        # Sort manhattan distances from Pacman's new position to the new Food position in ascending order and find the closest food distance
        foodDistances = sorted(foodDistances)
        closestFoodDistance = foodDistances[0]

        # Threshold value is used to check whether the ghost is within a certain range of Pacman
        threshold = 4

        # If a scary ghost is within the range of Pacman
        if closestScaryGhostDist < threshold:
            # If Pacman is losing in the next move, then return MIN
            if newPacmanPosition in scaryGhosts:
                return MIN
            # If Pacman is not losing in next move, then return the closest manhattan distance between the ghost and the Pacman
            else:
                return closestScaryGhostDist

        # If a non scary ghost is within the range of Pacman
        if closestNonScaryGhostDist < threshold:
            # If the manhattan distance between Pacman and closest non scary ghost is greater than the manhattan distance from closest food,
            # then chase food
            # else chase the non scary ghost
            if closestNonScaryGhostDist > closestFoodDistance:
                return threshold + 1 - closestFoodDistance
            else:
                return threshold + 1 - closestNonScaryGhostDist

        # If Pacman eats the food in the next move, then return MAX value so that Pacman eats it
        if newPacmanPosition in currentFood:
            return MAX

        # If Pacman eats the pellet in the next move, then return MAX value so that the Pacman eats it
        if newPos in successorGameState.getCapsules():
            return MAX

        # If no ghost is in range and no capsule is nearby, chase the closest food
        return threshold + 1 - closestFoodDistance

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

        #self.nodesExpanded = 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minimaxFunction(self, gameState, action, currentDepth, currentAgentIndex, isMax):
        # This function takes the following arguments:
            # gameState: current game state
            # action: the predecessor's action which leads to the current game state
            # currentDepth: the depth of the current game state
            # currentAgentIndex: the agent whose successor states have to be evaluated
            # isMax: True for max nodes (Pacman), False for min nodes (ghosts)
        # This function returns [score, action], which is the best possible score and the corresponding action

        # Max node
        if isMax:
            # If maximum depth is reached, return the evaluated value of the node
            if currentDepth == self.depth:
                return [self.evaluationFunction(gameState), action]
            # List to store the score of each Pacman action
            chooseMaxArray = []
            # Get legal Pacman actions and generate Pacman successor states
            for pacmanAction in gameState.getLegalActions(currentAgentIndex):
                pacmanSuccessorState = gameState.generateSuccessor(currentAgentIndex, pacmanAction)
                #self.nodesExpanded = self.nodesExpanded + 1
                # Whenever min is called from max, the Pacman successor state is passed to the FIRST ghost
                currentGhostIndex = 1
                # Pass the successor state to the min node and append the result to chooseMaxArray
                chooseMaxArray.append(self.minimaxFunction(pacmanSuccessorState, pacmanAction, currentDepth, currentGhostIndex, False))
            # Sort the list in descending order to choose the action having maximum return value
            chooseMaxArray = sorted(chooseMaxArray, reverse=True)
            # If chooseMaxArray is not empty
            if len(chooseMaxArray) > 0:
                # If the current node is not the root, return the node with maximum return value, with the action of the previous node
                if currentDepth != 0:
                    return [chooseMaxArray[0][0], action]
                # If the current node is the root, return the node with maximum return value with its corresponding action
                else:
                    return chooseMaxArray[0]
            # If chooseMaxArray is empty, return the evaluated value of the node, with the action of the previous node
            else:
                return [self.evaluationFunction(gameState), action]
        # Min node
        else:
            # If maximum depth is reached, return the evaluated value of the node
            if currentDepth == self.depth:
                return [self.evaluationFunction(gameState), action]
            # Get the number of ghosts
            numGhosts = gameState.getNumAgents() - 1
                # List to store the score of each ghost action
            chooseMinArray = []
            # Get legal ghost actions and generate ghost successor states
            for ghostAction in gameState.getLegalActions(currentAgentIndex):
                ghostSuccessorState = gameState.generateSuccessor(currentAgentIndex, ghostAction)
                #self.nodesExpanded = self.nodesExpanded + 1
                # If this is the last ghost
                if currentAgentIndex == numGhosts:
                    # As this is the last ghost, pass the ghost successor state to the max node (Pacman)
                    # by incrementing the current depth value
                    # and append the result to chooseMinArray
                    chooseMinArray.append(self.minimaxFunction(ghostSuccessorState, ghostAction, currentDepth + 1, 0, True)) # call max with depth + 1
                else:
                    # As this is not the last ghost, pass the ghost successor state to the NEXT ghost
                    # and append the result to chooseMinArray
                    chooseMinArray.append(self.minimaxFunction(ghostSuccessorState, ghostAction, currentDepth, currentAgentIndex + 1, False)) # calling min of next ghost

            # Sort the list in ascending order to choose the action having minimum return value
            chooseMinArray = sorted(chooseMinArray)
            # If chooseMinArray is not empty, return the node with minimum return value, with the action of the previous node
            if len(chooseMinArray) > 0:
                return [chooseMinArray[0][0], action]
            # If chooseMinArray is empty, return the evaluated value of the node, with the action of the previous node
            else:
                return [self.evaluationFunction(gameState), action]

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # Call minimaxFunction with initial arguments as:
            # current game state
            # current action = dummy, as the start state has no predecessor
            # current depth = 0, as we start from the root node
            # current agent index = 0, as we start from Pacman
            # isMax = True, as Pacman node is a max node
        # The function returns the best possible score and the action associated with it
        score, action = self.minimaxFunction(gameState, "dummy", 0, 0, True)
        #print "Nodes expanded = ", self.nodesExpanded

        # Return the best action obtained above
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabetaFunction(self, gameState, action, currentDepth, currentAgentIndex, isMax, alpha, beta):
        # This function takes the following arguments:
            # gameState: current game state
            # action: the predecessor's action which leads to the current game state
            # currentDepth: the depth of the current game state
            # currentAgentIndex: the agent whose successor states have to be evaluated
            # isMax: True for max nodes (Pacman), False for min nodes (ghosts)
            # alpha: the alpha value propagated from the parent
            # beta: the beta value propagated from the parent
        # This function returns [score, action, retAlpha, retBeta], where:
            # score is the best possible score
            # action is the corresponding move
            # retAlpha is the evaluated alpha value of the current node
            # retBeta is the evaluated beta value of the current node
        import sys
        INF = sys.maxint
        # Max node
        if isMax:
            # If maximum depth is reached, return the evaluated value of the node as alpha
            if currentDepth >= self.depth or gameState.isWin() or gameState.isLose():
                alpha = self.evaluationFunction(gameState)
                return [self.evaluationFunction(gameState), action, alpha, beta]
            # List to store the score of each Pacman action
            chooseMaxArray = []
            # Get legal Pacman actions and generate Pacman successor states
            for pacmanAction in gameState.getLegalActions(currentAgentIndex):
                # Pruning - if beta becomes less than alpha, stop exploring further successors
                if beta < alpha:
                    return [retScore, retAction, alpha, beta]
                pacmanSuccessorState = gameState.generateSuccessor(currentAgentIndex, pacmanAction)
                #self.nodesExpanded = self.nodesExpanded + 1
                # Whenever min is called from max, the Pacman successor state is passed to the FIRST ghost
                currentGhostIndex = 1
                retScore, retAction, retAlpha, retBeta = self.alphabetaFunction(pacmanSuccessorState, pacmanAction, currentDepth, currentGhostIndex, False, alpha, beta)
                # Update the alpha value
                if retBeta > alpha:
                    alpha = retBeta
                # Append the result to chooseMaxArray
                chooseMaxArray.append([retScore, retAction])
            # Sort chooseMaxArray in descending order to choose the action having maximum return value
            chooseMaxArray = sorted(chooseMaxArray, reverse=True)
            # If chooseMaxArray is not empty
            if len(chooseMaxArray) > 0:
                # If the current node is not the root, return the node with maximum return value, with the action of the previous node, and the score as alpha
                if currentDepth != 0:
                    return [chooseMaxArray[0][0], action, chooseMaxArray[0][0], beta]
                # If the current node is the root, return the node with maximum return value with its corresponding action, and the score as alpha
                else:
                    return [chooseMaxArray[0][0], chooseMaxArray[0][1], chooseMaxArray[0][0], beta]
            # If chooseMaxArray is empty, return the evaluated value of the node, with the action of the previous node, with updated alpha-beta values
            else:
                retScore, retAction, retAlpha, retBeta = self.alphabetaFunction(gameState, action, currentDepth + 1, 1, False, alpha, beta)
                alpha = retBeta
                return [retScore, retAction, alpha, beta]
        # Min node
        else:
            # If maximum depth is reached, return the evaluated value of the node as beta
            if currentDepth >= self.depth or gameState.isWin() or gameState.isLose():
                beta = self.evaluationFunction(gameState)
                return [self.evaluationFunction(gameState), action, alpha, beta]
            # Get number of ghosts
            numGhosts = gameState.getNumAgents() - 1
            # List to store the score of each ghost action
            chooseMinArray = []
            # Get legal ghost actions and generate ghost successor states
            for ghostAction in gameState.getLegalActions(currentAgentIndex):
                # Pruning - if beta becomes less than alpha, stop exploring further successors
                if beta < alpha:
                    return [retScore, retAction, alpha, beta]
                ghostSuccessorState = gameState.generateSuccessor(currentAgentIndex, ghostAction)
                #self.nodesExpanded = self.nodesExpanded + 1
                # If this is the last ghost
                if currentAgentIndex == numGhosts:
                    # As this is the last ghost, pass the ghost successor state to the max node (Pacman)
                    # by incrementing the current depth value
                    # and append the result to chooseMinArray
                    retScore, retAction, retAlpha, retBeta = self.alphabetaFunction(ghostSuccessorState, ghostAction, currentDepth + 1, 0, True, alpha, beta) # call max with depth + 1
                    # Update the beta value
                    if retAlpha < beta:
                        beta = retAlpha
                # If this is not the last ghost
                else:
                    # pass the ghost successor state to the max node (Pacman)
                    # and append the result to chooseMinArray
                    retScore, retAction, retAlpha, retBeta = self.alphabetaFunction(ghostSuccessorState, ghostAction, currentDepth, currentAgentIndex + 1, False, alpha, beta) # calling min of next ghost
                    # Update the beta value
                    if beta > retBeta:
                        beta = retBeta
                # Append the result to chooseMinArray
                chooseMinArray.append([retScore, retAction])
            # Sort chooseMinArray in ascending order to choose the action having minimum return value
            chooseMinArray = sorted(chooseMinArray)
            # If chooseMinArray is not empty, return the node with minimum return value, with the action of the previous node, and the score as beta
            if len(chooseMinArray) > 0:
                return [chooseMinArray[0][0], action, alpha, chooseMinArray[0][0]]
            # If chooseMinArray is empty, return the evaluated value of the node, with the action of the previous node, and the score as beta
            else:
                retScore, retAction, retAlpha, retBeta = self.alphabetaFunction(gameState, action, currentDepth + 1, 0, True, alpha, beta)
                beta = retAlpha
                return[retScore, retAction, alpha, beta]

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys
        INF = sys.maxint
        # Call alphabetaFunction with initial arguments as:
            # current game state
            # current action = dummy, as the start state has no predecessor
            # current depth = 0, as we start from the root node
            # current agent index = 0, as we start from Pacman
            # isMax = True, as Pacman node is a max node
            # alpha = -INF initially
            # beta = INF initially
        # The function returns the best possible score and the action associated with it, along with the associated alpha-beta values
        score, action, alpha, beta = self.alphabetaFunction(gameState, "dummy", 0, 0, True, -INF, INF)
        #print "Nodes expanded = ", self.nodesExpanded

        # Return the best action obtained above
        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimaxFunction(self, gameState, action, currentDepth, currentAgentIndex, isMax):
        # This function takes the following arguments:
            # gameState: current game state
            # action: the predecessor's action which leads to the current game state
            # currentDepth: the depth of the current game state
            # currentAgentIndex: the agent whose successor states have to be evaluated
            # isMax: True for max nodes (Pacman), False for min nodes (ghosts)
        # This function returns [score, action], which is the best possible score and the corresponding action

        # Max node
        if isMax:
            # If maximum depth is reached, return the evaluated value of the node
            if currentDepth == self.depth:
                return [self.evaluationFunction(gameState), action]
            # List to store the score of each Pacman action
            chooseMaxArray = []
            # Get legal Pacman actions and generate Pacman successor states
            for pacmanAction in gameState.getLegalActions(currentAgentIndex):
                pacmanSuccessorState = gameState.generateSuccessor(currentAgentIndex, pacmanAction)
                #self.nodesExpanded = self.nodesExpanded + 1
                # Whenever min is called from max, the Pacman successor state is passed to the FIRST ghost
                currentGhostIndex = 1
                # Pass the successor state to the min node and append the result to chooseMaxArray
                chooseMaxArray.append(self.expectimaxFunction(pacmanSuccessorState, pacmanAction, currentDepth, currentGhostIndex, False))
            # Sort the list in descending order to choose the action having maximum return value
            chooseMaxArray = sorted(chooseMaxArray, reverse=True)
            # If chooseMaxArray is not empty
            if len(chooseMaxArray) > 0:
                # If the current node is not the root, return the node with maximum return value, with the action of the previous node
                if currentDepth != 0:
                    return [chooseMaxArray[0][0], action]
                # If the current node is the root, return the node with maximum return value with its corresponding action
                else:
                    return chooseMaxArray[0]
            # If chooseMaxArray is empty, return the evaluated value of the node, with the action of the previous node
            else:
                return [self.evaluationFunction(gameState), action]
        # Exp node
        else:
            # If maximum depth is reached, return the evaluated value of the node
            if currentDepth == self.depth:
                return [self.evaluationFunction(gameState), action]
            # Get the number of ghosts
            numGhosts = gameState.getNumAgents() - 1
                # List to store the score of each ghost action
            chooseMinArray = []
            # Get legal ghost actions and generate ghost successor states
            for ghostAction in gameState.getLegalActions(currentAgentIndex):
                ghostSuccessorState = gameState.generateSuccessor(currentAgentIndex, ghostAction)
                #self.nodesExpanded = self.nodesExpanded + 1
                # If this is the last ghost
                if currentAgentIndex == numGhosts:
                    # As this is the last ghost, pass the ghost successor state to the max node (Pacman)
                    # by incrementing the current depth value
                    # and append the result to chooseMinArray
                    chooseMinArray.append(self.expectimaxFunction(ghostSuccessorState, ghostAction, currentDepth + 1, 0, True)) # call max with depth + 1
                else:
                    # As this is not the last ghost, pass the ghost successor state to the NEXT ghost
                    # and append the result to chooseMinArray
                    chooseMinArray.append(self.expectimaxFunction(ghostSuccessorState, ghostAction, currentDepth, currentAgentIndex + 1, False)) # calling min of next ghost
            # If chooseMinArray is not empty, return the node with minimum return value, with the action of the previous node
            if len(chooseMinArray) > 0:
                    length = len(chooseMinArray)
                    sumval = 0.0
                    # Calculate the mean of all scores of the successor states and return along with action
                    for elem in chooseMinArray:
                        sumval = sumval + elem[0]
                    return [sumval / length, action]
            # If chooseMinArray is empty, return the evaluated value of the node, with the action of the previous node
            else:
                return [self.evaluationFunction(gameState), action]

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        score, action = self.expectimaxFunction(gameState, "dummy", 0, 0, True)
        #print "Nodes expanded = ", self.nodesExpanded

        return action

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
