# Pac-Man with Ghosts
Design Agents for the classic version of Pac-Man including ghosts.

# Reflex Agent
Implemented reflex agent to consider both food locations and ghosts locations to perform well <br />
python pacman.py -p ReflexAgent -l testClassic <br />
python pacman.py --frameTime 0 -p ReflexAgent -k 1 <br />
python pacman.py --frameTime 0 -p ReflexAgent -k 2 <br />

Default ghosts are random; you can also play for fun with slightly smarter directional ghosts using -g DirectionalGhost. If the randomness is preventing you from telling whether your agent is improving, you can use -f to run with a fixed random seed (same random choices every game). You can also play multiple games in a row with -n. Turn off graphics with -q to run lots of games quickly. <br />

python autograder.py -q q1 <br />
To run it without graphics, use: <br />

python autograder.py -q q1 --no-graphics

# Minimax
Wrote an adversarial search agent in the provided MinimaxAgent class stub in multiAgents.py. Minimax agent works with any number of ghosts. In particular, the minimax tree has multiple min layers (one for each ghost) for every max layer.

The code should also expand the game tree to an arbitrary depth. Score the leaves of your minimax tree with the supplied self.evaluationFunction. MinimaxAgent extends MultiAgentSearchAgent, which gives access to self.depth and self.evaluationFunction. <br />

Important: A single search ply is considered to be one Pacman move and all the ghosts' responses, so depth 2 search will involve Pacman and each ghost moving two times. <br />

Grading: We will be checking your code to determine whether it explores the correct number of game states. This is the only way reliable way to detect some very subtle bugs in implementations of minimax. As a result, the autograder will be very picky about how many times you call GameState.generateSuccessor. If you call it any more or less than necessary, the autograder will complain. To test and debug your code, run <br />

python autograder.py -q q2 <br />
This will show what your algorithm does on a number of small trees, as well as a pacman game. To run it without graphics, use: <br />

python autograder.py -q q2 --no-graphics <br />

python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

# Alpha-Beta Pruning

Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in AlphaBetaAgent.

You should see a speed-up (perhaps depth 3 alpha-beta will run as fast as depth 2 minimax). Ideally, depth 3 on smallClassic should run in just a few seconds per move or faster. <br />

python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic <br />

To test and debug the code, run <br />

python autograder.py -q q3 <br />
This will show what your algorithm does on a number of small trees, as well as a pacman game. To run it without graphics, use: <br />

python autograder.py -q q3 --no-graphics <br />
The correct implementation of alpha-beta pruning will lead to Pacman losing some of the tests. This is not a problem: as it is correct behaviour, it will pass the tests. <br />

# Expectimax
Minimax and alpha-beta are great, but they both assume that you are playing against an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell you, this is not always the case. In this case, I implemented the ExpectimaxAgent, which is useful for modeling probabilistic behavior of agents who may make suboptimal choices. To run the test cases, run <br />

python autograder.py -q q4 <br />

To see how the ExpectimaxAgent behaves in Pacman, run: <br />

python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
