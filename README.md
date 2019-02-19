# DQN Block Game

This project is to create an AI that can play a game slightly modified from a famous mobile game "[Ten Ten](https://play.google.com/store/apps/details?id=com.gramgames.tenten&hl=en_US)". The framework of this AI is Deep Reinforcement Learning, and [Q-Learning](https://en.wikipedia.org/wiki/Q-learning).

## Game Rule
1. You will begin with a grid board of 10 by 10
2. In each step, the game will randomly spawn a block, you need to place the block to somewhere empty on the board
3. If a line (row or column) is fully filled, this line will vanish and provide you with more space
4. If current bolck can not be legitimately put anywhere on the board, then game is over
5. The goal is to survive as many steps as possible

## Reinforcement Learning
Using Q-learning framework in this project. The essense of Q learning is bellman equation:
![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/47fa1e5cf8cf75996a777c11c7b9445dc96d4637)

Similar to dynamic programming, this algorithm is doing a step-wise update on Q. In each step, the agent sees a new state of the game and using its current knowledge to (roughly) forecast the value of possible states in next step and using that value to (more accurately) evaluate the value of current state. Then it will feed in the current state and its evaluated value to train the Q-function, which is realized by a neural network, with a small amount of (typically just one) epoch. Click [here](https://en.wikipedia.org/wiki/Q-learning) to learn more about Q learning.

In this project, here is the definition of key variables in this algorithm:

**Agent:** This AI

**Action(A):** Choose a place to put the current block

**Environment:** The game rule introduced above

**State(S):** The grid map of game board

**Reward:** Max(0, # of grids vanished by this action)

**Policy(π):** The strategy to play game, i.e. where to place block is best solution given the board map, in this project, we choose the action with max(Q) to put the block

**Q-value:** The value of taking action **A** under policy **π** given state **S<sub>t</sub>**

## Performance and Discussion

The agent learns gradually how to play. In beginning stage, when policy is basically to random choose an action, the agent can survive about 30-60 steps. After about 1700 episode of game, it can survive hundreds of steps. The training process took a whole night to complete. If we keep training, it can be expected that the performance will keep increase.


![alt text](https://github.com/Guanzy2224/DQN-Block-Game/blob/master/Analysis/Steps%20survived.png)

However, even after a thousand of games of training, it will still take bad actions sometimes. One reason could be that the agent will act poorly when it met some state that it has never seen anything similar before. A fact is that when the agent is surprised by a new state, the training loss will be higher. So we can use the training loss to prove this hypothesis:

![alt text](https://github.com/Guanzy2224/DQN-Block-Game/blob/master/Analysis/Episode%201%20-%209.png)
**Training Loss of episodes 1-9**


![alt text](https://github.com/Guanzy2224/DQN-Block-Game/blob/master/Analysis/Episode%201001%20-%201009.png)
**Training Loss of episodes 1001-1009**


In the early stage of training process, training loss decreases as steps increase, but it occassionally surprised by the unseen state. It estimates the state to be just, but the deduction of next step shows this is a dead end, so the agent is surprised. This is reflected by high training loss. This happens less frequently in later stage of training process, but when it happened, the game is likely to end.

Another possible explanation is overfitting. Overfitting will guide the agent to learn harder on some unimportant states and ignore other more important states.

To make the agent works better, there are a few choices I came up with:

1. Using 2D convolution. Convolution filter may help the agent recognize the shape of some typical local states.

2. Using DFS. Exploit the next 2 to 3 steps instead of only 1 step would result in a more correct evaluation of Q of current state and train the agent with smarter data.

3. Any other ideas? Welcome to discuss.
