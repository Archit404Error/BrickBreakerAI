# BrickBreakerAI
## What it does
This AI automatically generates thousands of Brick Breaker players who compete with one another to break the most amount of bricks possible. The players automatically learn and grow over time, until they become artificially intelligent enough to beat the game. The NEAT evolutionary algorithm was used to develop this simulation, along-side pygame for the brick breaker game itself.

## Training the AI
Simply run 
> git clone github.com/Archit404Error/BrickBreaker

Once you've successfully cloned the repo, all you have to do is cd into the folder and run the following command:
```python
python3 main.py
```

After that, the simulation will begin to run and train automatically!

## Adjusting the AI
Screen width and height can be found at the beginning of the main.py file and can be manually adjusted. Evolutionary parameters(number of players, number of input/output/hidden layers, etc.) can be found and adjusted in the config-feedforward.txt file.

Additionally, the fitness function currently takes into account the amount of bricks a given player has broken and the distance from the player to the ball at all times. This calculation can be modified in the getFitness() function in main.py
