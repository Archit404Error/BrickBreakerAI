import pygame
import os
import sys
from random import *
from math import *
import neat

screenWidth = 1200
screenHeight = 600
generation = 0
startBrickAmt = 0

class Brick:
    def __init__(self, x, y, c):
        self.width = 75
        self.height = 20
        self.hit = False
        self.pos = [x, y]
        self.color = c

    def isHit(self, ball):
        brickBox = pygame.Rect(self.pos[0], self.pos[1], self.width, self.height)
        ballBox = pygame.Rect(ball.pos[0], ball.pos[1], ball.width, ball.height)
        if brickBox.colliderect(ballBox):
            self.hit = True
            ball.setVel(ball.velX, ball.velY * -1)

    def draw(self, screen):
        if not self.hit:
            pygame.draw.rect(screen, self.color, (self.pos[0], self.pos[1], self.width, self.height))

class BrickWall:
    def __init__(self, c):
        self.bricks = []
        self.color = c
        for i in range(0, screenWidth - 75, 85):
            for j in range(0, int(screenHeight / 2) - 20, 30):
                self.bricks.append(Brick(i, j, self.color))
        global startBrickAmt
        if startBrickAmt == 0:
            startBrickAmt = len(self.bricks)

    def draw(self, screen, ball):
        for b in self.bricks:
            if not b.hit:
                b.isHit(ball)
            else:
                self.bricks.pop(self.bricks.index(b))
                continue
            b.draw(screen)

class Ball:
    def __init__(self, c):
        self.pos = [int(screenWidth / 2), int(screenHeight / 2) + 100]
        self.height = 20
        self.width = 20
        self.color = c
        self.velX = -10
        self.velY = -10

    def paddleHit(self, pad):
        ballBox = pygame.Rect(self.pos[0], self.pos[1], self.width, self.height)
        padBox = pygame.Rect(pad.pos[0], pad.pos[1], pad.width, pad.height)
        if(ballBox.colliderect(padBox)):
            self.velY *= -1
            if self.pos[0] > pad.pos[0] + pad.width or self.pos[0] < pad.pos[0]:
                self.pos[1] += 50
            elif self.pos[0] < (pad.pos[0] + pad.width / 2):
                self.velX = -1 * abs(self.velX)
            else:
                self.velX = abs(self.velX)

    def setVel(self, x, y):
        self.velX = x
        self.velY = y

    def draw(self, screen, pad):
        self.pos = [self.pos[0] + self.velX, self.pos[1] + self.velY]
        if self.pos[0] >= screenWidth - self.width or self.pos[0] <= 0:
            self.velX *= -1
        if self.pos[1] <= 0:
            self.velY *= -1
        self.paddleHit(pad)
        pygame.draw.rect(screen, self.color, (self.pos[0], self.pos[1], self.width, self.height))

class Paddle:
    def __init__(self, c):
        self.height = 20
        self.width = 150
        self.color = c
        self.velX = 10
        self.pos = [int(screenWidth / 2) - self.width / 2, int(screenHeight / 2) + 200 - self.height]

    def move(self, val):
        if val == 1 and self.pos[0] <= screenWidth - self.width - self.velX:
            self.pos[0] += self.velX
        elif val == 2 and self.pos[0] >= 0:
            self.pos[0] -= self.velX

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.pos[0], self.pos[1], self.width, self.height))

class BrickPair:
    def __init__(self):
        col = (randint(0, 255), randint(0, 255), randint(0, 255))
        self.paddle = Paddle(col)
        self.ball = Ball(col)
        self.wall = BrickWall(col)
        self.currBricks = len(self.wall.bricks)
        self.timeSinceHit = 0
        self.alive = True

    def isAlive(self):
        if self.ball.pos[1] >= screenHeight - self.ball.height:
            self.alive = False
        if self.alive:
            if len(self.wall.bricks) < self.currBricks:
                self.currBricks = len(self.wall.bricks)
                self.timeSinceHit = 0
            else:
                self.timeSinceHit += 1
                if self.timeSinceHit == 1250:
                    self.alive = False
        return self.alive

    def getInfo(self):
        return [self.ball.pos[0], self.ball.pos[1], self.paddle.pos[0], self.paddle.pos[1], self.wall.bricks[len(self.wall.bricks) - 1].pos[0], self.wall.bricks[len(self.wall.bricks) - 1].pos[1]]

    def getFitness(self):
        return (startBrickAmt - len(self.wall.bricks)) + (sqrt(screenWidth**2 + screenHeight**2) - sqrt((self.ball.pos[0] - self.paddle.pos[0]) ** 2 + (self.ball.pos[1] - self.paddle.pos[1]) ** 2))/10

    def draw(self, screen):
        if self.alive:
            self.isAlive()
            self.paddle.draw(screen)
            self.ball.draw(screen, self.paddle)
            self.wall.draw(screen, self.ball)

def runGame(genomes, config):
    networks = []
    players = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        networks.append(net)
        g.fitness = 0
        players.append(BrickPair())

    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    clock = pygame.time.Clock()
    global generation
    generation += 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        clock.tick(40)
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, screenWidth, screenHeight))

        for i, player in enumerate(players):
            output = networks[i].activate(player.getInfo())
            val = output.index(max(output))
            player.paddle.move(val)

        alive = 0
        for i, player in enumerate(players):
            if player.isAlive():
                alive += 1
                genomes[i][1].fitness += player.getFitness()

        if alive == 0:
            break

        for player in players:
            player.draw(screen)
        pygame.display.update()
    pygame.display.flip()

if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.run(runGame, 1000)
