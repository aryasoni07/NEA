#BreakoutGame

import os
import json
import random
import numpy as np
import pyglet
from pyglet import shapes
from paddle import Paddle
from ball import Ball
from brick import Brick

#GameState
#Enumeration class to store the different states the game can be in.
class GameState:
    MENU = "menu"
    SOLO_MODE_MENU = "solomodemenu"
    TRAINING_MODE_MENU = "trainingmodemenu"
    WATCH_MODE_MENU = "watchmodemenu"
    SCORES_MENU = "scoresmenu"
    SOLO_BASIC = "solobasic"
    SOLO_ADVANCED = "soloadvanced"
    TRAINING_BASIC = "trainingbasic"
    TRAINING_ADVANCED = "trainingadvanced"
    WATCH_BASIC = "watchbasic"
    WATCH_ADVANCED = "watchadvanced"
    GAME_OVER = "gameover"
    GAME_WON = "gamewon"

class BreakoutGame:

    def __init__(self, window, keys,
        timerLabel, livesLabel, scoreLabel, stageLabel, accuracyLabel, gameoverScoreLabel,
        nn_basic, nn_advanced, scoresFile="scores.json", learningRate=0.001):

        self.window = window
        self.keys = keys

        self.timerLabel = timerLabel
        self.livesLabel = livesLabel
        self.scoreLabel = scoreLabel
        self.stageLabel = stageLabel
        self.accuracyLabel = accuracyLabel
        self.gameoverScoreLabel = gameoverScoreLabel

        self.nn_basic = nn_basic
        self.nn_advanced = nn_advanced
        self.learningRate = learningRate
        self.accuracyBasicHistory = []
        self.accuracyAdvancedHistory = []

        self.scoresFile = scoresFile
        self.scoresData = {"Solo Mode Basic": 0,
            "Solo Mode Advanced": 0,
            "Watch Mode Basic": 0,
            "Watch Mode Advanced": 0,}
        self.loadScores()

        self.gameState = GameState.MENU
        self.elapsedTime = 0.0
        self.lives = 3
        self.gameStarted = False
        self.score = 0
        self.stage = 1
        self.respawnTime = 1.0
#Ball, Paddle, Bricks are grouped by Pyglet in a batch. This means that these objects are drawn together.
        self.batch = pyglet.graphics.Batch()
        self.paddle = Paddle(x=290,y=0,width=60,height=10,batch=self.batch,window_width=self.window.width,)
        self.ball = Ball(x=320, y=100, size=10, batch=self.batch)

        self.bricks = []
        self.createBricks()

#Saves the high scores in a json file
    def saveScores(self):
        with open(self.scoresFile, "w") as f:
            json.dump(self.scoresData, f)

#Loads the saved high scores
    def loadScores(self):
        if os.path.exists(self.scoresFile):
            with open(self.scoresFile, "r") as f:
                self.scoresData = json.load(f)

#Inital variable values for when a new game is begun
    def resetGame(self):
        self.ball.x = self.paddle.x + self.paddle.width // 2 - self.ball.width // 2
        self.ball.y = 10
        self.ball._dx = 0
        self.ball._dy = 0
        self.paddle.x = 270
        self.elapsedTime = 0.0
        self.lives = 3
        self.score = 0
        self.stage = 1
        self.gameStarted = False
        self.respawnTime = 1.0
        self.ball.color = (255, 0, 0)
        self.createBricks()

#Makes a new set of bricks, in the required layout
    def createBricks(self):
        self.bricks = []
        rows = 5
        cols = 11
        brickWidth = 60
        brickHeight = 20
        topOffset = 60
        totalWidth = cols * brickWidth
        startx = (self.window.width - totalWidth) // 2
        for row in range(rows):
            for col in range(cols):
                x = startx + col * brickWidth
                y = 480 - topOffset - row * brickHeight
                self.bricks.append(Brick(x, y, brickWidth, brickHeight, self.batch))

#Normalises the game state values to be input into a neural network
    def get_nn_input(self):
        nx = self.ball.x / 630.0
        ny = self.ball.y / 480.0
        ndx = self.ball._dx / 200.0
        ndy = self.ball._dy / 200.0
        pa = self.paddle.x / 580.0
        vec = [nx, ny, ndx, ndy, pa]
        return np.array(vec, dtype=np.float32).reshape(-1, 1)

#Predicts the ball's landing position, for a training instance
    @staticmethod
    def predict_landing(x, y, dx, dy):

#If the vertical component of velocity is zero, it will land at the same x-coordinate
        if dx == 0:
            fx = x

#Calculates ball landing position from when the ball is moving downwards
        elif dy<0:
            fx = x-(y-10)*(dx/dy)
            
#Calculates ball landing position from when the ball is moving upwards
        else:
            fx = x+(dx/dy)*(950-y)

#Ensures the predicted landing is within the window dimensions
        while fx<0 or fx>630:
            if fx<0:
                fx = abs(fx)
            elif fx>630:
                fx = 1260-fx
        return fx

#Creates a training instance for basic mode
    @classmethod
    def get_training_input_basic(cls):

#Generates a random ball position in line with what is possible in basic mode
        x = random.uniform(0, 630)
        y = random.uniform(0, 480)
        dx = random.choice([-200.0, 200.0])
        dy = random.choice([-200.0, 200.0])
        paddlex = random.uniform(0, 580)

#Normalises the values to input into the neural network
        nx = x / 630.0
        ny = y / 480.0
        ndx = dx / 200.0
        ndy = dy / 200.0
        pa = paddlex / 580.0
        vec = [nx, ny, ndx, ndy, pa]

#Gets the x-value of where the ball is going to land
        fx = cls.predict_landing(x, y, dx, dy)

#Determines whether the paddle must be moved to the left or right
#Keeps a buffer of 20 pixels (paddle width is 60 pixels) where the correct answer is to keep still.
        if fx < paddlex + 20:
            ans = 0
        elif fx > paddlex + 40:
            ans = 2
        else:
            ans = 1
        return np.array(vec, dtype=np.float32).reshape(-1, 1), ans

#Generates a training instance for advanced mode
    @classmethod
    def get_training_input_advanced(cls):
        x = random.uniform(0, 630)
        y = random.uniform(0, 480)
        dy = random.choice([-200.0, 200.0])
#In advanced mode, dx can vary, so the random value is taken from a normal distribution
#The mean is randomly chosen to be a value that represents either positive or negative horizontal velocity
        mean = -200 if random.random() < 0.5 else 200
        dx = random.gauss(mu=mean, sigma=100)

        paddlex = random.uniform(0, 580)
        nx = x / 630.0
        ny = y / 480.0
        ndx = dx / 200.0
        ndy = dy / 200.0
        pa = paddlex / 580.0
        vec = [nx, ny, ndx, ndy, pa]

        fx = cls.predict_landing(x, y, dx, dy)
        if fx < paddlex + 30:
            ans = 0
        elif fx > paddlex + 30:
            ans = 2
        else:
            ans = 1
        return np.array(vec, dtype=np.float32).reshape(-1, 1), ans

#Updates the accuracy, looking at the results of a sliding window of the last 1000 training instances
    @staticmethod
    def update_accuracy(history, isCorrect):
        history.append(1 if isCorrect else 0)
        if len(history) > 1000:
            history.pop(0)
        percent = 100.0 * sum(history) / len(history)
        return f"Accuracy: {percent:.1f}%"

#Performs the move decided by the neural network for the current game state
    def ai_move(self, nn, dt):
        inp = self.get_nn_input()
        scores = nn.forprop(inp)
        action = int(np.argmax(scores))
        if action == 0:
            self.paddle.move_left(dt)
        elif action == 2:
            self.paddle.move_right(dt)

#Updated every frame, includes checks that are common to both basic and advanced modes within solo mode
    def common_update_solo(self, dt, scoreKey):
        self.timerLabel.text = f"Time elapsed: {self.elapsedTime:.2f}"
        self.livesLabel.text = f"Lives: {self.lives}"
        self.scoreLabel.text = f"Score: {self.score}"
        self.stageLabel.text = f"Stage: {self.stage}"

        if self.ball.y <= 0:
            self.gameStarted = False
#Pauses gameplay for 1 second when the ball hits the ground, ball also fades to white
            if self.respawnTime > 0.0:
                self.respawnTime -= dt
                progress = 1 - self.respawnTime / 1.0
                self.ball.color = (255, int(255 * progress), int(255 * progress))
                return
            else:
#A life is lost when the ball hits the ground
                self.lives -= 1
#When all lives are lost, if a highscore is attained, it is saved, and the mode is exited
                if self.lives == 0:
                    if self.score > self.scoresData[scoreKey]:
                        self.scoresData[scoreKey] = self.score
                        self.saveScores()
                    self.gameoverScoreLabel.text = f"Score: {self.score}"
                    self.gameState = GameState.GAME_OVER
                    self.resetGame()
                    return
                self.respawnTime = 1.0
                self.ball.color = (255, 0, 0)

#If the game has not started, the timer is inactive and the ball is stuck to the paddle, which can move
#This can be exited py pressing the up arrow key, and the direction will be a random choice between diagonally left or right
        if not self.gameStarted:
            if self.keys[pyglet.window.key.UP]:
                self.ball._dx = random.choice([-200, 200])
                self.ball._dy = 200
                self.gameStarted = True
            else:
                if self.keys[pyglet.window.key.LEFT]:
                    self.paddle.move_left(dt)
                if self.keys[pyglet.window.key.RIGHT]:
                    self.paddle.move_right(dt)
                self.ball.x = (self.paddle.x + self.paddle.width/2 - self.ball.width/2)
                self.ball.y = 10
                return
#Timer is updated
        self.elapsedTime += dt

#Paddle moves in response to key presses
        if self.keys[pyglet.window.key.LEFT]:
            self.paddle.move_left(dt)
        if self.keys[pyglet.window.key.RIGHT]:
            self.paddle.move_right(dt)

#Ball moves depending on its velocity
        self.ball.update_position(dt)

#Checks each brick whether it will be hit by the ball, the score will be updated if so
        for brick in self.bricks:
            if brick.checkCollision(self.ball):
                self.score += 1
                self.scoreLabel.text = f"Score: {self.score}"
                break

#Checks for wall collision, updates velocity if so
        if self.ball.x + self.ball._dx / 60 <= 0:
            self.ball._dx = abs(self.ball._dx)
        elif self.ball.x > self.window.width - self.ball.width:
            self.ball._dx = -abs(self.ball._dx)

#Checks for ceiling collision, updates velocity if so
        if self.ball.y > 480 - self.ball.height:
            self.ball._dy = -abs(self.ball._dy)

#When all bricks are broken, all bricks are restored and we move on to the next stage
        if all(not brick.alive for brick in self.bricks):
            self.stage += 1
            self.stageLabel.text = f"Stage: {self.stage}"
            self.gameStarted = False
            self.ball._dx = 0
            self.ball._dy = 0
            self.ball.x = (self.paddle.x + self.paddle.width//2 - self.ball.width//2)
            self.ball.y = 1
            self.createBricks()

#Performs checks every frame, for solo basic mode
    def update_solo_basic(self, dt):

#Performs checks common to both solo modes
        self.common_update_solo(dt, "Solo Mode Basic")

#Paddle collision handling is unique to the mode, so is checked here
        if (self.paddle.y <= self.ball.y <= self.paddle.y + self.paddle.height
        ) and (self.paddle.x-self.ball.width <= self.ball.x <= self.paddle.x+self.paddle.width+self.ball.width ):
            self.ball._dy = abs(self.ball._dy) 

    def update_solo_advanced(self, dt):
        self.common_update_solo(dt, "Solo Mode Advanced")

#Paddle collision handling
#Velocity direction changes to be further from normal when hitting the edge of the paddle, closer when hitting the centre of the paddle
        if (self.ball.y + self.ball._dy / 60 <= self.paddle.y + self.paddle.height
        ) and (self.paddle.x-self.ball.width <= self.ball.x <= self.paddle.x+self.paddle.width):
            self.ball._dy = abs(self.ball._dy)
            mid = self.paddle.x + self.paddle.width/2
            midx = self.ball.x + self.ball.width/2
            if (self.paddle.x - self.ball.width/2 <= midx <= self.paddle.x + self.paddle.width/4
            ) or (self.paddle.x+3*(self.paddle.width/4) <= midx <= self.paddle.x+self.paddle.width+self.ball.width/2):
                self.ball._dx *= 1.2
            elif mid < midx < self.paddle.x+3*(self.paddle.width/4):
                self.ball._dx *= 0.8

#Performed every frame in training basic mode
    def update_training_basic(self, dt):
        self.timerLabel.text = f"Time elapsed: {self.elapsedTime:.2f}"
        self.elapsedTime += dt

#Passes game state to neural network, compares output to predicted landing
        inp, ans = self.get_training_input_basic()
        scores_now = self.nn_basic.forprop(inp)
        pred = int(np.argmax(scores_now))
        self.accuracyLabel.text = self.update_accuracy(self.accuracyBasicHistory, pred==ans)

#Determines the change required for this game state, grouped every 10 instances
#The neural network parameters are updated by the mean of the group of 10
        dW, db = self.nn_basic.backprop(inp, ans)
        self.nn_basic.record_grads(dW, db)
        self.nn_basic.apply_grads(self.learningRate)

#Performed every frame in training basic mode
#Identical to update_training_basic but interacts with nn_advanced
    def update_training_advanced(self, dt):
        self.timerLabel.text = f"Time elapsed: {self.elapsedTime:.2f}"
        self.elapsedTime += dt

        inp, ans = self.get_training_input_advanced()
        scores_now = self.nn_advanced.forprop(inp)
        pred = int(np.argmax(scores_now))
        self.accuracyLabel.text = self.update_accuracy(self.accuracyAdvancedHistory, pred==ans)

        dW, db = self.nn_advanced.backprop(inp, ans)
        self.nn_advanced.record_grads(dW, db)
        self.nn_advanced.apply_grads(self.learningRate)

#Updated every frame, includes checks that are common to both basic and advanced modes within watch mode
#Similar to common_update_solo
    def common_update_watch(self, dt, nn, scoreKey):
        self.timerLabel.text = f"Time elapsed: {self.elapsedTime:.2f}"
        self.livesLabel.text = f"Lives: {self.lives}"
        self.scoreLabel.text = f"Score: {self.score}"
        self.stageLabel.text = f"Stage: {self.stage}"

        if self.ball.y <= 0:
            self.gameStarted = False
            if self.respawnTime > 0.0:
                self.respawnTime -= dt
                progress = 1 - self.respawnTime / 1.0
                self.ball.color = (255, int(255 * progress), int(255 * progress))
                return
            else:
                self.ball.x = (self.paddle.x + self.paddle.width/2 - self.ball.width/2)
                self.ball.y = self.paddle.height + 1
                self.lives -= 1
                if self.lives == 0:
                    if self.score > self.scoresData[scoreKey]:
                        self.scoresData[scoreKey] = self.score
                        self.saveScores()
                    self.gameoverScoreLabel.text = f"Score: {self.score}"
                    self.gameState = GameState.GAME_OVER
                    self.resetGame()
                    return
                self.respawnTime = 1.0
                self.ball.color = (255, 0, 0)

        if not self.gameStarted:
            self.ball._dx = random.choice([-200, 200])
            self.ball._dy = 200
            self.gameStarted = True

        self.elapsedTime += dt

#Paddle changes are taken from the neural network
        self.ai_move(nn, dt)

        self.ball.update_position(dt)

        for brick in self.bricks:
            if brick.checkCollision(self.ball):
                self.score += 1
                self.scoreLabel.text = f"Score: {self.score}"
                break

        if self.ball.x + self.ball._dx / 60 <= 0:
            self.ball._dx = abs(self.ball._dx)
        elif self.ball.x > self.window.width - self.ball.width:
            self.ball._dx = -abs(self.ball._dx)

        if self.ball.y > 480 - self.ball.height:
            self.ball._dy = -abs(self.ball._dy)

        if all(not brick.alive for brick in self.bricks):
            self.stage += 1
            self.stageLabel.text = f"Stage: {self.stage}"
            self.gameStarted = False
            self.ball._dx = 0
            self.ball._dy = 0
            self.ball.x = (self.paddle.x + self.paddle.width//2 - self.ball.width//2)
            self.ball.y = 1
            self.createBricks()

#Performs checks every frame, for watch basic mode
    def update_watch_basic(self, dt):
        self.common_update_watch(dt, self.nn_basic, "Watch Mode Basic")

        if (self.paddle.y <= self.ball.y <= self.paddle.y + self.paddle.height
        ) and (self.paddle.x-self.ball.width <= self.ball.x <= self.paddle.x+self.paddle.width+self.ball.width ):
            self.ball._dy = abs(self.ball._dy)

#Performs checks every frame, for watch advanced mode
    def update_watch_advanced(self, dt):
        self.common_update_watch(dt, self.nn_advanced, "Watch Mode Advanced")

        if (self.ball.y + self.ball._dy / 60 <= self.paddle.y + self.paddle.height
        ) and (self.paddle.x-self.ball.width <= self.ball.x <= self.paddle.x+self.paddle.width):
            self.ball._dy = abs(self.ball._dy)
            mid = self.paddle.x + self.paddle.width/2
            midx = self.ball.x + self.ball.width/2
            if (self.paddle.x - self.ball.width/2 <= midx <= self.paddle.x + self.paddle.width/4
            ) or (self.paddle.x+3*(self.paddle.width/4) <= midx <= self.paddle.x+self.paddle.width+self.ball.width/2):
                self.ball._dx *= 1.25
            elif mid < midx < self.paddle.x+3*(self.paddle.width/4):
                self.ball._dx *= 0.75

#When update is called in the main file, the update is chosen depending on the game state
    def update(self, dt):
        if self.gameState == GameState.SOLO_BASIC:
            self.update_solo_basic(dt)
        elif self.gameState == GameState.SOLO_ADVANCED:
            self.update_solo_advanced(dt)
        elif self.gameState == GameState.TRAINING_BASIC:
            self.update_training_basic(dt)
        elif self.gameState == GameState.TRAINING_ADVANCED:
            self.update_training_advanced(dt)
        elif self.gameState == GameState.WATCH_BASIC:
            self.update_watch_basic(dt)
        elif self.gameState == GameState.WATCH_ADVANCED:
            self.update_watch_advanced(dt)