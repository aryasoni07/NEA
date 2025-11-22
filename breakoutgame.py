import os
import json
import random
import numpy as np
import pyglet
from pyglet import shapes
from paddle import Paddle
from ball import Ball
from brick import Brick

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

        self.batch = pyglet.graphics.Batch()
        self.paddle = Paddle(x=290,y=0,width=60,height=10,batch=self.batch,window_width=self.window.width,)
        self.ball = Ball(x=320, y=100, size=10, batch=self.batch)

        self.bricks = []
        self.createBricks()

    def loadScores(self):
        if os.path.exists(self.scoresFile):
            with open(self.scoresFile, "r") as f:
                self.scoresData = json.load(f)

    def saveScores(self):
        with open(self.scoresFile, "w") as f:
            json.dump(self.scoresData, f)

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

    def get_nn_input(self):
        nx = self.ball.x / 630.0
        ny = self.ball.y / 480.0
        ndx = self.ball._dx / 200.0
        ndy = self.ball._dy / 200.0
        pa = self.paddle.x / 580.0
        vec = [nx, ny, ndx, ndy, pa]
        return np.array(vec, dtype=np.float32).reshape(-1, 1)

    @staticmethod
    def predict_landing(x, y, dx, dy):
        if dx == 0:
            fx = x
        elif dy<0:
            fx = x-(y-60)*(dx/dy)
        else:
            fx = x+(dx/dy)*(950-y)

        while fx<0 or fx>630:
            if fx<0:
                fx = abs(fx)
            elif fx>630:
                fx = 1260-fx
        return fx

    @classmethod
    def get_training_input_basic(cls):
        x = random.uniform(0, 630)
        y = random.uniform(0, 480)
        dx = random.choice([-200.0, 200.0])
        dy = random.choice([-200.0, 200.0])
        paddlex = random.uniform(0, 580)
        nx = x / 630.0
        ny = y / 480.0
        ndx = dx / 200.0
        ndy = dy / 200.0
        pa = paddlex / 580.0
        vec = [nx, ny, ndx, ndy, pa]

        fx = cls.predict_landing(x, y, dx, dy)
        if fx < paddlex + 20:
            ans = 0
        elif fx > paddlex + 40:
            ans = 2
        else:
            ans = 1
        return np.array(vec, dtype=np.float32).reshape(-1, 1), ans

    @classmethod
    def get_training_input_advanced(cls):
        x = random.uniform(0, 630)
        y = random.uniform(0, 480)
        dy = random.choice([-200.0, 200.0])
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

    @staticmethod
    def update_accuracy(history, isCorrect):
        history.append(1 if isCorrect else 0)
        if len(history) > 1000:
            history.pop(0)
        percent = 100.0 * sum(history) / len(history)
        return f"Accuracy: {percent:.1f}%"

    def ai_move(self, nn, dt):
        inp = self.get_nn_input()
        scores = nn.forprop(inp)
        action = int(np.argmax(scores))
        if action == 0:
            self.paddle.move_left(dt)
        elif action == 2:
            self.paddle.move_right(dt)

    def common_update_solo(self, dt, scoreKey):
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

        self.elapsedTime += dt
        if self.keys[pyglet.window.key.LEFT]:
            self.paddle.move_left(dt)
        if self.keys[pyglet.window.key.RIGHT]:
            self.paddle.move_right(dt)

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
            self.ball.x = (
                self.paddle.x + self.paddle.width // 2 - self.ball.width // 2
            )
            self.ball.y = 1
            self.createBricks()

    def update_solo_basic(self, dt):
        self.common_update_solo(dt, "Solo Mode Basic")

        if (self.paddle.y <= self.ball.y <= self.paddle.y + self.paddle.height
        ) and (self.paddle.x-self.ball.width <= self.ball.x <= self.paddle.x+self.paddle.width+self.ball.width ):
            self.ball._dy = abs(self.ball._dy) 

    def update_solo_advanced(self, dt):
        self.common_update_solo(dt, "Solo Mode Advanced")

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

    def update_training_basic(self, dt):
        self.timerLabel.text = f"Time elapsed: {self.elapsedTime:.2f}"
        self.elapsedTime += dt

        inp, ans = self.get_training_input_basic()
        scores_now = self.nn_basic.forprop(inp)
        pred = int(np.argmax(scores_now))
        self.accuracyLabel.text = self.update_accuracy(self.accuracyBasicHistory, pred==ans)

        dW, db = self.nn_basic.backprop(inp, ans)
        self.nn_basic.record_grads(dW, db)
        self.nn_basic.apply_grads(self.learningRate)

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

    def update_watch_basic(self, dt):
        self.common_update_watch(dt, self.nn_basic, "Watch Mode Basic")

        if (self.paddle.y <= self.ball.y <= self.paddle.y + self.paddle.height
        ) and (self.paddle.x-self.ball.width <= self.ball.x <= self.paddle.x+self.paddle.width+self.ball.width ):
            self.ball._dy = abs(self.ball._dy)

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