import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

import numpy as np
import os
import pickle
import atexit

class Brick:
    def __init__(self, x, y, width, height, color=(100, 200, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shape = shapes.Rectangle(x, y, width, height, color=color, batch=batch)
        self.alive = True
    def checkCollision(self, ball):
        if not self.alive:
            return False
        nextx = ball.x + ball.width/2 + ball.dx/60 #make sure this number is equal to the fps
        nexty = ball.y + ball.width/2 + ball.dy/60
        brickLeft, brickRight, brickTop, brickBottom = self.x, self.x+self.width, self.y, self.y+self.height
        ballLeft, ballRight, ballTop, ballBottom = nextx, nextx+ball.width, nexty, nexty+ball.height
        if (ballRight > brickLeft and ballLeft < brickRight and
            ballBottom > brickTop and ballTop < brickBottom):
            overlapLeft, overlapRight, overlapTop, overlapBottom = ballRight-brickLeft, brickRight-ballLeft, ballBottom-brickTop, brickBottom-ballTop
            overlapx = min(overlapLeft, overlapRight)
            overlapy = min(overlapTop, overlapBottom)
            if overlapx < overlapy:
                ball.dx *= -1
            else:
                ball.dy *= -1
            self.alive = False
            self.shape.delete()
            return True
        return False
    
logicalWidth, logicalHeight = 64, 48
scaleFactor = 10

window = pyglet.window.Window(logicalWidth*scaleFactor, logicalHeight*scaleFactor, caption="Breakout")
keys = key.KeyStateHandler()
window.push_handlers(keys)
state = "menu" 
elapsedTime = 0.0
lives = 5
gameStarted = False
score = 0
stage = 1

titleLabel = pyglet.text.Label("Breakout", font_size=36, x=window.width//2, y=350, anchor_x='center')
soloButton = shapes.Rectangle(220, 250, 200, 50, color=(100, 100, 255))
soloButtonText = pyglet.text.Label("Solo Mode", x=window.width//2, y=275, anchor_x='center', anchor_y='center')
solobasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
solobasicButtonText = pyglet.text.Label("Basic Mode", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
soloadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
soloadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
watchButton = shapes.Rectangle(220, 150, 200, 50, color=(100, 100, 255))
watchButtonText = pyglet.text.Label("Watch Mode", x=window.width//2, y=175, anchor_x='center', anchor_y='center')
watchbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
watchbasicButtonText = pyglet.text.Label("Watch Basic", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
watchadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
watchadvancedButtonText = pyglet.text.Label("Watch Advanced", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
trainingButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
trainingButtonText = pyglet.text.Label("Training Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
trainingbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
trainingbasicButtonText = pyglet.text.Label("Basic Mode", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
trainingadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
trainingadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
scoresButton = shapes.Rectangle(220, 100, 200, 50, color=(100, 100, 255))
scoresButtonText = pyglet.text.Label("Scores", x=window.width//2, y=125, anchor_x='center', anchor_y='center')

timerLabel = pyglet.text.Label("Time elapsed: 0.00", x=10, y=450)
livesLabel = pyglet.text.Label("Lives: 3", x=180, y=450)
scoreLabel = pyglet.text.Label("Score: 0", x=280, y=450)
stageLabel = pyglet.text.Label("Stage: 1", x=380, y=450)
batch = pyglet.graphics.Batch()
paddle = shapes.Rectangle(x=270, y=00, width=60, height=10, color=(255, 255, 255), batch=batch)
ball = shapes.Rectangle(x=320, y=100, width=10, height=10, color=(255, 0, 0), batch=batch)
ball.dx = 200
ball.dy = 200
bricks = []

exitButton = shapes.Rectangle(540, 440, 100, 40, color=(200, 50, 50))
exitButtonText = pyglet.text.Label("Exit", x=590, y=460, anchor_x='center', anchor_y='center')
gameoverLabel = pyglet.text.Label("GAME OVER", font_size=32, x=window.width//2, y=300, anchor_x='center')
restartButton = shapes.Rectangle(220, 200, 200, 50, color=(255, 100, 100))
restartButtonText = pyglet.text.Label("Return to Menu", x=window.width//2, y=225, anchor_x='center', anchor_y='center')

scoresFile = "scores.pkl"
scoresData = {"soloBasic": 0, "soloAdvanced": 0, "trainingBasic": 0, "trainingAdvanced": 0, "watchBasic": 0, "watchAdvanced": 0}
def loadScores():
    global scoresData
    if os.path.exists(scoresFile):
        with open(scoresFile, "rb") as f:
            scoresData = pickle.load(f)
def saveScores():
    with open(scoresFile, "wb") as f:
        pickle.dump(scoresData, f)

class NeuralNetwork:
    def __init__(self, layerSizes, saveFile):
        self.layerSizes = layerSizes
        self.saveFile = saveFile
        self.weights = []
        self.biases = []

        if os.path.exists(self.saveFile):
            self.load()
        else:
            self.initRandomWeights()

    def initRandomWeights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layerSizes) - 1):
            fan_in = self.layerSizes[i]
            fan_out = self.layerSizes[i+1]
            std = np.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_out, fan_in) * std
            b = np.zeros((fan_out, 1))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        a = x
        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ a + self.biases[i]
            a = np.maximum(z, 0.0)
        z = self.weights[-1] @ a + self.biases[-1]
        return z

    def save(self):
        with open(self.saveFile, "wb") as f:
            pickle.dump({"weights": self.weights, "biases": self.biases}, f)

    def load(self):
        with open(self.saveFile, "rb") as f:
            data = pickle.load(f)
            self.weights = data["weights"]
            self.biases = data["biases"]

nn_basic = NeuralNetwork([59, 32, 16, 3], saveFile="nn_basic.pkl")
nn_advanced = NeuralNetwork([59, 50, 30, 3], saveFile="nn_advanced.pkl")

atexit.register(nn_basic.save)
atexit.register(nn_advanced.save)

def resetGame():
    global ball, paddle, elapsedTime, state, lives, gameStarted, score, stage
    ball.x = paddle.x+paddle.width//2-ball.width//2
    ball.y = 10
    ball.dx, ball.dy = 0, 0
    paddle.x = 270
    elapsedTime = 0
    lives = 5
    score = 0
    stage = 1
    gameStarted = False
    createBricks()

def createBricks():
    global bricks
    bricks = []
    rows = 5
    cols = 11
    brickWidth = 60
    brickHeight = 20
    topOffset = 60
    totalWidth = cols*brickWidth
    startX = (window.width - totalWidth)//2
    for row in range(rows):
        for col in range(cols):
            x = startX + col*brickWidth
            y = window.height-topOffset-row*brickHeight
            bricks.append(Brick(x, y, brickWidth, brickHeight))

def get_nn_input_basic():
    brickBits = [1 if brick.alive else 0 for brick in bricks]

    nx = ball.x / window.width
    ny = ball.y / window.height

    scale = 300.0
    ndx = max(-1.0, min(1.0, ball.dx / scale))
    ndy = max(-1.0, min(1.0, ball.dy / scale))

    vec = brickBits + [nx, ny, ndx, ndy]
    return np.array(vec, dtype=np.float32).reshape(-1, 1)

def get_nn_input_advanced():
    brickBits = [1 if brick.alive else 0 for brick in bricks]

    nx = ball.x / window.width
    ny = ball.y / window.height

    speed_scale = 300.0
    ndx = np.tanh(ball.dx / speed_scale)
    ndy = np.tanh(ball.dy / speed_scale)

    vec = brickBits + [nx, ny, float(ndx), float(ndy)]
    return np.array(vec, dtype=np.float32).reshape(-1, 1)

def ai_move(nn, input_fn, dt):
    nn_input = input_fn()
    scores = nn.forward(nn_input)
    action = int(np.argmax(scores))

    if action == 0:
        paddle.x -= 300 * dt
    elif action == 2:
        paddle.x += 300 * dt

    paddle.x = max(0, min(window.width - paddle.width, paddle.x))

def update(dt):
    global elapsedTime, state, lives, gameStarted, score, stage
    if state == "solobasic":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"
        if not gameStarted:
            if keys[key.UP]:
                ball.dx, ball.dy = 200, 200
                gameStarted = True
            else:
                if keys[key.LEFT]:
                    paddle.x -= 300 * dt
                if keys[key.RIGHT]:
                    paddle.x += 300 * dt
                paddle.x = max(0, min(window.width - paddle.width, paddle.x))
                ball.x = paddle.x+paddle.width//2-ball.width//2
                ball.y = 10
                return
        elapsedTime += dt
        if keys[key.LEFT]:
            paddle.x -= 300 * dt
        if keys[key.RIGHT]:
            paddle.x += 300 * dt
        paddle.x = max(0, min(window.width - paddle.width, paddle.x))
        ball.x += ball.dx * dt
        ball.y += ball.dy * dt
        for brick in bricks:
            if brick.checkCollision(ball):
                score +=1
                scoreLabel.text = f"Score: {score}"
                break
        if ball.x+ball.dx/60 <= 0 or ball.x > window.width-ball.width:
            ball.dx *= -1
        if ball.y > window.height - ball.height:
            ball.dy *= -1

        if (paddle.y <= ball.y <= paddle.y+paddle.height) and \
            (paddle.x-ball.width <= ball.x <= paddle.x+paddle.width+ball.width):
            ball.dy = abs(ball.dy)

        if ball.y < 0:
            lives -=1
            gameStarted = False
        
        if lives == 0:
            state = "gameover"
            lives = 5
            score = 0
            stage = 1

        if all(not brick.alive for brick in bricks):
            stage += 1
            stageLabel.text = f"Stage: {stage}"
            gameStarted = False
            ball.dx, ball.dy = 0, 0
            ball.x = paddle.x+paddle.width//2-ball.width//2
            ball.y = 1
            createBricks()
    
    elif state == "soloadvanced":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"
        if not gameStarted:
            if keys[key.UP]:
                ball.dx, ball.dy = 200, 200
                gameStarted = True
            else:
                if keys[key.LEFT]:
                    paddle.x -= 300 * dt
                if keys[key.RIGHT]:
                    paddle.x += 300 * dt
                paddle.x = max(0, min(window.width - paddle.width, paddle.x))
                ball.x = paddle.x+paddle.width//2-ball.width//2
                ball.y = 10
                return
        elapsedTime += dt
        if keys[key.LEFT]:
            paddle.x -= 300 * dt
        if keys[key.RIGHT]:
            paddle.x += 300 * dt
        paddle.x = max(0, min(window.width - paddle.width, paddle.x))
        ball.x += ball.dx * dt
        ball.y += ball.dy * dt
        for brick in bricks:
            if brick.checkCollision(ball):
                score +=1
                scoreLabel.text = f"Score: {score}"
                break
        if ball.x+ball.dx/60 <= 0 or ball.x > window.width-ball.width:
            ball.dx *= -1
        if ball.y > window.height - ball.height:
            ball.dy *= -1

        if (ball.y+ball.dy/60 <= paddle.y+paddle.height)and(paddle.x-ball.width<=ball.x<=paddle.x+paddle.width):
            ball.dy = abs(ball.dy)
            if (paddle.x-ball.width/2<=ball.x+ball.width/2<=paddle.x+paddle.width/4) or \
                (paddle.x+3*(paddle.width/4)<=ball.x+ball.width/2<=paddle.x+paddle.width+ball.width/2):
                ball.dx *= 1.2
            elif paddle.x+(paddle.width/2)<ball.x+ball.width/2<paddle.x+3*(paddle.width/4):
                ball.dx *= 0.8

        if ball.y < 0:
            lives -=1
            gameStarted = False
        
        if lives == 0:
            state = "gameover"
            lives = 5
            score = 0
            stage = 1

        if all(not brick.alive for brick in bricks):
            stage += 1
            stageLabel.text = f"Stage: {stage}"
            gameStarted = False
            ball.dx, ball.dy = 0, 0
            ball.x = paddle.x+paddle.width//2-ball.width//2
            ball.y = 1
            createBricks()
    
    elif state == "watchbasic":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"

        if not gameStarted:
            ball.dx, ball.dy = 200, 200
            gameStarted = True

        elapsedTime += dt

        ai_move(nn_basic, get_nn_input_basic, dt)

        ball.x += ball.dx * dt
        ball.y += ball.dy * dt

        for brick in bricks:
            if brick.checkCollision(ball):
                score += 1
                scoreLabel.text = f"Score: {score}"
                break

        if ball.x + ball.dx/60 <= 0 or ball.x > window.width - ball.width:
            ball.dx *= -1
        if ball.y > window.height - ball.height:
            ball.dy *= -1

        if (paddle.y <= ball.y <= paddle.y + paddle.height) and \
           (paddle.x - ball.width <= ball.x <= paddle.x + paddle.width + ball.width):
            ball.dy = abs(ball.dy)

        if ball.y < 0:
            lives -= 1
            gameStarted = False
            ball.x = paddle.x + paddle.width//2 - ball.width//2
            ball.y = paddle.y + paddle.height + 1
            ball.dx, ball.dy = 0, 0

        if lives == 0:
            state = "gameover"
            lives = 5
            score = 0
            stage = 1

        if all(not brick.alive for brick in bricks):
            stage += 1
            stageLabel.text = f"Stage: {stage}"
            gameStarted = False
            ball.dx, ball.dy = 0, 0
            ball.x = paddle.x+paddle.width//2-ball.width//2
            ball.y = 1
            createBricks()

    elif state == "watchadvanced":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"

        if not gameStarted:
            ball.dx, ball.dy = 200, 200
            gameStarted = True

        elapsedTime += dt

        ai_move(nn_advanced, get_nn_input_advanced, dt)

        ball.x += ball.dx * dt
        ball.y += ball.dy * dt

        for brick in bricks:
            if brick.checkCollision(ball):
                score += 1
                scoreLabel.text = f"Score: {score}"
                break

        if ball.x + ball.dx/60 <= 0 or ball.x > window.width - ball.width:
            ball.dx *= -1
        if ball.y > window.height - ball.height:
            ball.dy *= -1

        if (ball.y + ball.dy/60 <= paddle.y + paddle.height) and \
           (paddle.x - ball.width <= ball.x <= paddle.x + paddle.width):
            ball.dy = abs(ball.dy)
            if (paddle.x - ball.width/2 <= ball.x + ball.width/2 <= paddle.x + paddle.width/4) or \
               (paddle.x + 3*(paddle.width/4) <= ball.x + ball.width/2 <= paddle.x + paddle.width + ball.width/2):
                ball.dx *= 1.2
            elif paddle.x + (paddle.width/2) < ball.x + ball.width/2 < paddle.x + 3*(paddle.width/4):
                ball.dx *= 0.8

        if ball.y < 0:
            lives -= 1
            gameStarted = False
            ball.x = paddle.x + paddle.width//2 - ball.width//2
            ball.y = paddle.y + paddle.height + 1
            ball.dx, ball.dy = 0, 0

        if lives == 0:
            state = "gameover"
            lives = 5
            score = 0
            stage = 1

        if all(not brick.alive for brick in bricks):
            stage += 1
            stageLabel.text = f"Stage: {stage}"
            gameStarted = False
            ball.dx, ball.dy = 0, 0
            ball.x = paddle.x+paddle.width//2-ball.width//2
            ball.y = 1
            createBricks()

@window.event
def on_draw():
    window.clear()
    if state == "menu":
        titleLabel.draw()
        soloButton.draw()
        soloButtonText.draw()
        trainingButton.draw()
        trainingButtonText.draw()
        watchButton.draw()
        watchButtonText.draw()
        scoresButton.draw()
        scoresButtonText.draw()
    elif state == "solomodemenu":
        solobasicButton.draw()
        solobasicButtonText.draw()
        soloadvancedButton.draw()
        soloadvancedButtonText.draw()
    elif state == "watchmodemenu":
        watchbasicButton.draw()
        watchbasicButtonText.draw()
        watchadvancedButton.draw()
        watchadvancedButtonText.draw()
    elif state == "trainingmodemenu":
        trainingbasicButton.draw()
        trainingbasicButtonText.draw()
        trainingadvancedButton.draw()
        trainingadvancedButtonText.draw()
    elif state == "scoresmenu":
        pass
    elif state == "solobasic":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
        exitButton.draw()
        exitButtonText.draw()
    elif state == "soloadvanced":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
        exitButton.draw()
        exitButtonText.draw()
    elif state == "watchbasic":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
        exitButton.draw()
        exitButtonText.draw()
    elif state == "watchadvanced":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
        exitButton.draw()
        exitButtonText.draw()
    elif state == "gameover":
        gameoverLabel.draw()
        scoreLabel.text = f"Final Score: {score}"
        scoreLabel.draw()
        restartButton.draw()
        restartButtonText.draw()

@window.event
def on_mouse_press(x, y, button, modifiers):
    global state
    if button == mouse.LEFT:
        if state == "menu":
            if 220 <= x <= 420 and 250 <= y <= 300:
                state = "solomodemenu"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                state = "trainingmodemenu"
            elif 220 <= x <= 420 and 150 <= y <= 200:
                state = "watchmodemenu"
            elif 220 <= x <= 420 and 100 <= y <= 150:
                state = "scoresmenu"
        elif state == "solomodemenu":
            if 220 <= x <= 420 and 275 <= y <= 325:
                resetGame()
                state = "solobasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                resetGame()
                state = "soloadvanced"
        elif state == "trainingMenu":
            if 220 <= x <= 420 and 275 <= y <= 325:
                state = "trainingmodebasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                state = "trainingmodeadvanced"
        elif state == "watchmodemenu":
            if 220 <= x <= 420 and 275 <= y <= 325:
                resetGame()
                state = "watchbasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                resetGame()
                state = "watchadvanced"
        elif state == "solobasic":
            if 510 <= x <= 610 and 450 <= y <= 490:
                state = "menu"
        elif state == "soloadvanced":
            if 510 <= x <= 610 and 450 <= y <= 490:
                state = "menu"
        elif state == "watchbasic":
            if 510 <= x <= 610 and 450 <= y <= 490:
                state = "menu"
        elif state == "watchadvanced":
            if 510 <= x <= 610 and 450 <= y <= 490:
                state = "menu"
        elif state == "scoresScreen":
            if 10 <= x <= 110 and 10 <= y <= 50:
                state = "menu"
        elif state == "gameover":
            if 220 <= x <= 420 and 200 <= y <= 250:
                state = "menu"
        elif state == "gamewon":
            if 540 <= x <= 640 and 440 <= y <= 480:
                state = "menu"

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()