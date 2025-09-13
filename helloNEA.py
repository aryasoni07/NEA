import pyglet
from pyglet.window import key, mouse
from pyglet import shapes
import numpy as np
import os
import json
import atexit
import random
import math

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

window = pyglet.window.Window(640, 520, caption="Breakout")
keys = key.KeyStateHandler()
window.push_handlers(keys)
state = "menu" 
elapsedTime = 0.0
lives = 5
gameStarted = False
score = 0
stage = 1
batch = pyglet.graphics.Batch()
titleLabel = pyglet.text.Label("BREAKOUT", font_size=36, x=window.width//2, y=350, anchor_x='center')
soloButton = shapes.Rectangle(220, 250, 200, 50, color=(255, 100, 100))
soloButtonText = pyglet.text.Label("Solo Mode", x=window.width//2, y=275, anchor_x='center', anchor_y='center')
solobasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
solobasicButtonText = pyglet.text.Label("Basic Mode", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
soloadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
soloadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
watchButton = shapes.Rectangle(220, 150, 200, 50, color=(100, 255, 100))
watchButtonText = pyglet.text.Label("Watch Mode", x=window.width//2, y=175, anchor_x='center', anchor_y='center')
watchbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
watchbasicButtonText = pyglet.text.Label("Watch Basic", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
watchadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
watchadvancedButtonText = pyglet.text.Label("Watch Advanced", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
trainingButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 255, 255))
trainingButtonText = pyglet.text.Label("Training Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
trainingbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
trainingbasicButtonText = pyglet.text.Label("Basic Mode", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
trainingadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
trainingadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
scoresButton = shapes.Rectangle(220, 100, 200, 50, color=(100, 100, 255))
scoresButtonText = pyglet.text.Label("Scores", x=window.width//2, y=125, anchor_x='center', anchor_y='center')
exitButton = shapes.Rectangle(540, 480, 100, 40, color=(200, 50, 50))
exitButtonText = pyglet.text.Label("Exit", x=590, y=500, anchor_x='center', anchor_y='center')
gameoverLabel = pyglet.text.Label("GAME OVER", font_size=32, x=window.width//2, y=310, anchor_x='center')
gameoverscoreLabel = pyglet.text.Label("Score: 0", font_size=16, x=window.width//2, y=270, anchor_x='center')
restartButton = shapes.Rectangle(220, 200, 200, 50, color=(255, 100, 100))
restartButtonText = pyglet.text.Label("Return to Menu", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
timerLabel = pyglet.text.Label("Time elapsed: 0.00", x=20, y=493)
livesLabel = pyglet.text.Label("Lives: 3", x=200, y=493)
scoreLabel = pyglet.text.Label("Score: 0", x=310, y=493)
stageLabel = pyglet.text.Label("Stage: 1", x=430, y=493)
header = shapes.Rectangle(x=0, y=480, width=640, height=40, color=(100, 100, 100), batch=batch)
paddle = shapes.Rectangle(x=290, y=00, width=60, height=10, color=(255, 255, 255), batch=batch)
ball = shapes.Rectangle(x=320, y=100, width=10, height=10, color=(255, 0, 0), batch=batch)
ball.dx = 200
ball.dy = 200
bricks = []

scoresFile = "scores.json"
scoresData = {"Solo Mode Basic": 0,
            "Solo Mode Advanced": 0,
            "Training Mode Basic": 0,
            "Training Mode Advanced": 0,
            "Watch Mode Basic": 0,
            "Watch Mode Advanced": 0}
def loadScores():
    global scoresData
    if os.path.exists(scoresFile):
        with open(scoresFile, "r") as f:
            scoresData = json.load(f)
def saveScores():
    with open(scoresFile, "w") as f:
        json.dump(scoresData, f)
loadScores()

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
            numsIn = self.layerSizes[i]
            numsOut = self.layerSizes[i+1]
            HeInit = np.sqrt(2.0 / numsIn)
            W = np.random.randn(numsOut, numsIn) * HeInit
            b = np.zeros((numsOut, 1))
            self.weights.append(W)
            self.biases.append(b)

    def forprop(self, x):
        a = x
        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ a + self.biases[i]
            a = np.maximum(z, 0.0)
        z = self.weights[-1] @ a + self.biases[-1]
        return z
    
    def backprop(self, inp, exp):
        a = inp
        activations = [inp]
        preact = []
        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ a + self.biases[i]
            preact.append(z)
            a = np.maximum(z, 0.0)
            activations.append(a)
        z = self.weights[-1] @ a + self.biases[-1]
        preact.append(z)
        output = z
        activations.append(output)
        
        if isinstance(exp, (int, float)):
            exp_vec = np.zeros((3, 1))
            exp_vec[int(exp)] = 1.0
        else:
            exp_vec = exp.reshape(-1, 1) if exp.shape != (3, 1) else exp
        
        delta = output - exp_vec
        deltaw = []
        deltab = []
        deltaw.append(delta @ activations[-2].T)
        deltab.append(delta)
        
        for i in range(len(self.weights) - 2, -1, -1):
            delta = self.weights[i + 1].T @ delta
            delta = delta * (preact[i] > 0).astype(np.float32)
            deltaw.insert(0, delta @ activations[i].T)
            deltab.insert(0, delta)
        return deltaw, deltab


    def save(self):
        with open(self.saveFile, "w") as f:
            json.dump({"weights": [w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.biases]}, f)

    def load(self):
        with open(self.saveFile, "r") as f:
            data = json.load(f)
            self.weights = [np.array(w) for w in data["weights"]]
            self.biases = [np.array(b) for b in data["biases"]]

nn_basic = NeuralNetwork([60, 32, 16, 3], saveFile="nn_basic.json")
nn_advanced = NeuralNetwork([60, 50, 30, 3], saveFile="nn_advanced.json")
atexit.register(nn_basic.save)
atexit.register(nn_advanced.save)
lr = 0.001
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
            y = 480-topOffset-row*brickHeight
            bricks.append(Brick(x, y, brickWidth, brickHeight))

def get_training_input_basic():
    mode = int(np.random.choice([1, 2, 3, 4, 5]))
    if mode == 1:
        vec = [1]*55
    elif mode == 2:
        vec = [0]*55
    elif mode == 3:
        vec = [1 if random.random() < 0.8 else 0 for _ in range(55)]
    else:
        vec = [1 if random.random() < 0.2 else 0 for _ in range(55)]

    x = random.uniform(0, 630)
    y = random.uniform(0, 310)
    dx = random.choice([-200.0, 200.0])
    dy = random.choice([-200.0, 200.0])
    paddlex = random.uniform(0, 580)
    nx = x/630.0
    ny = y/310.0
    ndx = dx/200.0
    ndy = dy/200.0
    pa = paddlex / 580.0
    vec = vec + [nx, ny, ndx, ndy, pa]

    if dx == 0:
        fx = x
    elif dy/dx < 0:
        fx = x - y*(dx/dy)
    else:
        fx = x + (dx/dy)*(620-y)
    while fx<0 or fx>630:
        if fx < 0:
            fx=abs(fx)
        elif fx > 630:
            fx = 630 - fx
    if fx < paddlex:
        ans = 0
    elif fx > paddlex:
        ans = 2
    else:
        ans = 1
    return np.array(vec, dtype=np.float32).reshape(-1, 1), ans

def get_training_input_advanced():
    mode = int(np.random.choice([1, 2, 3, 4, 5]))
    if mode == 1:
        vec = [1]*55
    elif mode == 2:
        vec = [0]*55
    elif mode == 3:
        vec = [1 if random.random() < 0.8 else 0 for _ in range(55)]
    else:
        vec = [1 if random.random() < 0.2 else 0 for _ in range(55)]

    x = random.uniform(0, 630)
    y = random.uniform(0, 310)
    dy = random.choice([-200.0, 200.0])
    mean = -200 if random.random() < 0.5 else 200
    dx = random.gauss(mu=mean, sigma=100)
    paddlex = random.uniform(0, 580)
    nx = x / 630.0
    ny = y / 310.0
    ndx = dx / 200.0
    ndy = dy / 200.0
    pa = paddlex / 580.0
    vec = vec + [nx, ny, ndx, ndy, pa]

    if dx == 0:
        fx = x
    elif dy/dx < 0:
        fx = x - y*(dx/dy)
    else:
        fx = x + (dx/dy)*(620-y)
    while fx<0 or fx>630:
        if fx < 0:
            fx=abs(fx)
        elif fx > 630:
            fx = 630 - fx
    if fx < paddlex:
        ans = 0
    elif fx > paddlex:
        ans = 2
    else:
        ans = 1
    return np.array(vec, dtype=np.float32).reshape(-1, 1), ans

def get_nn_input():
    vec = [1 if Brick.alive else 0 for Brick in bricks]
    nx = ball.x/630.0
    ny = ball.y/310.0
    ndx = ball.dx/200.0
    ndy = ball.dy/200.0
    pa = paddle.x/580.0
    vec = vec + [nx, ny, ndx, ndy, pa]
    return np.array(vec, dtype=np.float32).reshape(-1, 1)

def ai_move(nn, inp, dt):
    scores = nn.forprop(inp)
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
        if ball.x+ball.dx/60 <= 0:
            ball.dx = abs(ball.dx)
        elif ball.x > window.width-ball.width:
            ball.dx = -1*abs(ball.dx)
        if ball.y > 480 - ball.height:
            ball.dy = -1*abs(ball.dy)

        if (paddle.y <= ball.y <= paddle.y+paddle.height) and \
            (paddle.x-ball.width <= ball.x <= paddle.x+paddle.width+ball.width):
            ball.dy = abs(ball.dy)

        if ball.y < 0:
            lives -=1
            gameStarted = False
        
        if lives == 0:
            if score>scoresData["Solo Mode Basic"]:
                scoresData["Solo Mode Basic"] = score
                saveScores()
            gameoverscoreLabel.text = f"Score: {score}"
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
        if ball.x+ball.dx/60 <= 0:
            ball.dx = abs(ball.dx)
        elif ball.x > window.width-ball.width:
            ball.dx = -1*abs(ball.dx)
        if ball.y > 480 - ball.height:
            ball.dy = -1*abs(ball.dy)

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
            if score>scoresData["Solo Mode Advanced"]:
                scoresData["Solo Mode Advanced"] = score
                saveScores()
            gameoverscoreLabel.text = f"Score: {score}"
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
    
    elif state == "trainingbasic":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"

        if not gameStarted:
            ball.dx, ball.dy = 200, 200
            gameStarted = True

        elapsedTime += dt
        inp, ans = get_training_input_basic()
        ai_move(nn_basic, inp, dt)

        dW_list, db_list = nn_basic.backprop(inp, ans)

        for i in range(len(nn_basic.weights)):
            nn_basic.weights[i] -= lr * dW_list[i]
            nn_basic.biases[i]  -= lr * db_list[i]

    elif state == "trainingadvanced":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"

        if not gameStarted:
            ball.dx, ball.dy = 200, 200
            gameStarted = True

        elapsedTime += dt
        inp, ans = get_training_input_advanced()
        ai_move(nn_advanced, inp, dt)

        dW_list, db_list = nn_advanced.backprop(inp, ans)

        for i in range(len(nn_advanced.weights)):
            nn_advanced.weights[i] -= lr * dW_list[i]
            nn_advanced.biases[i]  -= lr * db_list[i]

    elif state == "watchbasic":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"

        if not gameStarted:
            ball.dx, ball.dy = 200, 200
            gameStarted = True

        elapsedTime += dt
        inp = get_nn_input()
        ai_move(nn_basic, inp, dt)

        ball.x += ball.dx * dt
        ball.y += ball.dy * dt

        for brick in bricks:
            if brick.checkCollision(ball):
                score += 1
                scoreLabel.text = f"Score: {score}"
                break
        if ball.x+ball.dx/60 <= 0:
            ball.dx = abs(ball.dx)
        elif ball.x > window.width-ball.width:
            ball.dx = -1*abs(ball.dx)
        if ball.y > 480 - ball.height:
            ball.dy = -1*abs(ball.dy)

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
            if score>scoresData["Watch Mode Basic"]:
                scoresData["Watch Mode Basic"] = score
                saveScores()
            gameoverscoreLabel.text = f"Score: {score}"
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
        inp = get_nn_input()
        ai_move(nn_advanced, inp, dt)

        ball.x += ball.dx * dt
        ball.y += ball.dy * dt

        for brick in bricks:
            if brick.checkCollision(ball):
                score += 1
                scoreLabel.text = f"Score: {score}"
                break

        if ball.x + ball.dx/60 <= 0 or ball.x > window.width - ball.width:
            ball.dx *= -1
        if ball.y > 480 - ball.height:
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
            if score>scoresData["Watch Mode Advanced"]:
                scoresData["Watch Mode Advanced"] = score
                saveScores()
            gameoverscoreLabel.text = f"Score: {score}"
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
        y = 400
        for gamestate, highscore in scoresData.items():
            pyglet.text.Label(f"{gamestate}: {highscore}",x=window.width//2,y=y,anchor_x='center').draw()
            y-=60
        exitButton.draw()
        exitButtonText.draw()
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
    elif state == "trainingbasic":
        paddle.draw()
        timerLabel.draw()
        exitButton.draw()
        exitButtonText.draw()
    elif state == "trainingadvanced":
        paddle.draw()
        timerLabel.draw()
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
        gameoverscoreLabel.draw()
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
        elif state == "trainingmodemenu":
            if 220 <= x <= 420 and 275 <= y <= 325:
                resetGame()
                state = "trainingbasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                resetGame()
                state = "trainingadvanced"
        elif state == "watchmodemenu":
            if 220 <= x <= 420 and 275 <= y <= 325:
                resetGame()
                state = "watchbasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                resetGame()
                state = "watchadvanced"
        elif state == "solobasic":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "soloadvanced":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "trainingbasic":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "trainingadvanced":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "watchbasic":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "watchadvanced":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "scoresmenu":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"
        elif state == "gameover":
            if 220 <= x <= 420 and 200 <= y <= 250:
                state = "menu"
        elif state == "gamewon":
            if 540 <= x <= 640 and 480 <= y <= 520:
                state = "menu"

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()