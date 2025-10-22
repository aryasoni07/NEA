import pyglet
from pyglet.window import key, mouse
from pyglet import shapes
import numpy as np
import os
import json
import atexit
import random
from brick import Brick
from neuralnetwork import NeuralNetwork

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
watchButton = shapes.Rectangle(220, 150, 200, 50, color=(50, 200, 100))
watchButtonText = pyglet.text.Label("Watch Mode", x=window.width//2, y=175, anchor_x='center', anchor_y='center')
watchbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
watchbasicButtonText = pyglet.text.Label("Watch Basic", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
watchadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
watchadvancedButtonText = pyglet.text.Label("Watch Advanced", x=window.width//2, y=225, anchor_x='center', anchor_y='center')
trainingButton = shapes.Rectangle(220, 200, 200, 50, color=(0, 150, 200))
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

accuracyLabel = pyglet.text.Label("Accuracy: 0%", x=20, y=470)
trainingbatch = pyglet.graphics.Batch()
trainingelements = []
accuracyBasic = []
accuracyAdvanced = []
groupBasic = []
groupAdvanced = []

header = shapes.Rectangle(x=0, y=480, width=640, height=40, color=(100, 100, 100), batch=batch)
paddle = shapes.Rectangle(x=290, y=00, width=60, height=10, color=(255, 255, 255), batch=batch)
ball = shapes.Rectangle(x=320, y=100, width=10, height=10, color=(255, 0, 0), batch=batch)
ball.dx = 200
ball.dy = 200
bricks = []
respawnTime = 1.0
timer = 0.5

scoresFile = "scores.json"
scoresData = {"Solo Mode Basic": 0,
            "Solo Mode Advanced": 0,
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

nn_basic = NeuralNetwork([60, 70, 50, 3], saveFile="nn_basic.json")
nn_advanced = NeuralNetwork([60, 80, 60, 3], saveFile="nn_advanced.json")
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
            bricks.append(Brick(x, y, brickWidth, brickHeight, batch))

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
    if fx < (paddlex+30):
        ans = 0
    elif fx > (paddlex+30):
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
    if fx < paddlex+30:
        ans = 0
    elif fx > paddlex+30:
        ans = 2
    else:
        ans = 1
    return np.array(vec, dtype=np.float32).reshape(-1, 1), ans

def update_training_visual(vec):
    global trainingelements
    for s in trainingelements:
        try:
            s.delete()
        except:
            pass
    trainingelements = []

    bricks_mask = vec[:55]
    nx, ny, ndx, ndy, pa = vec[55:]

    x = float(nx) * 630.0
    y = float(ny) * 310.0
    paddlex = float(pa) * 580.0

    rows, cols = 5, 11
    brick_w, brick_h = 60, 20
    top_offset = 60
    startX = (window.width - cols * brick_w) // 2

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if bricks_mask[idx] >= 0.5:
                rx = startX + col * brick_w
                ry = 480 - top_offset - row * brick_h
                r = shapes.Rectangle(rx, ry, brick_w, brick_h, color=(120, 200, 255), batch=trainingbatch)
                trainingelements.append(r)

    pr = shapes.Rectangle(int(paddlex), 0, 60, 10, color=(255, 255, 255), batch=trainingbatch)
    trainingelements.append(pr)

    br = shapes.Rectangle(int(x), int(y), 10, 10, color=(255, 0, 0), batch=trainingbatch)
    trainingelements.append(br)

def update_accuracy(history, isCorrect):
    history.append(1 if isCorrect else 0)
    if len(history) > 1000:
        history.pop(0)
    percent = 100.0 * sum(history) / len(history)
    return f"Accuracy: {percent:.1f}%"

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
    global elapsedTime, state, lives, gameStarted, score, stage, respawnTime
    if state == "solobasic":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"
        
        if ball.y <= 0:
            gameStarted = False
            if respawnTime > 0.0:
                respawnTime -= dt
                progress = 1 - respawnTime/1.0
                ball.color = (255, int(255 * progress), int(255 * progress))
                return
            else:
                lives -= 1
                if lives == 0:
                    if score>scoresData["Solo Mode Basic"]:
                        scoresData["Solo Mode Basic"] = score
                        saveScores()
                    gameoverscoreLabel.text = f"Score: {score}"
                    state = "gameover"
                    lives = 5
                    score = 0
                    stage = 1
                    respawnTime = 1.0
                    ball.color = (255, 0, 0)
                    return
                respawnTime = 1.0
                ball.color = (255, 0, 0)
        
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
        
        if ball.y <= 0:
            gameStarted = False
            if respawnTime > 0.0:
                respawnTime -= dt
                progress = 1 - respawnTime/1.0
                ball.color = (255, int(255 * progress), int(255 * progress))
                return
            else:
                lives -= 1
                if lives == 0:
                    if score>scoresData["Solo Mode Advanced"]:
                        scoresData["Solo Mode Advanced"] = score
                        saveScores()
                    gameoverscoreLabel.text = f"Score: {score}"
                    state = "gameover"
                    lives = 5
                    score = 0
                    stage = 1
                    respawnTime = 1.0
                    ball.color = (255, 0, 0)
                    return
                respawnTime = 1.0
                ball.color = (255, 0, 0)
        
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

        if all(not brick.alive for brick in bricks):
            stage += 1
            stageLabel.text = f"Stage: {stage}"
            gameStarted = False
            ball.dx, ball.dy = 0, 0
            ball.x = paddle.x+paddle.width//2-ball.width//2
            ball.y = 1
            createBricks()
    
    elif state == "trainingbasic":
        global accuracyBasic
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        elapsedTime += dt

        inp, ans = get_training_input_basic()
        scores_now = nn_basic.forprop(inp)
        pred = int(np.argmax(scores_now))
        accuracyLabel.text = update_accuracy(accuracyBasic, pred == ans)
        update_training_visual(inp.flatten())
        ai_move(nn_basic, inp, dt)

        dW_list, db_list = nn_basic.backprop(inp, ans)
        groupBasic.append([dW_list, db_list])

        if len(groupBasic) == 10:
            dW_lists = [dw for (dw, db) in groupBasic]
            db_lists = [db for (dw, db) in groupBasic]
            avg_dW = []
            avg_db = []
            for layer in range(len(nn_basic.weights)):
                dW_layers = [dW[layer] for dW in dW_lists]
                db_layers = [db[layer] for db in db_lists]
                avg_dW.append(np.mean(np.stack(dW_layers, axis=0), axis=0))
                avg_db.append(np.mean(np.stack(db_layers, axis=0), axis=0))
            for i in range(len(nn_basic.weights)):
                nn_basic.weights[i] -= lr * avg_dW[i]
                nn_basic.biases[i]  -= lr * avg_db[i]
            groupBasic.clear()

    elif state == "trainingadvanced":
        global accuracyAdvanced
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        elapsedTime += dt

        inp, ans = get_training_input_advanced()
        scores_now = nn_advanced.forprop(inp)
        pred = int(np.argmax(scores_now))
        accuracyLabel.text = update_accuracy(accuracyAdvanced, pred == ans)
        update_training_visual(inp.flatten())
        ai_move(nn_advanced, inp, dt)

        dW_list, db_list = nn_advanced.backprop(inp, ans)
        groupAdvanced.append([dW_list, db_list])

        if len(groupAdvanced) == 10:
            dW_lists = [dw for (dw, db) in groupAdvanced]
            db_lists = [db for (dw, db) in groupAdvanced]
            avg_dW = []
            avg_db = []
            for layer in range(len(nn_advanced.weights)):
                dW_layers = [dW[layer] for dW in dW_lists]
                db_layers = [db[layer] for db in db_lists]
                avg_dW.append(np.mean(np.stack(dW_layers, axis=0), axis=0))
                avg_db.append(np.mean(np.stack(db_layers, axis=0), axis=0))
            for i in range(len(nn_advanced.weights)):
                nn_advanced.weights[i] -= lr * avg_dW[i]
                nn_advanced.biases[i]  -= lr * avg_db[i]
            groupAdvanced.clear()

    elif state == "watchbasic":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
        scoreLabel.text = f"Score: {score}"
        stageLabel.text = f"Stage: {stage}"

        if ball.y <= 0:
            gameStarted = False
            if respawnTime > 0.0:
                respawnTime -= dt
                progress = 1 - respawnTime/1.0
                ball.color = (255, int(255 * progress), int(255 * progress))
                return
            else:
                ball.x = paddle.x+(paddle.width/2)-(ball.width/2)
                ball.y = paddle.height+1
                lives -= 1
                if lives == 0:
                    if score>scoresData["Watch Mode Basic"]:
                        scoresData["Watch Mode Basic"] = score
                        saveScores()
                    gameoverscoreLabel.text = f"Score: {score}"
                    state = "gameover"
                    lives = 5
                    score = 0
                    stage = 1
                    respawnTime = 1.0
                    ball.color = (255, 0, 0)
                    return
                respawnTime = 1.0
                ball.color = (255, 0, 0)

        if not gameStarted:
            ball.dx, ball.dy = 200, 200
            gameStarted = True

        if ball.y<320:
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

        if ball.y <= 0:
            gameStarted = False
            if respawnTime > 0.0:
                respawnTime -= dt
                progress = 1 - respawnTime/1.0
                ball.color = (255, int(255 * progress), int(255 * progress))
                return
            else:
                ball.x = paddle.x+(paddle.width/2)-(ball.width/2)
                ball.y = paddle.height+1
                lives -= 1
                if lives == 0:
                    if score>scoresData["Watch Mode Advanced"]:
                        scoresData["Watch Mode Advanced"] = score
                        saveScores()
                    gameoverscoreLabel.text = f"Score: {score}"
                    state = "gameover"
                    lives = 5
                    score = 0
                    stage = 1
                    respawnTime = 1.0
                    ball.color = (255, 0, 0)
                    return
                respawnTime = 1.0
                ball.color = (255, 0, 0)

        if not gameStarted:
            ball.dx = random.choice([-200, 200])
            ball.dy = 200
            gameStarted = True

        if ball.y<320:
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
        pyglet.text.Label("HIGH SCORES", font_size=28, x=window.width//2, y=460, anchor_x="center", color=(200, 255, 255, 255)).draw()
        y = 360
        for gamestate, highscore in scoresData.items():
            pyglet.text.Label(f"{gamestate.upper().ljust(28, '.')} {highscore}", font_name="Courier New", font_size=16, x=125, y=y).draw()
            y -= 70
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
        timerLabel.draw()
        accuracyLabel.draw()
        trainingbatch.draw()
        exitButton.draw()
        exitButtonText.draw()
    elif state == "trainingadvanced":
        timerLabel.draw()
        accuracyLabel.draw()
        trainingbatch.draw()
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
                paddle.x = 580 * random.random()
                state = "watchbasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                resetGame()
                paddle.x = 580 * random.random()
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