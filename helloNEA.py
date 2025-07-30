import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

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
maxStage = 3

titleLabel = pyglet.text.Label("Breakout", font_size=36, x=window.width//2, y=350, anchor_x='center')
soloButton = shapes.Rectangle(220, 250, 200, 50, color=(100, 100, 255))
soloButtonText = pyglet.text.Label("Solo Mode", x=window.width//2, y=275, anchor_x='center', anchor_y='center')
solobasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
solobasicButtonText = pyglet.text.Label("Basic Mode", x=window.width//2, y=300, anchor_x='center', anchor_y='center')
soloadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
soloadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width//2, y=225, anchor_x='center', anchor_y='center')

timerLabel = pyglet.text.Label("Time elapsed: 0.00", x=10, y=450)
livesLabel = pyglet.text.Label("Lives: 3", x=210, y=450)
scoreLabel = pyglet.text.Label("Score: 0", x=370, y=450)
stageLabel = pyglet.text.Label("Stage: 1", x=510, y=450)
batch = pyglet.graphics.Batch()
paddle = shapes.Rectangle(x=270, y=00, width=60, height=10, color=(255, 255, 255), batch=batch)
ball = shapes.Rectangle(x=320, y=100, width=10, height=10, color=(255, 0, 0), batch=batch)
ball.dx = 200
ball.dy = 200
bricks = []
gamewonLabel = pyglet.text.Label("Well Done, You have won Breakout!", font_size=28, x=window.width//2, y=300, anchor_x='center')
gamewonTimeLabel = pyglet.text.Label("", font_size=18, x=window.width//2, y=260, anchor_x='center')
gameoverLabel = pyglet.text.Label("GAME OVER", font_size=32, x=window.width//2, y=300, anchor_x='center')
restartButton = shapes.Rectangle(220, 200, 200, 50, color=(255, 100, 100))
restartButtonText = pyglet.text.Label("Return to Menu", x=window.width//2, y=225, anchor_x='center', anchor_y='center')

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

        if (paddle.y <= ball.y <= paddle.y+paddle.height)and(paddle.x-ball.width <= ball.x <= paddle.x+paddle.width-ball.width):
            ball.dy *= -1

        if ball.y < 0:
            lives -=1
            gameStarted = False
        
        if lives == 0:
            state = "gameover"
            lives = 5
            score = 0
            stage = 1

        if all(not brick.alive for brick in bricks):
            if stage < maxStage:
                stage += 1
                stageLabel.text = f"Stage: {stage}"
                gameStarted = False
                ball.dx, ball.dy = 0, 0
                ball.x = paddle.x+paddle.width//2-ball.width//2
                ball.y = 1
                createBricks()
            else:
                state = "gamewon"
                gamewonTimeLabel.text = f"Time elapsed: {elapsedTime:.2f}"
    
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

        if ball.y+ball.dy/60 <= paddle.y+paddle.height:
            if (paddle.x<=ball.x+ball.width/2<=paddle.x+paddle.width/4)or(paddle.x+3*(paddle.width/4)<=ball.x+ball.width/2<=paddle.x+paddle.width):
                ball.dy *= -1
                ball.dx *= 1.25
            elif paddle.x+(paddle.width/2)<=ball.x+ball.width/2<=paddle.x+3*(paddle.width/4):
                ball.dy *= -1
                ball.dx *= 0.75

        if ball.y < 0:
            lives -=1
            gameStarted = False
        
        if lives == 0:
            state = "gameover"
            lives = 5
            score = 0
            stage = 1

        if all(not brick.alive for brick in bricks):
            if stage < maxStage:
                stage += 1
                stageLabel.text = f"Stage: {stage}"
                gameStarted = False
                ball.dx, ball.dy = 0, 0
                ball.x = paddle.x+paddle.width//2-ball.width//2
                ball.y = 1
                createBricks()
            else:
                state = "gamewon"
                gamewonTimeLabel.text = f"Time elapsed: {elapsedTime:.2f}"

@window.event
def on_draw():
    window.clear()
    if state == "menu":
        titleLabel.draw()
        soloButton.draw()
        soloButtonText.draw()
    elif state == "solomodemenu":
        solobasicButton.draw()
        solobasicButtonText.draw()
        soloadvancedButton.draw()
        soloadvancedButtonText.draw()
    elif state == "solobasic":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
    elif state == "soloadvanced":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
    elif state == "gameover":
        gameoverLabel.draw()
        restartButton.draw()
        restartButtonText.draw()
    elif state == "gamewon":
        gamewonLabel.draw()
        gamewonTimeLabel.draw()
        restartButton.draw()
        restartButtonText.draw()

@window.event
def on_mouse_press(x, y, button, modifiers):
    global state
    if button == mouse.LEFT:
        if state == "menu":
            if 220 <= x <= 420 and 250 <= y <= 300:
                state = "solomodemenu"
        elif state == "solomodemenu":
            if 220 <= x <= 420 and 275 <= y <= 325:
                resetGame()
                state = "solobasic"
            elif 220 <= x <= 420 and 200 <= y <= 250:
                resetGame()
                state = "soloadvanced"
        elif state == "gameover":
            if 220 <= x <= 420 and 200 <= y <= 250:
                state = "menu"
        elif state == "gamewon":
            if 220 <= x <= 420 and 200 <= y <= 250:
                state = "menu"

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()