import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

window = pyglet.window.Window(640, 480, caption="Breakout")
keys = key.KeyStateHandler()
window.push_handlers(keys)
state = "menu" 
elapsedTime = 0.0
lives = 3
gameStarted = False
titleLabel = pyglet.text.Label("Breakout", font_size=36, x=window.width//2, y=400, anchor_x='center')
solomodeButton = shapes.Rectangle(220, 250, 200, 50, color=(100, 100, 255))
solomodeButtonText = pyglet.text.Label("Solo Mode", x=window.width//2, y=275, anchor_x='center', anchor_y='center')
timerLabel = pyglet.text.Label("Time elapsed: 0.00", x=10, y=450)
livesLabel = pyglet.text.Label("Lives: 3", x=210, y=450)
batch = pyglet.graphics.Batch()
paddle = shapes.Rectangle(x=270, y=0, width=60, height=10, color=(255, 255, 255), batch=batch)
ball = shapes.Rectangle(x=320, y=100, width=10, height=10, color=(255, 0, 0), batch=batch)
ball.dx = 200
ball.dy = 200
bricks = []

gameoverLabel = pyglet.text.Label("GAME OVER", font_size=32, x=window.width//2, y=300, anchor_x='center')
restartButton = shapes.Rectangle(220, 200, 200, 50, color=(255, 100, 100))
restartButtonText = pyglet.text.Label("Return to Menu", x=window.width//2, y=225, anchor_x='center', anchor_y='center')

def resetGame():
    global ball, paddle, elapsedTime, state, lives
    ball.x = paddle.x + paddle.width // 2 - ball.width // 2
    ball.y = 10
    ball.dx, ball.dy = 0, 0
    paddle.x = 270
    elapsedTime = 0
    state = "solo"

def update(dt):
    global elapsedTime, state, lives, gameStarted
    if state == "solo":
        timerLabel.text = f"Time elapsed: {elapsedTime:.2f}"
        livesLabel.text = f"Lives: {lives}"
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
                ball.x = paddle.x + paddle.width // 2 - ball.width // 2
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

        if ball.x < ball.width or ball.x > window.width - ball.width:
            ball.dx *= -1
        if ball.y > window.height - ball.height:
            ball.dy *= -1

        if (paddle.y <= ball.y <= paddle.y + paddle.height)and(paddle.x <= ball.x <= paddle.x + paddle.width):
            ball.dy *= -1

        if ball.y < 0:
            lives -=1
            gameStarted = False
        
        if lives == 0:
            state = "gameover"
            lives = 3

        

@window.event
def on_draw():
    window.clear()

    if state == "menu":
        titleLabel.draw()
        solomodeButton.draw()
        solomodeButtonText.draw()

    elif state == "solo":
        batch.draw()
        timerLabel.draw()
        livesLabel.draw()

    elif state == "gameover":
        gameoverLabel.draw()
        restartButton.draw()
        restartButtonText.draw()

@window.event
def on_mouse_press(x, y, button, modifiers):
    global state

    if button == mouse.LEFT:
        if state == "menu":
            if 220 <= x <= 420 and 250 <= y <= 300:
                resetGame()

        elif state == "gameover":
            if 220 <= x <= 420 and 200 <= y <= 250:
                state = "menu"

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()