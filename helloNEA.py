#Main file

import random
import atexit
import pyglet
from pyglet.window import key, mouse
from pyglet import shapes
from neuralnetwork import NeuralNetwork
from breakoutgame import BreakoutGame, GameState

#Defining window dimensions
window = pyglet.window.Window(640, 520, caption="Breakout")
keys = key.KeyStateHandler()
window.push_handlers(keys)


#Making labels and buttons for buttons in the several menus
#A button is a rectange with a label over it
titleLabel = pyglet.text.Label("BREAKOUT", font_size=36, x=window.width // 2, y=350, anchor_x='center')

soloButton = shapes.Rectangle(220, 250, 200, 50, color=(255, 100, 100))
soloButtonText = pyglet.text.Label("Solo Mode", x=window.width // 2, y=275, anchor_x='center', anchor_y='center')

trainingButton = shapes.Rectangle(220, 200, 200, 50, color=(0, 150, 200))
trainingButtonText = pyglet.text.Label("Training Mode", x=window.width // 2, y=225, anchor_x='center', anchor_y='center')

watchButton = shapes.Rectangle(220, 150, 200, 50, color=(50, 200, 100))
watchButtonText = pyglet.text.Label("Watch Mode", x=window.width // 2, y=175, anchor_x='center', anchor_y='center')

scoresButton = shapes.Rectangle(220, 100, 200, 50, color=(100, 100, 255))
scoresButtonText = pyglet.text.Label("High Scores", x=window.width // 2, y=125, anchor_x='center', anchor_y='center')

solobasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
solobasicButtonText = pyglet.text.Label("Basic Mode", x=window.width // 2, y=300, anchor_x='center', anchor_y='center')
soloadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
soloadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width // 2, y=225, anchor_x='center', anchor_y='center')

trainingbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
trainingbasicButtonText = pyglet.text.Label("Basic Mode", x=window.width // 2, y=300, anchor_x='center', anchor_y='center')
trainingadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
trainingadvancedButtonText = pyglet.text.Label("Advanced Mode", x=window.width // 2, y=225, anchor_x='center', anchor_y='center')

watchbasicButton = shapes.Rectangle(220, 275, 200, 50, color=(100, 100, 255))
watchbasicButtonText = pyglet.text.Label("Watch Basic", x=window.width // 2, y=300, anchor_x='center', anchor_y='center')
watchadvancedButton = shapes.Rectangle(220, 200, 200, 50, color=(100, 100, 255))
watchadvancedButtonText = pyglet.text.Label("Watch Advanced", x=window.width // 2, y=225, anchor_x='center', anchor_y='center')

#Exit button is available in game modes, in order to exit the game
exitButton = shapes.Rectangle(540, 480, 100, 40, color=(200, 50, 50))
exitButtonText = pyglet.text.Label("Exit", x=590, y=500, anchor_x='center', anchor_y='center')

topBanner = shapes.Rectangle(0, 480, 540, 40, color=(100, 100, 100))

gameoverLabel = pyglet.text.Label("GAME OVER", font_size=32, x=window.width // 2, y=310, anchor_x='center')
gameoverscoreLabel = pyglet.text.Label("Score: 0", font_size=16, x=window.width // 2, y=270, anchor_x='center')
restartButton = shapes.Rectangle(220, 200, 200, 50, color=(255, 100, 100))
restartButtonText = pyglet.text.Label("Return to Menu", x=window.width // 2, y=225, anchor_x='center', anchor_y='center')

#Information displayed in game modes. Present in all modes except training mode
#With the exception of timer, which is in all game modes, and accuracy is only in training mode
timerLabel = pyglet.text.Label("Time elapsed: 0.00", x=20, y=493)
livesLabel = pyglet.text.Label("Lives: 3", x=200, y=493)
scoreLabel = pyglet.text.Label("Score: 0", x=310, y=493)
stageLabel = pyglet.text.Label("Stage: 1", x=430, y=493)
accuracyLabel = pyglet.text.Label("Accuracy: 0%", x=20, y=470)

#Defining the sizes of the neural networks
nn_basic = NeuralNetwork([5, 2000, 1000, 3], saveFile="nn_basic.json")
nn_advanced = NeuralNetwork([5, 5000, 2000, 3], saveFile="nn_advanced.json")
atexit.register(nn_basic.save)
atexit.register(nn_advanced.save)

#Calling the game
game = BreakoutGame(
    window=window,
    keys=keys,
    timerLabel=timerLabel,
    livesLabel=livesLabel,
    scoreLabel=scoreLabel,
    stageLabel=stageLabel,
    accuracyLabel=accuracyLabel,
    gameoverScoreLabel=gameoverscoreLabel,
    nn_basic=nn_basic,
    nn_advanced=nn_advanced,
    scoresFile="scores.json",
    learningRate=0.001)

#Performing the required checks every frame, depending on the game mode
def update(dt):
    game.update(dt)

#Choosing which objects to draw, depending on the game state
@window.event
def on_draw():
    window.clear()
    if game.gameState == GameState.MENU:
        titleLabel.draw()
        soloButton.draw()
        soloButtonText.draw()
        trainingButton.draw()
        trainingButtonText.draw()
        watchButton.draw()
        watchButtonText.draw()
        scoresButton.draw()
        scoresButtonText.draw()
    elif game.gameState == GameState.SOLO_MODE_MENU:
        solobasicButton.draw()
        solobasicButtonText.draw()
        soloadvancedButton.draw()
        soloadvancedButtonText.draw()
    elif game.gameState == GameState.TRAINING_MODE_MENU:
        trainingbasicButton.draw()
        trainingbasicButtonText.draw()
        trainingadvancedButton.draw()
        trainingadvancedButtonText.draw()
    elif game.gameState == GameState.WATCH_MODE_MENU:
        watchbasicButton.draw()
        watchbasicButtonText.draw()
        watchadvancedButton.draw()
        watchadvancedButtonText.draw()

#Displaying the high score labels in regular spaces
    elif game.gameState == GameState.SCORES_MENU:
        pyglet.text.Label("HIGH SCORES", font_size=28, x=window.width // 2, y=460, anchor_x="center", color=(200, 255, 255, 255)).draw()
        y = 360
        for gamestate, highscore in game.scoresData.items():
            pyglet.text.Label(f"{gamestate.upper().ljust(28, '.')}{highscore}", font_name="Courier New", font_size=16, x=125, y=y).draw()
            y -= 70
        exitButton.draw()
        exitButtonText.draw()

    elif game.gameState in (GameState.SOLO_BASIC, GameState.SOLO_ADVANCED):
        game.batch.draw()
        topBanner.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
        exitButton.draw()
        exitButtonText.draw()

    elif game.gameState in (GameState.TRAINING_BASIC, GameState.TRAINING_ADVANCED):
        timerLabel.draw()
        accuracyLabel.draw()
        exitButton.draw()
        exitButtonText.draw()

    elif game.gameState in (GameState.WATCH_BASIC, GameState.WATCH_ADVANCED):
        game.batch.draw()
        topBanner.draw()
        timerLabel.draw()
        livesLabel.draw()
        scoreLabel.draw()
        stageLabel.draw()
        exitButton.draw()
        exitButtonText.draw()

    elif game.gameState == GameState.GAME_OVER:
        gameoverLabel.draw()
        gameoverscoreLabel.draw()
        restartButton.draw()
        restartButtonText.draw()

#Determining what action to take when the mouse is used
@window.event
def on_mouse_press(x, y, button, modifiers):

#Only mouse left clicks have any effect
    if button != mouse.LEFT:
        return

#Buttons work by displaying a rectangle, and the game state is changed accordingly when a mouse click occurs within the rectangle
    if game.gameState == GameState.MENU:
        if 220 <= x <= 420 and 250 <= y <= 300:
            game.resetGame()
            game.gameState = GameState.SOLO_MODE_MENU
        elif 220 <= x <= 420 and 200 <= y <= 250:
            game.resetGame()
            game.gameState = GameState.TRAINING_MODE_MENU
        elif 220 <= x <= 420 and 150 <= y <= 200:
            game.resetGame()
            game.gameState = GameState.WATCH_MODE_MENU
        elif 220 <= x <= 420 and 100 <= y <= 150:
            game.gameState = GameState.SCORES_MENU

    elif game.gameState == GameState.SOLO_MODE_MENU:
        if 220 <= x <= 420 and 275 <= y <= 325:
            game.resetGame()
            game.gameState = GameState.SOLO_BASIC
        elif 220 <= x <= 420 and 200 <= y <= 250:
            game.resetGame()
            game.gameState = GameState.SOLO_ADVANCED

    elif game.gameState == GameState.TRAINING_MODE_MENU:
        if 220 <= x <= 420 and 275 <= y <= 325:
            game.resetGame()
            game.gameState = GameState.TRAINING_BASIC
        elif 220 <= x <= 420 and 200 <= y <= 250:
            game.resetGame()
            game.gameState = GameState.TRAINING_ADVANCED

    elif game.gameState == GameState.WATCH_MODE_MENU:
        if 220 <= x <= 420 and 275 <= y <= 325:
            game.resetGame()
            game.paddle.x = 580 * random.random()
            game.gameState = GameState.WATCH_BASIC
        elif 220 <= x <= 420 and 200 <= y <= 250:
            game.resetGame()
            game.paddle.x = 580 * random.random()
            game.gameState = GameState.WATCH_ADVANCED

    elif game.gameState in (GameState.SOLO_BASIC,
        GameState.SOLO_ADVANCED,
        GameState.TRAINING_BASIC,
        GameState.TRAINING_ADVANCED,
        GameState.WATCH_BASIC,
        GameState.WATCH_ADVANCED,
        GameState.SCORES_MENU):
#Exit button pressed
        if 540 <= x <= 640 and 480 <= y <= 520:
            game.gameState = GameState.MENU

    elif game.gameState == GameState.GAME_OVER:
        if 220 <= x <= 420 and 200 <= y <= 250:
            game.gameState = GameState.MENU

#The game is updated 60 times a second (frame rate is 60fps)
pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()