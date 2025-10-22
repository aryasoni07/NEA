from pyglet import shapes

class Brick:
    def __init__(self, x, y, width, height, batch, color=(100, 200, 255)):
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
            overlapLeft =  ballRight-brickLeft
            overlapRight = brickRight-ballLeft
            overlapTop = ballBottom-brickTop
            overlapBottom = brickBottom-ballTop
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