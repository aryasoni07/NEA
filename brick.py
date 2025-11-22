from pyglet import shapes

class Brick:
    def __init__(self, x, y, width, height, batch, color=(100, 200, 255)):
        self._shape = shapes.Rectangle(x, y, width, height, color=color, batch=batch)
        self._alive = True
    @property
    def x(self):
        return self._shape.x
    
    @property
    def y(self):
        return self._shape.y
    
    @property
    def width(self):
        return self._shape.width
    
    @property
    def height(self):
        return self._shape.height

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, value):
        self._alive = value
        if not value and self._shape is not None:
            self._shape.delete()
            self._shape = None

    def checkCollision(self, ball):
        if not self._alive:
            return False

        nextx = ball.x + ball.width/2 + ball._dx/60
        nexty = ball.y + ball.width/2 + ball._dy/60

        brickLeft = self.x
        brickRight = self.x + self.width
        brickTop = self.y
        brickBottom = self.y + self.height
        ballLeft = nextx
        ballRight = nextx + ball.width
        ballTop = nexty
        ballBottom = nexty + ball.height

        if (ballRight > brickLeft and ballLeft < brickRight and
            ballBottom > brickTop and ballTop < brickBottom):

            overlapLeft = ballRight - brickLeft
            overlapRight = brickRight - ballLeft
            overlapTop = ballBottom - brickTop
            overlapBottom = brickBottom - ballTop

            overlapx = min(overlapLeft, overlapRight)
            overlapy = min(overlapTop, overlapBottom)

            if overlapx < overlapy:
                ball._dx *= -1
            else:
                ball._dy *= -1

            self.alive = False
            return True
        return False