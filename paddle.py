# Paddle

from pyglet import shapes

class Paddle:
    def __init__(self, x, y, width, height, batch, window_width):
        self._shape = shapes.Rectangle(x, y, width, height, color=(255, 255, 255), batch=batch)
        self._window_width = window_width
        self._speed = 300.0
#Means that self.x can refer to the x-coordinate, from self.shape.x
    @property
    def x(self):
        return self._shape.x
#Ensures that the paddle is always within the window
    @x.setter
    def x(self, value):
        self._shape.x = max(0, min(self._window_width - self._shape.width, value))

    @property
    def y(self):
        return self._shape.y

    @property
    def width(self):
        return self._shape.width

    @property
    def height(self):
        return self._shape.height

    def move_left(self, dt):
        self.x -= self._speed * dt

    def move_right(self, dt):
        self.x += self._speed * dt