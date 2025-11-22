from pyglet import shapes

class Ball:
    def __init__(self, x, y, size, batch):
        self._shape = shapes.Rectangle(x, y, size, size, color=(255, 0, 0), batch=batch)
        self._dx = 200.0
        self._dy = 200.0

    @property
    def x(self):
        return self._shape.x

    @x.setter
    def x(self, value):
        self._shape.x = value

    @property
    def y(self):
        return self._shape.y

    @y.setter
    def y(self, value):
        self._shape.y = value

    @property
    def width(self):
        return self._shape.width

    @property
    def height(self):
        return self._shape.height

    @property
    def color(self):
        return self._shape.color

    @color.setter
    def color(self, value):
        self._shape.color = value

    def update_position(self, dt):
        self.x += self._dx * dt
        self.y += self._dy * dt