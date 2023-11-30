from trackElement import TrackElement


class Grass(TrackElement):

    def __init__(self, x, y):
        super().__init__(x, y, (0, 255, 0))