from trackElement import TrackElement


class Lava(TrackElement):

    def __init__(self, x, y):
        super().__init__(x, y, (255, 0, 0))