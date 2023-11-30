from trackElement import TrackElement


class Road(TrackElement):

    def __init__(self, x, y):
        super().__init__(x, y, (0, 0, 0))