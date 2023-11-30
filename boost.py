from trackElement import TrackElement


class Boost(TrackElement):

    def __init__(self, x, y):
        super().__init__(x, y, (255, 255, 0))