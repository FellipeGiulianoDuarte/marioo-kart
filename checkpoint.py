from trackElement import TrackElement


class Checkpoint(TrackElement):

    def __init__(self, x, y, checkpoint_id):
        super().__init__(x, y, (128, 128, 128))
