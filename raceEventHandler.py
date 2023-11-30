class RaceEventHandler:

    def __init__(self):
        self.orientation_from_checkpoint = 0
        self.position_from_checkpoint = (0, 0)

    def handle_checkpoint(self, checkpoint_params, string):
        checkpoint_id = checkpoint_params[0]
        last_checkpoint_id = self.get_last_checkpoint_id(string)

        if checkpoint_id == self.next_checkpoint_id:
            # Last checkpoint
            if checkpoint_id == last_checkpoint_id:
                self.has_finished = True
                return

            # Save the checkpoint, but this is not the last one
            self.position_from_checkpoint = self.position
            self.orientation_from_checkpoint = self.angle
            self.next_checkpoint_id += 1

    def handle_lava(self):
        # Reset from the last checkpoint
        self.last_position = self.position_from_checkpoint
        self.position = self.position_from_checkpoint
        self.last_angle = self.orientation_from_checkpoint
        self.angle = self.orientation_from_checkpoint

    # Returns the last checkpoint ID on the track
    def get_last_checkpoint_id(self, string):
        letters = {'C', 'D', 'E', 'F'}
        count = 0
        seen_letters = set()

        for char in string:
            if char in letters and char not in seen_letters:
                count += 1
                seen_letters.add(char)

        return count - 1