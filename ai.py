import math

import pygame

MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50


class AI():
    """
    A simple AI class for controlling a kart in a racing game.

    Attributes:
        kart: The kart object controlled by the AI.
        graph: The graph representing the track.
        checkpoints: The checkpoints on the track.
        current_path: The current path the kart is following.
        path_index: The index of the current position in the path.
    """

    def __init__(self):
        """
        Initialize the AI with default values.
        """
        self.kart = None
        self.graph = None
        self.checkpoints = None
        self.current_path = None
        self.path_index = 0

    def move(self, string):
        """
        Move the kart based on the current track information.

        Args:
            string (str): The string describing the track.

        Returns:
            dict: A dictionary of keys (UP, DOWN, LEFT, RIGHT) and corresponding boolean values.
        """

        # Find the position of the next checkpoint
        global char, col_index, row_index
        if self.kart.next_checkpoint_id == 0:
            char = 'C'
        elif self.kart.next_checkpoint_id == 1:
            char = 'D'
        elif self.kart.next_checkpoint_id == 2:
            char = 'E'
        elif self.kart.next_checkpoint_id == 3:
            char = 'F'

        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]

        for row_index in range(len(rows)):
            row = rows[row_index]

            for col_index in range(len(row)):
                track_char = row[col_index]

                if track_char == char:
                    break
            else:
                continue
            break

        next_checkpoint_position = [col_index * BLOCK_SIZE, row_index * BLOCK_SIZE + .5 * BLOCK_SIZE]

        next_move = self.get_minimum_valid_neighbor(string, self.kart.position[0], self.kart.position[1],
                                                    next_checkpoint_position[0], next_checkpoint_position[1])

        if next_move:
            next_point = next_move
            relative_x = next_point[0] - self.kart.position[0]
            relative_y = next_point[1] - self.kart.position[1]
            next_point_angle = math.atan2(relative_y, relative_x)
            relative_angle = (next_point_angle - self.kart.angle + math.pi) % (2 * math.pi) - math.pi

            # Adjust movement based on the relative angle
            if relative_angle > MAX_ANGLE_VELOCITY:
                command = [False, False, False, True]  # Turn right
            elif relative_angle < -MAX_ANGLE_VELOCITY:
                command = [False, False, True, False]  # Turn left
            else:
                command = [True, False, False, False]  # Move forward

            key_list = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
            keys = {key: command[i] for i, key in enumerate(key_list)}
            return keys
        else:
            return {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}

    def get_minimum_valid_neighbor(self, string, x, y, c_x, c_y):
        """
        Get the minimum valid neighbor position based on the current kart position and checkpoint position.

        Args:
            string (str): The string describing the track.
            x (float): The x-coordinate of the kart position.
            y (float): The y-coordinate of the kart position.
            c_x (float): The x-coordinate of the checkpoint position.
            c_y (float): The y-coordinate of the checkpoint position.

        Returns:
            tuple: The coordinates of the minimum valid neighbor position.
        """
        # when you increase this value, the kart performs better but slows the code, I recommend 30
        degree_level = 50
        x, y = int(x), int(y)
        rows = len(string)
        cols = len(string[0])

        # Use a set for possible_neighbors
        possible_neighbors = {
            (int(x + i), int(y + j))
            for i in range(-degree_level, degree_level + 1)
            for j in range(-degree_level, degree_level + 1)
            if i != 0 or j != 0
        }

        # Use a generator expression for valid_neighbors
        valid_neighbors = (
            (nx, ny)
            for nx, ny in possible_neighbors
            if self.kart.get_track_element(string, nx, ny)[0].__name__ in ("Road", "Boost", "Checkpoint")
        )

        # Use min with a key function and provide a default value
        min_valid_neighbor = min(
            valid_neighbors,
            default=None,
            key=lambda neighbor: abs(neighbor[0] - c_x) + abs(neighbor[1] - c_y)
        )

        return min_valid_neighbor
