import math
from simpleHeap import SimpleHeap

import pygame

MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50


class AI():
    """
    A simple AI class for controlling a kart in a racing game.

    Attributes:
        kart: The kart object controlled by the AI.
        checkpoints: The checkpoints on the track.
    """

    def __init__(self):
        """
        Initialize the AI with default values.
        """
        self.kart = None
        self.checkpoints = None

    def move(self, string):
        """
        Move the kart based on the current track information.

        Args:
            string (str): The string describing the track.

        Returns:
            dict: A dictionary of keys (UP, DOWN, LEFT, RIGHT) and corresponding boolean values.
        """

        closest_checkpoint = self.find_closest_checkpoint(string)
        next_move = self.get_minimum_valid_neighbor(string, self.kart.position[0], self.kart.position[1],
                                                    closest_checkpoint[0], closest_checkpoint[1])

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

        possible_neighbors = {
            (int(x + i), int(y + j))
            for i in range(-degree_level, degree_level + 1)
            for j in range(-degree_level, degree_level + 1)
            if i != 0 or j != 0
        }
        valid_neighbors = (
            (nx, ny)
            for nx, ny in possible_neighbors
            if self.kart.get_track_element(string, nx, ny)[0].__name__ in ("Road", "Boost", "Checkpoint", "Grass")
        )

        priority_queue = SimpleHeap()

        for nx, ny in valid_neighbors:
            cost = self.calculate_cost(string, nx, ny, c_x, c_y)
            SimpleHeap.push(priority_queue, (cost, (nx, ny)))
        min_valid_neighbor = SimpleHeap.pop(priority_queue)[1]

        return min_valid_neighbor

    def calculate_cost(self, string, nx, ny, c_x, c_y):
        track_element = self.kart.get_track_element(string, nx, ny)[0].__name__
        distance_to_checkpoint = abs(nx - c_x) + abs(ny - c_y)

        if track_element in ("Road", "Boost", "Checkpoint"):
            # Add 1 for road, boost, and checkpoint
            return 1 + distance_to_checkpoint
        elif track_element == "Grass":
            # Add 100 for grass and distance to checkpoint
            return 100 + distance_to_checkpoint

    def find_closest_checkpoint(self, string):
        """
        Find the closest checkpoint with the correct ID to the kart's current position.

        Returns:
            tuple: The position of the closest checkpoint.
        """
        kart_x, kart_y = self.kart.position[0], self.kart.position[1]
        kart_next_checkpoint_id = self.kart.next_checkpoint_id

        checkpoints = []

        char = ''
        if kart_next_checkpoint_id == 0:
            char = 'C'
        elif kart_next_checkpoint_id == 1:
            char = 'D'
        elif kart_next_checkpoint_id == 2:
            char = 'E'
        elif kart_next_checkpoint_id == 3:
            char = 'F'

        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]

        for row_index in range(len(rows)):
            row = rows[row_index]
            for col_index in range(len(row)):
                track_char = row[col_index]
                if track_char == char:
                    checkpoints.append((col_index * BLOCK_SIZE + 0.5 * BLOCK_SIZE,
                                        row_index * BLOCK_SIZE + 0.5 * BLOCK_SIZE))

        closest_checkpoint = None
        min_distance = float('inf')

        for checkpoint in checkpoints:
            c_x, c_y = checkpoint
            distance_to_checkpoint = abs(kart_x - c_x) + abs(kart_y - c_y)

            if distance_to_checkpoint < min_distance:
                min_distance = distance_to_checkpoint
                closest_checkpoint = checkpoint

        return closest_checkpoint

