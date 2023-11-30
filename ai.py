import math
import time
import pygame
from boost import Boost
from grass import Grass
from lava import Lava
from road import Road
from checkpoint import Checkpoint
from heapq import heappush, heappop

MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50

class AI():

    def __init__(self):
        self.kart = None
        self.graph = None
        self.checkpoints = None
        self.current_path = None
        self.path_index = 0

    def move(self, string):
        """
        The AI uses A* pathfinding to navigate around obstacles (lava and grass) and reach checkpoints.

        :param string: The string describing the track
        :returns: A dictionary of keys (UP, DOWN, LEFT, RIGHT) and corresponding boolean values
        """

        # Find the position of the next checkpoint
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

        next_move = self.get_minimum_valid_neighbor(string, self.kart.position[0], self.kart.position[1], next_checkpoint_position[0], next_checkpoint_position[1])

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

    def get_minimum_valid_neighbor(self, string, x, y, c_x, c_y):
        # when you increase this value, the kart performs better but slows the code, i recommend 25
        degree_level = 25
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
            if self.kart.get_track_element(string, nx, ny)[0] in (Road, Boost, Checkpoint)
        )

        # Use min with a key function and provide a default value
        min_valid_neighbor = min(
            valid_neighbors,
            default=None,
            key=lambda neighbor: abs(neighbor[0] - c_x) + abs(neighbor[1] - c_y)
        )

        return min_valid_neighbor


