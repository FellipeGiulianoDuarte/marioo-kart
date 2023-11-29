import math
import pygame
from boost import Boost
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
                    # print(f"Found '{char}' at row {row_index}, column {col_index}")
                    break
            else:
                continue
            break

        next_checkpoint_position = [col_index * BLOCK_SIZE, row_index * BLOCK_SIZE + .5 * BLOCK_SIZE]

        # print(f"Next Checkpoint ID: {self.kart.next_checkpoint_id}")
        # print(f"Next Checkpoint Char: {char}")
        # print(f"Next Checkpoint Position: {next_checkpoint_position}")
    
        # Use A* pathfinding to find the shortest path to the next checkpoint
        path = self.a_star_pathfinding(string, self.kart.position, next_checkpoint_position)

        print(f"Path: {path}")
        
        # Determine the angle and movement based on the path
        if path:
            # print(f"Relative Angle: {relative_angle}")
            # print(f"Command: {command}")
            # Calculate angle towards the next point in the path
            next_point = path[0]
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
            # If no path is found, stop moving
            return {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}

    def a_star_pathfinding(self, string, start, goal):
        """
        A* pathfinding algorithm to find the shortest path on the track, avoiding obstacles.

        :param string: The string describing the track
        :param start: The starting position [x, y]
        :param goal: The goal position [x, y]
        :return: A list of points representing the shortest path
        """

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def neighbors(point):
            x, y = point
            possible_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            valid_neighbors = [
                (nx, ny) for nx, ny in possible_neighbors
                if self.kart.get_track_element(string, nx, ny)[0] in (Road, Boost, Checkpoint)
            ]

            # print(f"Point: {point}, Possible Neighbors: {possible_neighbors}")
            for nx, ny in possible_neighbors:
                track_class, _ = self.kart.get_track_element(string, nx, ny)
                # print(f"Neighbor: ({nx}, {ny}), Track Class: {track_class}")

            # print(f"Valid Neighbors: {valid_neighbors}")

            return valid_neighbors

        open_set = []
        closed_set = set()
        start_node = (start, None)
        heappush(open_set, (0, start_node))

        while open_set:
            current_cost, (current, came_from) = heappop(open_set)
            
            # print(f"Current: {current}, Goal: {goal}")

            if current == goal:
                path = [current]
                while came_from:
                    path.append(came_from[0])
                    current, came_from = came_from
                return path[::-1]

            closed_set.add(tuple(current))

            for neighbor in neighbors(current):
                if neighbor not in closed_set:
                    new_cost = current_cost + 1
                    heuristic_cost = new_cost + heuristic(neighbor, goal)
                    heappush(open_set, (heuristic_cost, (neighbor, current)))

        return []
